import torch
import torch.nn as nn
import torch.nn.functional as F
from model.function import *
import math


class TSRec(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, hidden_dim, gru_dim, max_seq_len, max_repeat_seq_len,
                 max_time_interval, max_index_gap, max_period, negative_sample_num, negative_sample_test_num,
                 dropout_rate=0.5):
        super(TSRec, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.gru_dim = gru_dim
        self.max_seq_len = max_seq_len
        self.stdv = 1.0 / math.sqrt(self.embed_dim)
        self.dropout_rate = dropout_rate
        self.w = 20
        self.user_embedding = nn.Embedding(n_users+1, embed_dim, padding_idx=0, max_norm=1.5)
        self.embedding = nn.Embedding(n_items + 1, embed_dim, padding_idx=0, max_norm=1.5)
        self.time_embedding = nn.Embedding(max_time_interval + 1, embed_dim, padding_idx=0, max_norm=1.5)
        self.period_embedding = nn.Embedding(max_time_interval + 1, embed_dim, padding_idx=0, max_norm=1.5)
        self.ind_embedding = nn.Embedding(max_index_gap + 1, embed_dim, padding_idx=0, max_norm=1.5)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.global_avg_pool = nn.AvgPool1d(max_seq_len)
        self.global_max_pool = nn.MaxPool1d(max_seq_len)
        self.s1_rnn = nn.GRU(embed_dim, gru_dim,
                             num_layers=2, dropout=dropout_rate, bidirectional=True, batch_first=True)
        self.s2_rnn = nn.GRU(embed_dim, gru_dim,
                             num_layers=2, dropout=dropout_rate, bidirectional=True, batch_first=True)
        self.cnn1 = Inception1(gru_dim * 2)
        self.cnn2 = Inception2(gru_dim * 2)
        self.cnn3 = Inception3(gru_dim * 2)
        self.fc_sub = FCSubtract((max_seq_len - 1) * 2 * 3 + gru_dim * 2 * 2, hidden_dim)
        self.fc_mul = FCMultiply((max_seq_len - 1) * 2 * 3 + gru_dim * 2 * 2, hidden_dim)
        self.gate_layer = nn.Sequential(
            nn.Linear(((max_seq_len - 1) * 2 * 3 + gru_dim * 2 * 2 + hidden_dim) * 2,
                      (max_seq_len - 1) * 2 * 3 + gru_dim * 2 * 2),
            nn.Sigmoid()
        )
        self.feed_forward_network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.period_enc = Inception_2d(self.embed_dim, self.hidden_dim)
        self.linear_query_i = nn.Linear(self.embed_dim, self.hidden_dim)
        self.linear_key_i = nn.Linear(self.embed_dim, self.hidden_dim)
        self.linear_value_i = nn.Linear(self.embed_dim, self.hidden_dim)
        self.linear_query_t = nn.Linear(self.embed_dim, self.hidden_dim)
        self.linear_key_t = nn.Linear(self.embed_dim, self.hidden_dim)
        self.linear_value_t = nn.Linear(self.embed_dim, self.hidden_dim)
        self.a_1 = nn.Linear(2 * self.gru_dim, self.hidden_dim, bias=False)
        self.a_2 = nn.Linear(4 * self.gru_dim, self.hidden_dim, bias=False)
        self.b_1 = nn.Linear(2 * self.gru_dim, self.hidden_dim, bias=False)
        self.b_2 = nn.Linear(4 * self.gru_dim, self.hidden_dim, bias=False)
        self.c_1 = nn.Linear(2 * self.gru_dim, self.hidden_dim, bias=False)
        self.c_2 = nn.Linear(4 * self.gru_dim, self.hidden_dim, bias=False)
        self.v_t = nn.Linear(self.hidden_dim, 1, bias=False)
        self.v_t_explore = nn.Linear(self.hidden_dim, 1, bias=False)
        self.v_t_mode = nn.Linear(self.hidden_dim, 1, bias=False)
        self.explore_repeat_mode = nn.Linear(gru_dim * 4 + self.hidden_dim * 2, 2)
        self.fc_layer = nn.Sequential(nn.Linear(gru_dim * 4,
                                                self.hidden_dim), nn.Sigmoid())

        self.mlp_layer = nn.Sequential(
            nn.Linear((max_seq_len - 1) * 2 * 3 + gru_dim * 2 * 2 + gru_dim*4 + self.hidden_dim * 5,
                      self.embed_dim*2), nn.Sigmoid())
    def build_map(self, b_map, max=None):
        batch_size, b_len = b_map.size()
        if max is None:
            max = b_map.max() + 1
        b_map_ = torch.zeros(batch_size, b_len, max)
        b_map_.scatter_(2, b_map.unsqueeze(2), 1.)
        # b_map_[:, :, 0] = 0.
        b_map_.requires_grad = False
        return b_map_

    def soft_align(self, input_1, input_2):
        attention = torch.bmm(input_1, input_2.permute(0, 2, 1))
        w_att_1 = F.softmax(attention, dim=1)
        w_att_2 = F.softmax(attention, dim=2).permute(0, 2, 1)
        in1_aligned = torch.bmm(w_att_1, input_1)
        in2_aligned = torch.bmm(w_att_2, input_2)
        return in1_aligned, in2_aligned

    def temper_attention_index(self, q, k, v, mask=None):
        # qw,kw,vw: [1, hidden_dim] [batch_size, max_len, hidden_dim] [batch_size, max_len, hidden_dim]
        qw = self.linear_query_i(q)
        kw = self.linear_key_i(k)
        vw = self.linear_value_i(v)
        # [batch_size, max_len, hidden_dim]*[batch_size,hidden_dim,1]
        atten_score = torch.bmm(kw, qw.transpose(1, 2)).squeeze(2) / math.sqrt(self.hidden_dim)
        if mask is not None:
            attn = atten_score.masked_fill(mask == 0, -float('inf'))
        norm_atten_score = atten_score.softmax(dim=-1).unsqueeze(1)

        out = torch.bmm(norm_atten_score, vw)
        return out

    def temper_attention_time(self, q, k, v, mask=None):
        # qw,kw,vw: [1, hidden_dim] [batch_size, max_len, hidden_dim] [batch_size, max_len, hidden_dim]
        qw = self.linear_query_t(q)
        kw = self.linear_key_t(k)
        vw = self.linear_value_t(v)
        # [batch_size, max_len, hidden_dim]*[batch_size,hidden_dim,1]
        atten_score = torch.bmm(kw, qw.transpose(1, 2)).squeeze(2) / math.sqrt(self.hidden_dim)
        if mask is not None:
            attn = atten_score.masked_fill(mask == 0, -float('inf'))
        norm_atten_score = atten_score.softmax(dim=-1).unsqueeze(1)

        out = torch.bmm(norm_atten_score, vw)
        return out


    def forward(self, uid, tar, s1, s2, t1, t2, i1, i2, rep_matrix):
        self.batch_size = s2.size(0)
        self.h_rep = rep_matrix.size(1)
        self.w_rep = rep_matrix.size(2)
        mask1 = s1.ne(0)
        mask2 = s2.ne(0)
        len_1 = mask1.float().sum(dim=-1).long()
        len_1 = torch.where(len_1 < 1, 1, len_1).long()
        len_2 = mask2.float().sum(dim=-1).long()
        ind_mask = i1.ne(0)

        s1_embed = self.embedding(s1)
        s2_embed = self.embedding(s2)
        tar_emb = self.embedding(tar)
        user_emb = self.user_embedding(uid.squeeze())

        # x1, x2: [batch_size , max_len, embed_dim]
        x1 = self.dropout(s1_embed)
        x2 = self.dropout(s2_embed)
        i1_embed = self.dropout(self.ind_embedding(i1))
        i2_embed = self.dropout(self.ind_embedding(i2))
        if len(i2_embed.size()) <= 2:
            i2_embed = i2_embed.unsqueeze(1)
        t1_embed = self.dropout(self.time_embedding(t1))
        t2_embed = self.dropout(self.time_embedding(t2))

        rep_matrix_emb = self.dropout(self.period_embedding(rep_matrix))
        rep_matrix_permute = rep_matrix_emb.reshape(self.batch_size, -1, self.embed_dim
                                                    ).permute(0, 2, 1).reshape(
            self.batch_size, self.embed_dim, self.h_rep, self.w_rep)
        o_rep = self.period_enc(rep_matrix_permute)


        o_time = self.temper_attention_time(t2_embed, t1_embed, t1_embed, mask=ind_mask)
        o_time = o_time.squeeze(dim=1)

        o_ind = self.temper_attention_index(i2_embed, i1_embed, i1_embed, mask=ind_mask)
        o_ind = o_ind.squeeze(dim=1)

        # s1_output,s2_output: [batch_size, max_len, gru_dim*2]

        s1_output, s1_hn = gru_forward(self.s1_rnn, x1, len_1, batch_first=True)
        s2_output, s2_hn = gru_forward(self.s2_rnn, x2, len_2, batch_first=True)

        s1_output = self.dropout(s1_output)
        s2_output = self.dropout(s2_output)
        # s2_hn:[2*2,batch_size, gru_dim]
        s2_hn = self.dropout(s2_hn)
        # s1_encoded_permute,s2_encoded_permute: [batch_size, gru_dim*2, max_len]
        s1_encoded_permute = s1_output.permute(0, 2, 1)
        s2_encoded_permute = s2_output.permute(0, 2, 1)
        # s2_hn_permute: [batch_size, gru_dim*4]
        s2_hn_permute = s2_hn.reshape(-1, 4 * self.gru_dim)
        # s1_aligned,s2_aligned: [batch_size, max_len, gru_dim*2]
        s1_aligned, s2_aligned = self.soft_align(s1_output, s2_output)

        # s1_att_mean, s1_att_max, s2_att_mean, s2_att_max: [batch_size, gru_dim*2]
        s1_att_mean, s1_att_max = mean_max(s1_aligned)
        s2_att_mean, s2_att_max = mean_max(s2_aligned)

        # s1_cnn, s2_cnn: [batch_size, (max_len-1)*2*3]
        s1_cnn = torch.cat((self.cnn1(s1_encoded_permute), self.cnn2(
            s1_encoded_permute), self.cnn3(s1_encoded_permute)), dim=1)
        s2_cnn = torch.cat((self.cnn1(s2_encoded_permute), self.cnn2(
            s2_encoded_permute), self.cnn3(s2_encoded_permute)), dim=1)

        # o1_cat,o2_cat: [batch_size, (max_len-1)*2*3+gru_dim*2*2]
        o1_cat = torch.cat(
            (s1_att_mean, s1_cnn, s1_att_max), dim=1)
        o2_cat = torch.cat(
            (s2_att_mean, s2_cnn, s2_att_max), dim=1)

        # o_sub,o_mul: [batch_size, hidden_dim]
        o_sub = self.fc_sub(o1_cat, o2_cat)
        o_mul = self.fc_mul(o1_cat, o2_cat)

        # fusion layer
        # o_input,: [batch_size, ((max_len-1)*2*3+gru_dim*2*2)*2+hidden_dim*2]
        # o_gate: [batch_size, (max_len-1)*2*3+gru_dim*2*2]
        # o_fuse: [batch_size, (max_len-1)*2*3+gru_dim*2*2]
        o_input = torch.cat((o1_cat, o2_cat, o_sub, o_mul), dim=1)
        o_gate = self.gate_layer(o_input)
        o_fuse = o_gate * o1_cat + (1 - o_gate) * o2_cat

        # s2 feed forward attention
        q1 = self.a_1(s2_output.contiguous().view(-1, self.gru_dim * 2)).reshape(self.batch_size, self.max_seq_len,
                                                                                 self.hidden_dim)
        q2 = self.a_2(s2_hn_permute)

        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q2_masked = mask2.unsqueeze(2).expand_as(q1) * q2_expand

        alpha = self.v_t(torch.sigmoid(q1 + q2_masked).view(-1, self.hidden_dim)).view(mask2.size())
        o_s2 = torch.sum(alpha.unsqueeze(2).expand_as(s2_output) * s2_output, 1)

        output = torch.cat([o_s2, s2_hn_permute, o_fuse, o_rep, o_time, o_ind,user_emb], dim=-1)
        # output: [batch_size, hidden_dim]
        output = self.mlp_layer(output)
        p = (output * tar_emb).sum(dim=-1)

        return p

