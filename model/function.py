import torch
import torch.nn as nn
import torch.nn.functional as F


def gather_indexes( output, gather_index):
    """Gathers the vectors at the specific positions over a minibatch"""
    gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
    output_tensor = output.gather(dim=1, index=gather_index)
    return output_tensor.squeeze(1)
def gru_forward(gru, input, lengths, batch_first=True):
    gru.flatten_parameters()
    input_lengths, perm = torch.sort(lengths, descending=True)

    input = input[perm]

    total_length=input.size(1)
    if not batch_first:
        input = input.transpose(0, 1)  # B x L x N -> L x B x N
    input_lengths = input_lengths.cpu()
    packed = torch.nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first)

    outputs, state = gru(packed)
    outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=batch_first, total_length=total_length)  # unpack (back to padded)

    _, perm = torch.sort(perm, descending=False)
    if not batch_first:
        outputs = outputs.transpose(0, 1)
    outputs=outputs[perm]
    state = state.transpose(0, 1)[perm]

    return outputs, state

def build_map(b_map, max=None):
    batch_size, b_len = b_map.size()
    if max is None:
        max=b_map.max() + 1
    if torch.cuda.is_available():
        b_map_ = torch.cuda.FloatTensor(batch_size, b_len, max).fill_(0)
    else:
        b_map_ = torch.zeros(batch_size, b_len, max)
    b_map_.scatter_(2, b_map.unsqueeze(2), 1.)
    # b_map_[:, :, 0] = 0.
    b_map_.requires_grad=False
    return b_map_
class BilinearAttention(nn.Module):
    def __init__(self, query_size, key_size, hidden_size):
        super().__init__()
        self.linear_key = nn.Linear(key_size, hidden_size, bias=False)
        self.linear_query = nn.Linear(query_size, hidden_size, bias=True)
        self.linear_value = nn.Linear(key_size,hidden_size,bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.hidden_size=hidden_size

    def score(self, query, key, softmax_dim=-1, mask=None):
        attn=self.matching(query, key, mask)

        norm_attn = F.softmax(attn, dim=softmax_dim)

        if mask is not None:
            norm_attn = norm_attn.masked_fill(~mask, 0)

        return attn, norm_attn


    def matching(self, query, key, mask=None):
        '''
        :param query: [batch_size, *, query_seq_len, query_size]
        :param key: [batch_size, *, key_seq_len, key_size]
        :param mask: [batch_size, *, query_seq_len, key_seq_len]
        :return: [batch_size, *, query_seq_len, key_seq_len]
        '''
        wq = self.linear_query(query)
        wq = wq.unsqueeze(-2)

        uh = self.linear_key(key)
        uh = uh.unsqueeze(-3)

        wuc = wq + uh

        wquh = torch.tanh(wuc)

        attn = self.v(wquh).squeeze(-1)

        if mask is not None:
            attn = attn.masked_fill(~mask, -float('inf'))

        return attn

    def forward(self, query, key, value, mask=None):
        '''
        :param query: [batch_size, *, query_seq_len, query_size]
        :param key: [batch_size, *, key_seq_len, key_size]
        :param value: [batch_size, *, value_seq_len=key_seq_len, value_size]
        :param mask: [batch_size, *, query_seq_len, key_seq_len]
        :return: [batch_size, *, query_seq_len, value_size]
        '''

        attn, norm_attn = self.score(query, key, mask=mask)
        value = self.linear_value(value)
        h = torch.bmm(norm_attn.view(-1, norm_attn.size(-2), norm_attn.size(-1)), value.view(-1, value.size(-2), value.size(-1)))

        return h.view(list(value.size())[:-2]+[norm_attn.size(-2), -1]), attn, norm_attn
class FCSubtract(nn.Module):
    def __init__(self, D_in, D_out):
        super(FCSubtract, self).__init__()
        self.dense = nn.Linear(D_in, D_out)

    def forward(self, input_1, input_2):
        res_sub = torch.sub(input_1, input_2)
        res_sub_mul = torch.mul(res_sub, res_sub)
        out = self.dense(res_sub_mul)
        return F.relu(out)


class FCMultiply(nn.Module):
    def __init__(self, D_in, D_out):
        super(FCMultiply, self).__init__()
        self.dense = nn.Linear(D_in, D_out)

    def forward(self, input_1, input_2):
        res_mul = torch.mul(input_1, input_2)
        out = self.dense(res_mul)
        return F.relu(out)
class Inception1(nn.Module):
    def __init__(self, input_dim, conv_dim=64):
        super(Inception1, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=2),
            nn.ReLU()
        )
        self.global_avg_pool = nn.AvgPool1d(input_dim)
        self.global_max_pool = nn.MaxPool1d(input_dim)

    def forward(self, x):
        x = self.cnn(x)
        avg_pool, max_pool = mean_max(x)
        res = torch.cat((avg_pool, max_pool), dim=1)
        # print('inception 1', res.size())
        return res


class Inception2(nn.Module):
    def __init__(self, input_dim, conv_dim=64):
        super(Inception2, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.cnn(x)
        avg_pool, max_pool = mean_max(x)
        res = torch.cat((avg_pool, max_pool), dim=1)
        # print('inception 2',res.size())
        return res


class Inception3(nn.Module):
    def __init__(self, input_dim, conv_dim=64):
        super(Inception3, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=3),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.cnn(x)
        avg_pool, max_pool = mean_max(x)
        res = torch.cat((avg_pool, max_pool), dim=1)
        # print('inception 3', res.size())
        return res
class Inception_2d(nn.Module):
    def __init__(self, input_dim, conv_dim):
        super(Inception_2d, self).__init__()
        self.conv_dim= conv_dim
        self.cnn1 = nn.Sequential(
            nn.Conv2d(input_dim, conv_dim, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(input_dim, conv_dim, kernel_size=(1, 3)),
            nn.ReLU()
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(input_dim, conv_dim, kernel_size=(3, 1)),
            nn.ReLU()
        )
        self.fc = nn.Linear(conv_dim*6, conv_dim)
    def mean_max_pooling(self, x):
        return torch.mean(x, dim=-1), torch.max(x, dim=-1)[0]
    def forward(self, x):
        self.batch_size = x.size(0)
        x1 = self.cnn1(x).reshape(self.batch_size, self.conv_dim, -1)
        x2 = self.cnn2(x).reshape(self.batch_size, self.conv_dim, -1)
        x3 = self.cnn3(x).reshape(self.batch_size, self.conv_dim, -1)

        x1_mean, x1_max = self.mean_max_pooling(x1)
        x2_mean, x2_max = self.mean_max_pooling(x2)
        x3_mean, x3_max = self.mean_max_pooling(x3)
        # print(x1_mean.size())
        out = torch.cat([x1_mean, x1_max, x2_mean, x2_max, x3_mean, x3_max], dim=-1)
        out = self.fc(out)
        return out

def mean_max(x):
    return torch.mean(x, dim=1), torch.max(x, dim=1)[0]