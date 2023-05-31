import torch


def get_HR(indices, k):
    """
    Calculates the HR score for the given predictions and targets

    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        HR (float): the HR score
    """
    # print('indices', indices.size())
    # print('indices', indices)
    ranks = indices.argsort(dim=-1, descending=True).argsort(dim=-1)[:, 0]
    zero = torch.zeros_like(ranks)
    one = torch.ones_like(ranks)
    ranks= torch.where(ranks<k, one, zero)
    HR = ranks.sum().data/ranks.size(0)
    return HR.item()


def get_mrr(indices, k):
    """
    Calculates the MRR score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        mrr (float): the mrr score
    """
    ranks = indices.argsort(dim=-1, descending=True).argsort(dim=-1)[:, 0]
    ranks = ranks.float()
    zero = torch.zeros_like(ranks)
    rranks = torch.where(ranks<k, 1.0/(ranks+1), zero)
    mrr = torch.sum(rranks).data / rranks.size(0)
    return mrr.item()

def get_ndcg(indices, k):
    """
    Calculates the MRR score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        mrr (float): the mrr score
    """
    ranks = indices.argsort(dim=-1, descending=True).argsort(dim=-1)[:, 0]
    ranks = ranks.float()
    zero = torch.zeros_like(ranks)
    rranks = torch.where(ranks<k, torch.reciprocal(torch.log2(ranks+2)), zero)
    ndcg = torch.sum(rranks).data / rranks.size(0)
    return ndcg.item()


def evaluate(indices, k):
    """
    Evaluates the model using HR@K, MRR@K scores.

    Args:
        logits (B,L): torch.LongTensor. The predicted logit lists for the next items.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        HR (float): the HR score
        mrr (float): the mrr score
    """

    HR = get_HR(indices, k)
    mrr = get_mrr(indices, k)
    ndcg = get_ndcg(indices,k)
    return HR, mrr, ndcg


