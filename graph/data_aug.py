#includes four kinds of data augmentation
import torch
import torch.nn as nn
import numpy as np

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def filter_adj(row, col, mask):
    return row[mask], col[mask]


def dropout_edg(edge_index,  p=0.5):

    row, col, vote = edge_index

    
    mask = torch.gt(vote,p).to(torch.bool)

    row, col = filter_adj(row, col, mask)

    
    edge_index = torch.stack([row, col], dim=0)

    return edge_index.long()


def dropout_edge_guided(edge_index, vote = None, edge_p=0.2, lambda_edge= -1):
    """
    generate two positive pairs
    """
    row, col = edge_index
    if vote is None:
        vote = torch.zeros(edge_index.shape[1]).to(edge_index.device)
        new_vote = vote + edge_p
        mask1 = torch.bernoulli(new_vote).to(torch.bool)
        mask2 = torch.bernoulli(new_vote).to(torch.bool)
        row1, col1 = filter_adj(row, col, mask1)
        edge_index1 = torch.stack([row1, col1], dim=0)
        row2, col2 = filter_adj(row, col, mask2)
        edge_index2 = torch.stack([row2, col2], dim=0)
    else:  
        vote = torch.clamp(vote, min = 0, max = 1)
        vote_threshold = vote.mean()+ lambda_edge*vote.std()
        vote[vote>vote_threshold]= 1
        mask1 = torch.bernoulli(vote).to(torch.bool)
        mask2 = ~mask1 
        mask2[vote>vote_threshold] = True
        row1, col1 = filter_adj(row, col, mask1)
        edge_index1 = torch.stack([row1, col1], dim=0)
        row2, col2 = filter_adj(row, col, mask2)
        edge_index2 = torch.stack([row2, col2], dim=0)

    return edge_index1.long(), edge_index2.long()

def drop_feature_guided(x, nodevote, node_p=0.2, lambda_node=-2):
    ''''
    This is  the updated version using the torch.bernoulli
    '''
    
    if  nodevote is None:
        drop_mask1 = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < node_p
        drop_mask2 = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < node_p
        x1 = x.clone()
        x2 = x.clone()
        x1[:, drop_mask1] = 0
        x2[:, drop_mask2] = 0
    else:
        new_vote = nodevote 
        vote_threshold = new_vote.mean()+ lambda_node*new_vote.std()
        clamp_mask = torch.clamp(new_vote, min = 0, max = 1)
        clamp_mask[new_vote > vote_threshold] = 1
        
        vote_matr= clamp_mask.repeat(x.shape[1], 1)
        mask1 = torch.bernoulli(vote_matr).to(torch.bool)
        mask2 = ~mask1 
        mask2[:,new_vote > vote_threshold] = True
        
        
        x1 = x*mask1.T
        x2 = x*mask2.T

    return x1, x2
