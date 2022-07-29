from functools import partial
import torch
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from tqdm import tqdm
import argparse
import csv
import faiss

import os
import os.path as osp
import sys
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader

import torch_geometric.transforms as T
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import normalize
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon
from torch_geometric.nn import GCNConv, GATConv
from eval import label_classification
from simsiam import Encoder, EncoderGAT, ModelSIM, prediction_MLP, projection_MLP
from data_aug import drop_feature_guided, dropout_edge_guided


def train(model, optimizer, x, edge_index,  vote, nodevote):
    model.train()
    optimizer.zero_grad()

    edge_index_1, edge_index_2 = dropout_edge_guided(edge_index, vote, edge_p = edge_p, lambda_edge= lambda_edge)
    x_1, x_2 = drop_feature_guided(x, nodevote, node_p=node_p, lambda_node= lambda_node)
    
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2)
    loss.backward()
    optimizer.step()

    return loss.item()

def test(model, x, edge_index, y, final=False):
    model.eval()
    z = model(x, edge_index)

    accuracy = label_classification(z, y, ratio=0.1)
    return accuracy



def k_near_select(emb):
    
    emb = emb.detach().cpu().numpy()
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb) 
    _, I = index.search(emb, k_near)
    node_avg = 0
    for i in range(1, k_near):
        node_avg = node_avg + emb[I[:,i],:]
    node_avg = normalize(node_avg, axis =1)
    emb = emb * node_avg
    emb = np.maximum(0, emb)
    
    return emb

def get_expl(model):
    
    emb = model(data.x, data.edge_index)
    with torch.no_grad():
        emb = k_near_select(emb)
        nodevote = emb.sum(1)
        nodevote = nodevote - nodevote.min()
        nodevote = nodevote / nodevote.max()
        nodevote_list = nodevote.tolist()
        
        edge = data.edge_index.cpu().tolist()
        vote = [nodevote_list[x]+nodevote_list[y] for x, y  in zip(*edge)]
        vote = torch.tensor(vote).to(device)
        vote = vote - vote.min()
        vote = vote / vote.max()
    
    return vote, torch.tensor(nodevote).to(device)
    
def gridtrain():
    
    encoder = model_encoder(dataset.num_features, coder_hid_num, num_hidden, activation,
                            base_model=base_model, k=num_layers).to(device)
    prejector = projection_MLP(num_hidden, num_hidden, num_hidden)
    predictor = prediction_MLP(num_hidden, num_mid_hidden, num_hidden) 

    model = ModelSIM(encoder, prejector, predictor).to(device)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = t()
    prev = start

   

    for epoch in tqdm(range(1, num_epochs + 1)):

        #adjust_learning_rate(optimizer,epoch)

        if epoch < int(args.start_e*num_epochs):
            loss = train(model, optimizer, data.x, data.edge_index, vote = None, nodevote= None)
        else:
            vote, nodevote = get_expl(model)
            loss = train(model, optimizer, data.x, data.edge_index, vote, nodevote)
        
        now = t()
        #print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, 'f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now

    print("=== Final ===")
    acc = test(model, data.x, data.edge_index, data.y, final=True)
    acc_mean = acc.get('F1Mi').get('mean')
    acc_std = acc.get('F1Mi').get('std')
    
    return acc_mean, acc_std

def get_dataset(path, name):
            assert name in ['Cora', 'CiteSeer', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                            'Amazon-Computers', 'Amazon-Photo']
            name = 'dblp' if name == 'DBLP' else name
            root_path = osp.expanduser('~/datasets')

            if name == 'Coauthor-CS':
                return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

            if name == 'Coauthor-Phy':
                return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

            if name == 'WikiCS':
                return WikiCS(root=path, transform=T.NormalizeFeatures())

            if name == 'Amazon-Computers':
                return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())

            if name == 'Amazon-Photo':
                return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())

            return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures())
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--config', type=str, default='./simsiam_self_guided.yaml')
    parser.add_argument('--model', type=str, default='GAT')
    parser.add_argument('--lrdec_1', type=float, default=0)
    parser.add_argument('--lrdec_2', type=int, default=0)
    parser.add_argument('--gpu_num', type=str, default='cuda')
    parser.add_argument('--runtimes', type=int, default =5)
    parser.add_argument('--start_e', type=float, default =0.3)

    args = parser.parse_args()

    device = torch.device(args.gpu_num if torch.cuda.is_available() else 'cpu')
   
    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset][args.model]
        
    
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    coder_hid_num = config['coder_hid_num']
    num_mid_hidden = config['num_mid_hidden']
    activation = ({'relu': F.relu, 'elu':F.elu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv, 'GATConv': GATConv})[config['base_model']]
    model_encoder =  ({'GCN': Encoder,'GAT':EncoderGAT})[args.model]
    num_layers = config['num_layers']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']
    lambda_edge = config['lambda_edge']
    lambda_node = config['lambda_node']
    edge_p = config['edge_p']
    node_p = config['node_p']
    k_near = config['k_near']


    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]
    data = data.to(device)
    
    for i in range(args.runtimes):
        gridtrain()
                
