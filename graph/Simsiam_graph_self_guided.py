import torch
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import argparse
import os


import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader


import torch_geometric.transforms as T
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import TUDataset
from svm_eval import svc_classify
from graphsimsiam import ModelSIM, prediction_MLP, projection_MLP, EncoderGIN
from data_aug import drop_feature_guided, dropout_edge_guided
from sklearn import preprocessing


def train(model, optimizer, dataloader, epoch):
    model.train()
    optimizer.zero_grad()
    for data in dataloader:
        data = data.to(device)
        if epoch < int(args.start_e*num_epochs):
            vote = None
            nodevote = None
        else:
            vote, nodevote = get_expl(model, data)
        
        edge_index_1, edge_index_2 = dropout_edge_guided(data.edge_index, vote, edge_p = edge_p, lambda_edge= lambda_edge)
        
        if data.x is None:
            if epoch > int(args.start_e*num_epochs):
                data.x = torch.reshape(nodevote, (-1,1)).to(device)
            else:
                data.x = torch.ones((data.batch.shape[0], 1)).to(device)

        drop_feature = drop_feature_guided
        x_1, x_2 = drop_feature(data.x, nodevote, node_p=node_p, lambda_node= lambda_node)   
        z1 = model(x_1, edge_index_1, data.batch)
        z2 = model(x_2, edge_index_2, data.batch)

        loss = model.loss(z1, z2)
        loss.backward()
        optimizer.step()

    return loss.item()


def get_expl(model,data):
    
    if data.x is None: 
        data.x = torch.ones((data.batch.shape[0], 1)).to(device)
    nodevote = model.get_emb_avg(data.x, data.edge_index, data.batch, k_near)
    with torch.no_grad():
        nodevote = nodevote - nodevote.min()
        nodevote = nodevote / nodevote.max()
        nodevote_list = nodevote.tolist()
        
        edge = data.edge_index.cpu().tolist()
        vote = [nodevote_list[x]+nodevote_list[y] for x, y  in zip(*edge)]
        vote = torch.tensor(vote).to(device)
        vote = vote - vote.min()
        vote = vote / vote.max()
    return vote, nodevote
    

def test(model: ModelSIM, dataloader):
    model.eval()
    F1mi =[]
    F1ma =[]
    for data in dataloader:   
        data = data.to(device)
        embeddings = model(data.x, data.edge_index, data.batch)
        labels = preprocessing.LabelEncoder().fit_transform(data.y.detach().cpu().numpy())
        x = embeddings.detach().cpu().numpy()
        y = labels
        ac1, ac2 = svc_classify(x,y)
        F1mi.append(ac1)
        F1ma.append(ac2)
    F1mi = np.array(F1mi)
    F1ma = np.array(F1ma)
    return np.mean(F1mi),np.std(F1mi), np.mean(F1ma), np.std(F1ma)

    

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (args.lrdec_1 ** (epoch // args.lrdec_2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def gridtrain():
    

    if args.dataset in ['COLLAB','REDDIT-BINARY','REDDIT-MULTI-5K','IMDB-BINARY','IMDB-MULTI']:
        encoder = model_encoder(1, num_hidden, num_layers)     
    else:
        encoder = model_encoder(dataset.num_features, num_hidden, num_layers)
    prejector = projection_MLP(num_hidden*num_layers, num_hidden, num_hidden) #in the encoder part, the return concatenate every layers embedding together
    predictor = prediction_MLP(num_hidden, num_mid_hidden, num_hidden) 

    model = ModelSIM(encoder, prejector, predictor).to(device)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)    
    
    start = t()
    prev = start

    for epoch in tqdm(range(1, num_epochs + 1)):

        adjust_learning_rate(optimizer,epoch)

        loss = train(model, optimizer, data_loader, epoch)
        
        now = t()
        #print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, 'f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now

    #print("=== Final ===")
    F1Mi, sF1Mi, F1Ma,sF1Ma = test(model, data_loader_eval)
    print( f'F1Mi:{F1Mi:.4f}+{sF1Mi:.4f},F1Ma:{F1Ma:.4f}+{sF1Ma:.4f}')
    return F1Mi

    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='COLLAB')
    parser.add_argument('--config', type=str, default='./config_defaults_simsiam_guided.yaml')
    parser.add_argument('--lrdec_1', type=float, default=0.8)
    parser.add_argument('--lrdec_2', type=int, default=200)
    parser.add_argument('--model', type = str, default='GIN')
    parser.add_argument('--num_gpu',type = str, default = 'cuda')
    parser.add_argument('--runtimes',type = int, default = 5)
    parser.add_argument('--start_e', type=float, default =0.3)
    args = parser.parse_args()
    

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset][args.model]
    

    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_layers = config['num_layers']
    num_mid_hidden = config['num_mid_hidden']
    model_encoder =  ({'GIN':EncoderGIN})[args.model]
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']
    batch_size= config['batch_size']
    lambda_edge = config['lambda_edge']
    lambda_node = config['lambda_node']
    edge_p = config['edge_p']
    node_p = config['node_p']
    k_near = config['k_near']

    
    device = torch.device(args.num_gpu if torch.cuda.is_available() else 'cpu')

    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    dataset = TUDataset(path, args.dataset)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    data_loader_eval = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for i in range(args.runtimes):
        F1Mi = gridtrain()
        


   
       

    

