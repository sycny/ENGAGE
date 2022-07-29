import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv,global_add_pool, global_mean_pool,global_max_pool
from torch.nn import Sequential, Linear, ReLU
import faiss
from time import perf_counter as t


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2, head = 36):
        super(Encoder, self).__init__()
        self.base_model = base_model

        self.k = k
        self.conv = [base_model(in_channels, 8 * out_channels)]
        for _ in range(1, k-1):
             self.conv.append(base_model(8 * out_channels, 8 * out_channels))
        self.conv.append(base_model(8 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(edge_index.device)
            
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        x = global_mean_pool(x, batch)
        return x

class EncoderGIN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(EncoderGIN, self).__init__()

        self.num_gc_layers = num_gc_layers

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):

            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)
            
    def embedding(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(edge_index.device)
        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

        return  xs
    
    def forward(self, x, edge_index, batch):
        
        
        xs = self.embedding(x, edge_index, batch)
        xpool = [global_add_pool(x, batch) for x in xs] 
        x = torch.cat(xpool, 1) 

        return x

class EncoderGAT(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GATConv, k: int = 1, head = 36):
        super(EncoderGAT, self).__init__()
        self.base_model = base_model

        self.conv1 = base_model(in_channels, head, heads=head, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = base_model(head * head, out_channels, heads=1, concat=False,
                             dropout=0.6)

        self.activation = activation
        #self.dropout = F.dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.activation(self.conv1(x, edge_index))
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.activation(self.conv2(x, edge_index))
        
        return x
  

class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 2
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 

class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 


class ModelSIM(torch.nn.Module):
    def __init__(self, encoder, prejector: prediction_MLP, predictor: projection_MLP):
        super(ModelSIM, self).__init__()
        self.encoder = encoder
        self.prejector = prejector
        self.predictor = predictor


    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor, batch) -> torch.Tensor:
        return self.encoder(x, edge_index, batch)

   
    def D(self, p, z): # negative cosine similarity
            z = z.detach() # stop gradient
            p = F.normalize(p, dim=1, p=2) # l2-normalize 
            z = F.normalize(z, dim=1, p=2) # l2-normalize 

            return -(p*z).sum(dim=1).mean()
        
    def loss(self, z1: torch.Tensor, z2: torch.Tensor):
        

        z1, z2 = self.prejector(z1), self.prejector(z1)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        L = self.D(p1, z2) / 2 + self.D(p2, z1) / 2
        return L
        
    def get_emb(self, x: torch.Tensor,
                edge_index: torch.Tensor, batch):
        orginal_emb = self.encoder.embedding(x, edge_index, batch)
        xmean = [x.sum(1) for x in orginal_emb] #this aims to sum all the node feature to only 1 dimesion, xmean.shape = [(node_number,),(node_number,)..]
        xs = xmean[0] # the following code aims to sum all layers' feature together
        for i in range(1,len(orginal_emb)):
            xs = xs + xmean[i]
            
        return xs
    
    def emb_avg(self, graph_emb, num_layer, batch, k):

        graph_emb = graph_emb.detach().cpu().numpy()
        index = faiss.IndexFlatL2(graph_emb.shape[1])
        index.add(graph_emb) 
        _, I = index.search(graph_emb, 5)
        graph_emb_new = 0
        for i in range(1, k):
            graph_emb_new = graph_emb_new + graph_emb[I[:,i],:] 
        graph_emb_new = torch.tensor(graph_emb_new).to(batch.device)
        node_emb  = graph_emb_new[batch] #expland the original graph number to node number
        node_emb  = F.normalize(node_emb,dim=1)

    
        return node_emb
    
    def get_emb_avg(self, x, edge_index, batch, k):
        
        orginal_emb = self.encoder.embedding(x, edge_index, batch)
        num_layer = len(orginal_emb)
        
        graph_emb = global_mean_pool(orginal_emb[0], batch)
        for i in range(1,len(orginal_emb)):
            graph_emb = graph_emb + global_mean_pool(orginal_emb[i], batch)

        xs = orginal_emb[0]
        for i in range(1,len(orginal_emb)):
            xs = xs + orginal_emb[i]
            
        xgraph = self.emb_avg(graph_emb, num_layer, batch, k)#this aims to sum only the last layers's node feature to only 1 dimesion,
        xs = xs * xgraph
        
        xs_mean = xs.sum(1)
        return xs_mean
    
    



