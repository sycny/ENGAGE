import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv,global_add_pool, global_mean_pool
from torch.nn import Sequential, Linear, ReLU
import faiss

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

class Model(torch.nn.Module):
    def __init__(self, encoder: EncoderGIN, num_hidden, num_proj_hidden,num_layer,
                 tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden*num_layer, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x, edge_index, batch) -> torch.Tensor:
        return self.encoder(x, edge_index, batch)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() 

        return ret
    
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