import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, coder_hid_num, out_channels, activation,
                 base_model=GCNConv, k = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, coder_hid_num * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(coder_hid_num * out_channels, coder_hid_num * out_channels))
        self.conv.append(base_model(coder_hid_num * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x

class EncoderGAT(torch.nn.Module):
    def __init__(self, in_channels, head, out_channels, activation,
                 base_model=GATConv, k: int = 1):
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
    def __init__(self, encoder: Encoder, prejector: prediction_MLP, predictor: projection_MLP):
        super(ModelSIM, self).__init__()
        self.encoder: Encoder = encoder
        self.prejector = prejector
        self.predictor = predictor


    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

   
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
        



