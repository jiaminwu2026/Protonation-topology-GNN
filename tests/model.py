import torch
from torch_geometric.nn import GraphConv, global_mean_pool, MLP
import torch.nn.functional as F
import numpy as np

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_conv_layers=3, activation=F.relu):
        super(GCN, self).__init__()
        torch.manual_seed(25)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GraphConv(4, hidden_channels))
        for _ in range(num_conv_layers - 1):
            self.convs.append(GraphConv(hidden_channels, hidden_channels))
        self.mlp = MLP(in_channels=hidden_channels+2, hidden_channels=hidden_channels*2, out_channels=1, num_layers=2)
        self.activation = activation
        self.double()

    def forward(self, x, edge_index, batch, features):
        # 1. Obtain node embeddings 
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.activation(x)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  

        # 3. Concatenate with additional features

        x = torch.cat([x,features],dim=1)
        x = F.dropout(x, p=0, training=self.training)

        # 4. MLP for final prediction
        x = self.mlp(x)
        
        return x
    
def train(model, loader, optimizer, criterion=torch.nn.MSELoss()):
    model.train()
    for data in loader:  
        out = model(data.x, data.edge_index, data.batch, data.features)  
        loss = criterion(out, data.y.unsqueeze(1)) 
        loss.backward() 
        optimizer.step() 
        optimizer.zero_grad()  

def test(model, loader, criterion=torch.nn.MSELoss()):
    model.eval()
    mse = []
    for data in loader:
        out = model(data.x, data.edge_index, data.batch, data.features)
        target = (data.y).unsqueeze(1)
        mse.append(criterion(out,target).item())
    average_mse = np.average(mse)
    return(average_mse)