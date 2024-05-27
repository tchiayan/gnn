import torch 
from torch_geometric.nn import GCNConv , global_mean_pool

class GCNEncoder(torch.nn.Module):
    
    def __init__(self , in_channels , out_channels):
        super(GCNEncoder , self).__init__()
        self.conv1 = GCNConv(in_channels , 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels , out_channels)
        
    def forward(self , x , edge_index):
        x = self.conv1(x , edge_index).relu()
        x = self.conv2(x , edge_index)
        return x
    
class VariationalGCNEncoder(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels, cached=True) # cached only for transductive learning
        self.conv_mu = GCNConv(out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(out_channels, out_channels, cached=True)
        

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
    
class Pooling(torch.nn.Module):
    
    def __init__(self, hidden_channels ,  number_of_classes , *args, **kwargs) -> None:
        super(Pooling , self).__init__(*args, **kwargs)
        
        self.classification = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels , number_of_classes)
        )
        
    def forward(self , x):
        x = x.mean(dim=0) # shape [ 502 , 64]
        return self.classification(x)