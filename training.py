import torch 
from amogel.model.GCN import GCN
from torch_geometric.data import DataLoader
from pytorch_lightning import Trainer

train_graph = torch.load("./artifacts/compare/traditional/train_graph.pt")
test_graph = torch.load("./artifacts/compare/traditional/test_graph.pt")

train_loader = DataLoader(train_graph , batch_size=32 , shuffle=True)
test_loader = DataLoader(test_graph , batch_size=32 , shuffle=False)

model = GCN(
    in_channels=1,
    hidden_channels=16,
    num_classes=5
)

trainer = Trainer(max_epochs=100)
trainer.fit(model , train_loader , test_loader)