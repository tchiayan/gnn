from amogel import logger 
from amogel.model.GCN import GCN
from torch_geometric.data import DataLoader 
import torch 
import pytorch_lightning as pl
from amogel.entity.config_entity import ModelTrainingConfig

class ModelTraining():
    
    def __init__(self  , config: ModelTrainingConfig): 
        self.traing_graph = torch.load(r"artifacts/knowledge_graph/BRCA/training_graphs_omic_1.pt")
        self.testing_graph = torch.load(r"artifacts/knowledge_graph/BRCA/testing_graphs_omic_1.pt")
        self.config = config
        
        self.model = GCN(
            num_features=self.traing_graph[0].x.size(1),
            hidden_channels=self.config.hidden_units,
            output_class=5, 
            lr=self.config.learning_rate, 
            drop_out=self.config.drop_out
        )
        
    def training(self):
        logger.info("Model training started")
        train_loader = DataLoader(self.traing_graph , batch_size=32 , shuffle=True)
        test_loader = DataLoader(self.testing_graph , batch_size=32 , shuffle=False)
        
        trainer = pl.Trainer(max_epochs=10)
        trainer.fit(self.model , train_loader , test_loader)
        
        logger.info("Model training completed")
        