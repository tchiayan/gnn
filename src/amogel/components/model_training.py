from amogel import logger 
from amogel.model.GCN import GCN
from amogel.model.graph_classification import GraphClassification
from torch_geometric.data import DataLoader 
import torch 
import pytorch_lightning as pl
from amogel.entity.config_entity import ModelTrainingConfig
import mlflow

MODEL = {
    'GCN' : GCN , 
    'GraphClassification' : GraphClassification
}

class ModelTraining():
    
    def __init__(self  , config: ModelTrainingConfig): 
        
        
        self.config = config
        
        if self.config.dataset == "unified":
            self.traing_graph = torch.load(r"artifacts/knowledge_graph/BRCA/training_unified_graphs_omic_1.pt")
            self.testing_graph = torch.load(r"artifacts/knowledge_graph/BRCA/testing_unified_graphs_omic_1.pt")
        else: 
            self.traing_graph = torch.load(r"artifacts/knowledge_graph/BRCA/training_embedding_graphs_omic_1.pt")
            self.testing_graph = torch.load(r"artifacts/knowledge_graph/BRCA/testing_embedding_graphs_omic_1.pt")
        
        self.model = MODEL[self.config.model](
            in_channels=self.traing_graph[0].x.size(1),
            hidden_channels=self.config.hidden_units,
            num_classes=5,
            lr=self.config.learning_rate,
            drop_out=self.config.drop_out
        )
        
    def training(self):
        logger.info("Model training started")
        mlflow.pytorch.autolog()
        
        with mlflow.start_run():
            mlflow.log_params(self.config.__dict__)
            mlflow.pytorch.log_model(self.model , "model")
            
            train_loader = DataLoader(self.traing_graph , batch_size=self.config.batch_size , shuffle=True)
            test_loader = DataLoader(self.testing_graph , batch_size=self.config.batch_size , shuffle=False)
            
            trainer = pl.Trainer(max_epochs=self.config.learning_epoch)
            trainer.fit(self.model , train_loader , test_loader)
            
        logger.info("Model training completed")
            