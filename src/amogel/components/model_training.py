from amogel import logger 
from amogel.model.GCN import GCN
from amogel.model.graph_classification import GraphClassification , MultiGraphClassification
from amogel.utils.pair_dataset import PairDataset
from torch_geometric.data import DataLoader , Batch
import torch 
import pytorch_lightning as pl
from amogel.entity.config_entity import ModelTrainingConfig
import mlflow

MODEL = {
    'MultiGraphClassification' : MultiGraphClassification
}

class ModelTraining():
    
    def __init__(self  , config: ModelTrainingConfig): 
        
        
        self.config = config
        
        logger.info(f"Loading dataset [{self.config.dataset}] for model [{self.config.model}] ")
        if self.config.dataset == "unified":
            train_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_unified_graphs_omic_1.pt")
            train_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_unified_graphs_omic_2.pt")
            train_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_unified_graphs_omic_3.pt")
            
            test_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_unified_graphs_omic_1.pt")
            test_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_unified_graphs_omic_2.pt")
            test_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_unified_graphs_omic_3.pt")
            
            self.in_channels = train_omic_1_graphs[0].x.size(1)
            
            self.train_loader = DataLoader(
                PairDataset(train_omic_1_graphs , train_omic_2_graphs , train_omic_3_graphs) , 
                batch_size=self.config.batch_size , 
                shuffle=True , 
                collate_fn=self.collate
            )
            
            self.test_loader = DataLoader(
                PairDataset(test_omic_1_graphs , test_omic_2_graphs , test_omic_3_graphs) , 
                batch_size=self.config.batch_size , 
                shuffle=False , 
                collate_fn=self.collate
            )
        elif self.config.dataset == "embedding": 
            train_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_embedding_graphs_omic_1.pt")
            train_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_embedding_graphs_omic_2.pt")
            train_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_embedding_graphs_omic_3.pt")
            
            test_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_embedding_graphs_omic_1.pt")
            test_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_embedding_graphs_omic_2.pt")
            test_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_embedding_graphs_omic_3.pt")
            
            self.in_channels = train_omic_1_graphs[0].x.size(1)
            
            self.train_loader = DataLoader(
                PairDataset(train_omic_1_graphs , train_omic_2_graphs , train_omic_3_graphs) , 
                batch_size=self.config.batch_size , 
                shuffle=True , 
                collate_fn=self.collate
            )
            
            self.test_loader = DataLoader(
                PairDataset(test_omic_1_graphs , test_omic_2_graphs , test_omic_3_graphs) , 
                batch_size=self.config.batch_size , 
                shuffle=False , 
                collate_fn=self.collate
            )
        elif self.config.dataset == "correlation":
            train_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_corr_graphs_omic_1.pt")
            train_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_corr_graphs_omic_2.pt")
            train_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_corr_graphs_omic_3.pt")
            
            test_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_corr_graphs_omic_1.pt")
            test_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_corr_graphs_omic_2.pt")
            test_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_corr_graphs_omic_3.pt")
            
            self.in_channels = train_omic_1_graphs[0].x.size(1)
            
            self.train_loader = DataLoader(
                PairDataset(train_omic_1_graphs , train_omic_2_graphs , train_omic_3_graphs) ,
                batch_size=self.config.batch_size ,
                shuffle=True ,
                collate_fn=self.collate
            )
            
            self.test_loader = DataLoader(
                PairDataset(test_omic_1_graphs , test_omic_2_graphs , test_omic_3_graphs) ,
                batch_size=self.config.batch_size ,
                shuffle=False ,
                collate_fn=self.collate
            )
        else: 
            raise ValueError("Invalid parameters for dataset")
        
        self.model = MODEL[self.config.model](
            in_channels=self.in_channels,
            hidden_channels=self.config.hidden_units,
            num_classes=5,
            lr=self.config.learning_rate,
            drop_out=self.config.drop_out, 
            mlflow=mlflow
        )
        
    def training(self) -> None:
        """Train model using Pytorch Lightning Trainer.
        """
        
        logger.info("Model training started")
        mlflow.pytorch.autolog()
        
        with mlflow.start_run():
            mlflow.log_params(self.config.__dict__)
            mlflow.pytorch.log_model(self.model , "model")
            
            trainer = pl.Trainer(max_epochs=self.config.learning_epoch)
            trainer.fit(self.model , self.train_loader , self.test_loader)
            
        logger.info("Model training completed")
        
    @staticmethod
    def collate(data_list):
        """Collate multiple data objects into a single data object."""
        batch = Batch.from_data_list(data_list)
        batchA = Batch.from_data_list([data[0] for data in data_list])
        batchB = Batch.from_data_list([data[1] for data in data_list])
        batchC = Batch.from_data_list([data[2] for data in data_list])
        return batchA, batchB , batchC
            