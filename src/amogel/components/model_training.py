from amogel import logger 
from amogel.model.GCN import GCN
from amogel.model.graph_classification import GraphClassification , MultiGraphClassification , BinaryLearning , ContrastiveLearning , TripletLearning
from amogel.utils.pair_dataset import PairDataset
from torch_geometric.data import Batch
from torch_geometric.loader  import DataLoader
import torch 
import pytorch_lightning as pl
from amogel.entity.config_entity import ModelTrainingConfig
import mlflow

MODEL = {
    'MultiGraphClassification' : MultiGraphClassification, 
    'BinaryLearning' : BinaryLearning, 
    "ContrastiveLearning": ContrastiveLearning, 
    "TripletLearning": TripletLearning
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
        elif self.config.dataset == "unified_multigraph":
            train_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_unified_multigraphs_omic_1.pt")
            train_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_unified_multigraphs_omic_2.pt")
            train_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_unified_multigraphs_omic_3.pt")
            
            test_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_unified_multigraphs_omic_1.pt")
            test_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_unified_multigraphs_omic_2.pt")
            test_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_unified_multigraphs_omic_3.pt")
            
            self.in_channels = train_omic_1_graphs[0].x.size(1)
            
            self.train_loader = DataLoader(
                PairDataset(train_omic_1_graphs , train_omic_2_graphs , train_omic_3_graphs) ,
                batch_size=self.config.batch_size ,
                shuffle=True ,
                collate_fn=self.collate
            )
            
            self.test_loader = DataLoader(
                PairDataset(test_omic_1_graphs , test_omic_2_graphs , test_omic_3_graphs) ,
                batch_size=1 ,
                shuffle=False ,
                collate_fn=self.collate_multigraph
            )
        elif self.config.dataset == "unified_multigraph_test":
            train_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_unified_test_multigraphs_omic_1.pt")
            train_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_unified_test_multigraphs_omic_2.pt")
            train_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_unified_test_multigraphs_omic_3.pt")
            
            test_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_unified_test_multigraphs_omic_1.pt")
            test_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_unified_test_multigraphs_omic_2.pt")
            test_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_unified_test_multigraphs_omic_3.pt")
            
            self.in_channels = train_omic_1_graphs[0].x.size(1)
            
            self.train_loader = DataLoader(
                PairDataset(train_omic_1_graphs , train_omic_2_graphs , train_omic_3_graphs) ,
                batch_size=self.config.batch_size ,
                shuffle=True ,
                collate_fn=self.collate
            )
            
            self.test_loader = DataLoader(
                PairDataset(test_omic_1_graphs , test_omic_2_graphs , test_omic_3_graphs) ,
                batch_size=1 ,
                shuffle=False ,
                collate_fn=self.collate_multigraph
            )
        elif self.config.dataset == "multiedges_multigraph":
            train_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_multiedges_multigraphs_omic_1.pt")
            train_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_multiedges_multigraphs_omic_2.pt")
            train_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_multiedges_multigraphs_omic_3.pt")
            
            test_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_multiedges_multigraphs_omic_1.pt")
            test_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_multiedges_multigraphs_omic_2.pt")
            test_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_multiedges_multigraphs_omic_3.pt")
            
            self.in_channels = train_omic_1_graphs[0].x.size(1)
            
            self.train_loader = DataLoader(
                PairDataset(train_omic_1_graphs , train_omic_2_graphs , train_omic_3_graphs) ,
                batch_size=self.config.batch_size ,
                shuffle=True ,
                collate_fn=self.collate
            )
            
            self.test_loader = DataLoader(
                PairDataset(test_omic_1_graphs , test_omic_2_graphs , test_omic_3_graphs) ,
                batch_size=1 ,
                shuffle=False ,
                collate_fn=self.collate_multigraph
            )
        elif self.config.dataset == "discretized_multiedges_multigraph":
            train_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_discretized_multiedges_multigraphs_omic_1.pt")
            train_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_discretized_multiedges_multigraphs_omic_2.pt")
            train_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_discretized_multiedges_multigraphs_omic_3.pt")
            
            test_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_discretized_multiedges_multigraphs_omic_1.pt")
            test_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_discretized_multiedges_multigraphs_omic_2.pt")
            test_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_discretized_multiedges_multigraphs_omic_3.pt")
            
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
                collate_fn=self.collate_multigraph
            )
        elif self.config.dataset == "common_multigraph":
            train_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_common_graphs_omic_1.pt")
            train_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_common_graphs_omic_2.pt")
            train_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_common_graphs_omic_3.pt")
            
            test_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_common_graphs_omic_1.pt")
            test_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_common_graphs_omic_2.pt")
            test_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_common_graphs_omic_3.pt")
            
            self.in_channels = train_omic_1_graphs[0].x.size(1)
            
            self.train_loader = DataLoader(
                PairDataset(train_omic_1_graphs , train_omic_2_graphs , train_omic_3_graphs) ,
                batch_size=self.config.batch_size ,
                shuffle=True ,
                collate_fn=self.collate
            )
            
            self.test_loader = DataLoader(
                PairDataset(test_omic_1_graphs , test_omic_2_graphs , test_omic_3_graphs) ,
                batch_size=1 ,
                shuffle=False ,
                collate_fn=self.collate_multigraph
            )
        elif self.config.dataset == "discretized_common_multigraph":
            train_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_discretized_common_graphs_omic_1.pt")
            train_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_discretized_common_graphs_omic_2.pt")
            train_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_discretized_common_graphs_omic_3.pt")
            
            test_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_discretized_common_graphs_omic_1.pt")
            test_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_discretized_common_graphs_omic_2.pt")
            test_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_discretized_common_graphs_omic_3.pt")
            
            self.in_channels = train_omic_1_graphs[0].x.size(1)
            
            self.train_loader = DataLoader(
                PairDataset(train_omic_1_graphs , train_omic_2_graphs , train_omic_3_graphs) ,
                batch_size=self.config.batch_size ,
                shuffle=True ,
                collate_fn=self.collate
            )
            
            self.test_loader = DataLoader(
                PairDataset(test_omic_1_graphs , test_omic_2_graphs , test_omic_3_graphs) ,
                batch_size=1 ,
                shuffle=False ,
                collate_fn=self.collate_multigraph
            )
         
        elif self.config.dataset == "binarylearning_multigraph":
            train_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_binaryclassifier_multigraphs_omic_1.pt")
            train_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_binaryclassifier_multigraphs_omic_2.pt")
            train_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_binaryclassifier_multigraphs_omic_3.pt")
            
            test_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_binaryclassifier_multigraphs_omic_1.pt")
            test_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_binaryclassifier_multigraphs_omic_2.pt")
            test_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_binaryclassifier_multigraphs_omic_3.pt")
            
            self.in_channels = train_omic_1_graphs[0].x.size(1)
            
            self.train_loader = DataLoader(
                PairDataset(train_omic_1_graphs , train_omic_2_graphs , train_omic_3_graphs) ,
                batch_size=self.config.batch_size ,
                shuffle=True ,
                collate_fn=self.collate
            )
            
            self.test_loader = DataLoader(
                PairDataset(test_omic_1_graphs , test_omic_2_graphs , test_omic_3_graphs) ,
                batch_size=1 ,
                shuffle=False ,
                collate_fn=self.collate_multigraph
            )
        elif self.config.dataset == "contrastive_multigraph":
            train_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_contrastive_multigraphs_omic_1.pt")
            train_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_contrastive_multigraphs_omic_2.pt")
            train_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_contrastive_multigraphs_omic_3.pt")
            
            test_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_constrastive_multigraphs_omic_1.pt")
            test_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_constrastive_multigraphs_omic_2.pt")
            test_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_constrastive_multigraphs_omic_3.pt")
            
            self.in_channels = train_omic_1_graphs[0][0].x.size(1)
            
            self.train_loader = DataLoader(
                PairDataset(train_omic_1_graphs , train_omic_2_graphs , train_omic_3_graphs) ,
                batch_size=self.config.batch_size ,
                shuffle=True ,
                collate_fn=self.collate_contrastive_multigraph
            )
            
            self.test_loader = DataLoader(
                PairDataset(test_omic_1_graphs , test_omic_2_graphs , test_omic_3_graphs) ,
                batch_size=1 ,
                shuffle=False ,
                collate_fn=self.collate_multigraph
            )
        elif self.config.dataset == "triplet_multigraph":
            train_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_triplet_multigraphs_omic_1.pt")
            train_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_triplet_multigraphs_omic_2.pt")
            train_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/training_triplet_multigraphs_omic_3.pt")
            
            test_omic_1_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_triplet_multigraphs_omic_1.pt")
            test_omic_2_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_triplet_multigraphs_omic_2.pt")
            test_omic_3_graphs = torch.load(r"artifacts/knowledge_graph/BRCA/testing_triplet_multigraphs_omic_3.pt")
            
            self.in_channels = train_omic_1_graphs[0][0].x.size(1)
            
            self.train_loader = DataLoader(
                PairDataset(train_omic_1_graphs , train_omic_2_graphs , train_omic_3_graphs) ,
                batch_size=self.config.batch_size ,
                shuffle=True ,
                collate_fn=self.collate_contrastive_multigraph
            )
            
            self.test_loader = DataLoader(
                PairDataset(test_omic_1_graphs , test_omic_2_graphs , test_omic_3_graphs) ,
                batch_size=1 ,
                shuffle=False ,
                collate_fn=self.collate_triplet_multigraph
            )
        else: 
            raise ValueError("Invalid parameters for dataset")
        
        self.model = MODEL[self.config.model](
            in_channels=self.in_channels,
            hidden_channels=self.config.hidden_units,
            num_classes=5,
            lr=self.config.learning_rate,
            drop_out=self.config.drop_out, 
            mlflow=mlflow, 
            multi_graph_testing = (self.config.dataset == "unified_multigraph" or self.config.dataset == "contrastive_multigraph" or self.config.dataset == "unified_multigraph_test"), 
            weight=self.config.weight , 
            alpha=self.config.alpha , 
            binary=self.config.binary , 
            multihead=self.config.multihead, 
            multihead_concat=self.config.multihead_concat
        )
        
        # clean multigraph_testing_logs.txt
        with open("multigraph_testing_logs.txt", "w") as f:
            f.write("")
        
    def training(self) -> None:
        """Train model using Pytorch Lightning Trainer.
        """
        
        logger.info("Model training started")
        mlflow.pytorch.autolog()
        
        with mlflow.start_run():
            mlflow.log_params(self.config.__dict__)
            mlflow.pytorch.log_model(self.model , "model")
            
            self.trainer = pl.Trainer(max_epochs=self.config.learning_epoch)
            if self.config.enable_validation:
                self.trainer.fit(self.model , self.train_loader , self.test_loader)
            else: 
                self.trainer.fit(self.model , self.train_loader)
            
        logger.info("Model training completed")
        
        logger.info(f"Save model checkpoing : artifacts/model/{self.config.model}.ckpt")
        self.trainer.save_checkpoint(f"artifacts/model/{self.config.model}.ckpt")
    
    def testing(self) -> None: 
        """Test model using Pytorch Lightning Trainer.
        """
        
        logger.info("Model testing started")
        
        self.trainer.test(self.model , self.test_loader)
            
        logger.info("Model testing completed")
    @staticmethod
    def collate(data_list):
        """Collate multiple data objects into a single data object."""
        batch = Batch.from_data_list(data_list)
        batchA = Batch.from_data_list([data[0] for data in data_list])
        batchB = Batch.from_data_list([data[1] for data in data_list])
        batchC = Batch.from_data_list([data[2] for data in data_list])
        return batchA, batchB , batchC
    
    @staticmethod 
    def collate_contrastive_multigraph(data_list):
        batchA = [
            Batch.from_data_list([data[0][0] for data in data_list]), # posiive
            Batch.from_data_list([data[0][1] for data in data_list]) # negative
        ]
        
        batchB = [
            Batch.from_data_list([data[1][0] for data in data_list]), # posiive
            Batch.from_data_list([data[1][1] for data in data_list]) # negative
        ]
        
        batchC = [
            Batch.from_data_list([data[2][0] for data in data_list]), # posiive
            Batch.from_data_list([data[2][1] for data in data_list]) # negative
        ]
    
        return batchA, batchB , batchC
    
    @staticmethod 
    def collate_triplet_multigraph(data_list):
        batchA = [
            Batch.from_data_list([data[0][0] for data in data_list]), # anchor
            Batch.from_data_list([data[0][1] for data in data_list]), # posiive
            Batch.from_data_list([data[0][2] for data in data_list]) # negative
        ]
        
        batchB = [
            Batch.from_data_list([data[1][0] for data in data_list]), # anchor
            Batch.from_data_list([data[1][1] for data in data_list]), # posiive
            Batch.from_data_list([data[1][2] for data in data_list]) # negative
        ]
        
        batchC = [
            Batch.from_data_list([data[2][0] for data in data_list]), # anchor
            Batch.from_data_list([data[2][1] for data in data_list]), # posiive
            Batch.from_data_list([data[2][2] for data in data_list]) # negative
        ]
    
        return batchA, batchB , batchC
    
    @staticmethod
    def collate_multigraph(data_list , class_num:int):
        batchAs = []
        batchBs = []
        batchCs = []
        for i in range(class_num):
            batchA = Batch.from_data_list([data[0][i] for data in data_list])
            batchB = Batch.from_data_list([data[1][i] for data in data_list])
            batchC = Batch.from_data_list([data[2][i] for data in data_list])
            
            batchAs.append(batchA)
            batchBs.append(batchB)
            batchCs.append(batchC)
        
        return batchAs, batchBs , batchCs