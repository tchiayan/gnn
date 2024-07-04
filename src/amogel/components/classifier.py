# Implement KNN Classification on dataset
import pandas as pd
import os 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report , roc_auc_score
from amogel import logger
from amogel.model.DNN import DNN , OmicDataset
from amogel.utils.ac import generate_ac_to_file , generate_ac_feature_selection
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from amogel.utils.common import symmetric_matrix_to_pyg , load_ppi , load_omic_features_name , load_feature_conversion 
from amogel.utils.gene import generate_edges_from_annotation
from amogel.model.GCN import GCN
import warnings
import torch
from torch.nn.functional import one_hot
from tqdm import tqdm
import mlflow
from amogel.entity.config_entity import CompareOtherConfig
warnings.filterwarnings("ignore")

class OtherClassifier:
    
    def __init__(self, config:CompareOtherConfig ,  dataset="BRCA"):
        
        self.config = config
        self.dataset = dataset 
        
    def load_data_without_ac(self):
        
        os.makedirs("./artifacts/compare/traditional" , exist_ok=True)
        
        # load train data 
        train_data_omic_1 = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , f"1_tr.csv"), header=None)
        train_data_omic_2 = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , f"2_tr.csv"), header=None)
        train_data_omic_3 = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , f"3_tr.csv"), header=None)
        
        self.train_data = pd.concat([train_data_omic_1 , train_data_omic_2 , train_data_omic_3 ] , axis=1)
        self.train_data.columns = list(range(self.train_data.shape[1]))
        
        # load train label 
        self.train_label = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , "labels_tr.csv") , header=None , names=["label"])
        self.num_classes = len(self.train_label['label'].unique())
        
        # load test data 
        test_data_omic_1 = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , f"1_te.csv"), header=None)
        test_data_omic_2 = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , f"2_te.csv"), header=None)
        test_data_omic_3 = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , f"3_te.csv"), header=None)
        
        self.test_data = pd.concat([test_data_omic_1  , test_data_omic_2 , test_data_omic_3 ] , axis=1)
        self.test_data.columns = list(range(self.test_data.shape[1]))
        
        
        # load test label
        self.test_label = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , "labels_te.csv") , header=None , names=["label"])
        
        logger.info("Data dimension for training and testing")
        logger.info(f"Train data: {self.train_data.shape}")
        logger.info(f"Train label: {self.train_label.shape}")
        logger.info(f"Test data: {self.test_data.shape}")
        logger.info(f"Test label: {self.test_label.shape}")

    def load_data(self , select_k="auto"):
        
        # load train data 
        train_data_omic_1 = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , f"1_tr.csv"), header=None)
        train_data_omic_2 = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , f"2_tr.csv"), header=None)
        train_data_omic_3 = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , f"3_tr.csv"), header=None)
        
        self.train_data = pd.concat([train_data_omic_1 , train_data_omic_2 , train_data_omic_3 ] , axis=1)
        self.train_data.columns = list(range(self.train_data.shape[1]))
        # selected_gene_omic_1 = self.get_arm_feature_selection(1)
        # selected_gene_omic_2 = self.get_arm_feature_selection(2)
        # selected_gene_omic_3 = self.get_arm_feature_selection(3)
        # self.train_data_filter = pd.concat([train_data_omic_1[selected_gene_omic_1] , train_data_omic_2[selected_gene_omic_2] , train_data_omic_3[selected_gene_omic_3] ] , axis=1)
        
        
        # load train label 
        self.train_label = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , "labels_tr.csv") , header=None , names=["label"])
        self.num_classes = len(self.train_label['label'].unique())
        
        
        # load test data 
        test_data_omic_1 = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , f"1_te.csv"), header=None)
        test_data_omic_2 = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , f"2_te.csv"), header=None)
        test_data_omic_3 = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , f"3_te.csv"), header=None)
        
        self.test_data = pd.concat([test_data_omic_1  , test_data_omic_2 , test_data_omic_3 ] , axis=1)
        self.test_data.columns = list(range(self.test_data.shape[1]))
        # self.test_data_filter = pd.concat([test_data_omic_1[selected_gene_omic_1] , test_data_omic_2[selected_gene_omic_2] , test_data_omic_3[selected_gene_omic_3] ] , axis=1)
        
        # load test label
        self.test_label = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , "labels_te.csv") , header=None , names=["label"])
        
        
        if select_k == "auto":
            est , selected_gene , information_edge_tensor  = generate_ac_feature_selection(self.train_data , self.train_label.copy(deep=True) , "" , n_bins=self.config.n_bins)
        else:
            est , selected_gene , information_edge_tensor  = generate_ac_feature_selection(self.train_data , self.train_label.copy(deep=True) , "" , fixed_k=select_k , n_bins=self.config.n_bins)
        logger.info(f"Selected gene: {len(selected_gene)}")
        selected_gene = sorted(selected_gene)
        selection = {0:0,1:0,2:0}
        for gene in selected_gene: 
            if gene in range(0 , train_data_omic_1.shape[1]):
                selection[0] += 1
            elif gene in range(train_data_omic_1.shape[1] , train_data_omic_1.shape[1] + train_data_omic_2.shape[1]):
                selection[1] += 1
            else:
                selection[2] += 1
        logger.info(f"Selected gene distribution: {selection}")
        self.selected_gene = selected_gene
        os.makedirs("./artifacts/ac_genes" , exist_ok=True)
        torch.save(selected_gene , f"./artifacts/ac_genes/gene.pt")
        self.information_edge_tensor = information_edge_tensor
        
        if self.config.discretized:
            self.train_data_ac =  pd.DataFrame(est.transform(self.train_data))[selected_gene]
            self.test_data_ac = pd.DataFrame(est.transform(self.test_data))[selected_gene]
        else:
            self.train_data_ac = self.train_data[selected_gene]
            self.test_data_ac = self.test_data[selected_gene]
        
        logger.info("Data dimension for training and testing")
        logger.info(f"Train data: {self.train_data.shape}")
        logger.info(f"Train label: {self.train_label.shape}")
        logger.info(f"Test data: {self.test_data.shape}")
        logger.info(f"Test label: {self.test_label.shape}")
        # logger.info(f"Filtered Train data: {self.train_data_filter.shape}")
        # logger.info(f"Filtered Test data: {self.test_data_filter.shape}")
        logger.info(f"Train data with AC: {self.train_data_ac.shape}")
        logger.info(f"Test data with AC: {self.test_data_ac.shape}")
        
    def get_arm_feature_selection(self , omic_type):
        
        omic1 = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , f"ac_rule_{omic_type}.tsv") , header=None , sep="\t" , names=["label" , "confidence" , "support" , "rules" , "interestingness_1" , "interestingness_2" , "interestingness_3"])
        
        # grouped_top_tr = omic1.groupby("label").apply(lambda x: x.nlargest(500 , "interestingness_1")).reset_index(drop=True)
        
        # Build unique genes list 
        genes = set()
        for idx , row in omic1.iterrows():
            genes.update([int(x.split(":")[0]) for x in  row['rules'].split(",")])
            
        return list(genes)
        
    def train_and_evaluate_knn(self):
        # fit the model and evaluate using KNN from sklearn
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(self.train_data , self.train_label)
        pred = knn.predict(self.test_data)
        pred_prob = knn.predict_proba(self.test_data)
        output  = classification_report(self.test_label , pred , digits=4)
        if pred_prob.shape[1] == 2:
            auc = roc_auc_score(self.test_label , pred_prob[:,1] , multi_class="ovr")
        else:
            auc = roc_auc_score(self.test_label , pred_prob , multi_class="ovr")
        output += f"au_graph\t{auc:.4f}\n"
        print(output)
        
        with open(os.path.join("./artifacts/compare/traditional"  , "knn_report.txt") , "w") as f:
            f.write(output)
            
    def train_and_evaluate_svm(self):
        
        # fit the model and evaluate using SVM from sklearn
        from sklearn.svm import SVC
        svm = SVC(probability=True)
        svm.fit(self.train_data , self.train_label)
        pred = svm.predict(self.test_data)
        pred_prob = svm.predict_proba(self.test_data)
        output  = classification_report(self.test_label , pred , digits=4)
        if pred_prob.shape[1] == 2:
            auc = roc_auc_score(self.test_label , pred_prob[:,1] , multi_class="ovr")
        else:
            auc = roc_auc_score(self.test_label , pred_prob , multi_class="ovr")
        output += f"au_graph: {auc:.4f}"
        
        print(output)
        
        with open(os.path.join("./artifacts/compare/traditional" , "svm_report.txt") , "w") as f:
            f.write(output)
    
    def train_and_evaluate_nb(self):
        
        # fit the model and evaluate using Naive Bayes from sklearn
        
        from sklearn.naive_bayes import GaussianNB
        nb = GaussianNB()
        nb.fit(self.train_data , self.train_label)
        
        pred = nb.predict(self.test_data)
        pred_prob = nb.predict_proba(self.test_data)
        output  = classification_report(self.test_label , pred , digits=4)
        if pred_prob.shape[1] == 2:
            auc = roc_auc_score(self.test_label , pred_prob[:,1] , multi_class="ovr")
        else:
            auc = roc_auc_score(self.test_label , pred_prob , multi_class="ovr")
        output += f"roc_auc: {auc:.4f} \n"
        
        print(output)
        
        with open(os.path.join("./artifacts/compare/traditional" , "nb_report.txt") , "w") as f:
            f.write(output)
            
    def train_and_evaluate_dnn(self):
        
        # load data and train the model using DNN
        train_data = OmicDataset(self.train_data , self.train_label)
        test_data = OmicDataset(self.test_data , self.test_label)
        
        # dataloader 
        train_loader = DataLoader(train_data , batch_size=32 , shuffle=True)
        test_loader = DataLoader(test_data , batch_size=32 , shuffle=False)
        
        model = DNN(input_dimension=self.train_data.shape[1] , num_classes=self.num_classes)
        
        trainer = Trainer(max_epochs=100)
        trainer.fit(model , train_loader , test_loader)
        
    def train_and_evaluate_dnn_feature_selection(self):
        # load data and train the model using DNN
        train_data = OmicDataset(self.train_data_filter , self.train_label)
        test_data = OmicDataset(self.test_data_filter , self.test_label)
        
        # dataloader 
        train_loader = DataLoader(train_data , batch_size=32 , shuffle=True)
        test_loader = DataLoader(test_data , batch_size=32 , shuffle=False)
        
        model = DNN(input_dimension=self.train_data_filter.shape[1] , num_classes=self.num_classes)
        
        trainer = Trainer(max_epochs=100)
        trainer.fit(model , train_loader , test_loader)
        
    def train_and_evaluate_dnn_feature_selection_ac(self):
        # load data and train the model using DNN
        train_data = OmicDataset(self.train_data_ac , self.train_label)
        test_data = OmicDataset(self.test_data_ac , self.test_label)
        
        # dataloader 
        train_loader = DataLoader(train_data , batch_size=32 , shuffle=True)
        test_loader = DataLoader(test_data , batch_size=32 , shuffle=False)
        
        model = DNN(input_dimension=self.train_data_ac.shape[1] , num_classes=self.num_classes)
        
        trainer = Trainer(max_epochs=100)
        trainer.fit(model , train_loader , test_loader)
        
    def train_and_evaluate_graph_feature_selection_ac(self):
        
        edge_matrix = []
        threshold = self.config.corr_threshold
        info_mean = 0
        
        if self.config.information: 
            logger.info(f"Generate information edges...")
            information_tensor = self.information_edge_tensor 
            information_tensor = information_tensor[self.selected_gene][:, self.selected_gene]
            
            # fill nan with 0
            information_tensor[torch.isnan(information_tensor)] = 0
            
            # calculate mean of information tensor given shape is 2D tensor
            info_mean = information_tensor.mean()
            
            edge_matrix.append(information_tensor)
            
        if self.config.corr:
            logger.info(f"Generating correlation edges...")
            # generate graph data
            corr = self.train_data_ac.corr()
            
            if self.config.negative_corr:
                corr = corr[corr.abs() >= threshold]
            else:
                corr = corr.abs()
                corr = corr[corr > threshold]
            
            # convert to tensor 
            corr_tensor = torch.tensor(corr.values , dtype=torch.float32)
            
            # fill nan with 0 
            corr_tensor[torch.isnan(corr_tensor)] = 0
            logger.info(f"Correlation matrix shape: {corr_tensor.shape}")
            edge_matrix.append(corr_tensor)
        
        # load ppi 
        if self.config.ppi:
            logger.info(f"Generating PPI edges...")
            feature_names = load_omic_features_name("./artifacts/data_preprocessing" , self.dataset , [1,2,3])
            ppi = load_ppi("./artifacts/ppi_data/unzip"  , feature_names , self.config.ppi_score)
            ppi_tensor = torch.zeros(feature_names.shape[0] , feature_names.shape[0])
            if self.config.ppi_edge == 'binary':
                ppi_tensor[ppi["gene1_idx"].values , ppi['gene2_idx'].values] = 1 
                ppi_tensor[ppi["gene2_idx"].values , ppi['gene1_idx'].values] = 1
            else:
                max_ppi_score = ppi['combined_score'].max()
                ppi_tensor[ppi["gene1_idx"].values , ppi['gene2_idx'].values] = torch.tensor(ppi['combined_score'].values / max_ppi_score , dtype=torch.float)
                ppi_tensor[ppi["gene2_idx"].values , ppi['gene1_idx'].values] = torch.tensor(ppi['combined_score'].values / max_ppi_score , dtype=torch.float)
            ppi_tensor = ppi_tensor[self.selected_gene][:, self.selected_gene]
            
            if self.config.information and self.config.info_mean:
                ppi_tensor = ppi_tensor * info_mean
                
            edge_matrix.append(ppi_tensor)
            logger.info(f"PPI matrix shape: {ppi_tensor.shape}")
            assert (ppi_tensor != ppi_tensor.T).int().sum() == 0 , "PPI should be symmetric"
            #assert ppi_tensor.shape[0] == corr_tensor.shape[0] , "PPI and AC should have the same dimension"
            #assert ppi_tensor.shape[1] == corr_tensor.shape[1] , "PPI and AC should have the same dimension"
        
            
            # assert (information_tensor != information_tensor.T).int().sum() == 0 , f"Information tensor should be symmetric: {(information_tensor != information_tensor.T).int().sum()} | {(information_tensor == information_tensor.T).int().sum()}" 
        if self.config.kegg: 
            logger.info(f"Loading KEGG edges...")
            feature_omic = load_omic_features_name(
                "./artifacts/data_preprocessing/" , dataset=self.dataset, type=[1,2,3]
            )
            feature_conversion = load_feature_conversion(
                "./artifacts/data_ingestion/unzip/" , dataset=self.dataset
            )
            features = feature_omic.merge(feature_conversion , left_on="gene_name" , right_on="gene" , how="left")
            kegg_edges = generate_edges_from_annotation(
                f"./artifacts/data_ingestion/unzip/{self.dataset}_kegg_go/KEGG_PATHWAY.txt", 
                features=features, 
                filter_p_value=self.config.kegg_filter
            )
            kegg_edges = kegg_edges[self.selected_gene][:, self.selected_gene]
            if self.config.information and self.config.info_mean:
                kegg_edges = kegg_edges * info_mean
            
            edge_matrix.append(kegg_edges)
        
        if self.config.go: 
            logger.info(f"Generating GO edges...")
            feature_omic = load_omic_features_name(
                "./artifacts/data_preprocessing/" , dataset=self.dataset, type=[1,2,3]
            )
            feature_conversion = load_feature_conversion(
                "./artifacts/data_ingestion/unzip/" , dataset=self.dataset
            )
            features = feature_omic.merge(feature_conversion , left_on="gene_name" , right_on="gene" , how="left")
            kegg_edges = generate_edges_from_annotation(
                f"./artifacts/data_ingestion/unzip/{self.dataset}_kegg_go/BP_DIRECT.txt", 
                features=features, 
                filter_p_value=self.config.kegg_filter
            )
            kegg_edges = kegg_edges[self.selected_gene][:, self.selected_gene]
            
            if self.config.information and self.config.info_mean:
                kegg_edges = kegg_edges * info_mean
            edge_matrix.append(kegg_edges)
        
        edge_matrix = torch.stack(edge_matrix , dim=-1)
        
        # scale the edge matrix to 0-1
        if self.config.scale_edge:
            edge_matrix = (edge_matrix - edge_matrix.min()) / (edge_matrix.max() - edge_matrix.min())
        
        logger.info(f"Generating graph data for training...")
        train_graph = []
        
        with tqdm(total=self.train_data_ac.shape[0]) as pbar:
            for idx , sample in self.train_data_ac.iterrows():
                torch_sample = torch.tensor(sample.values , dtype=torch.float32).unsqueeze(-1)
                if self.config.discretized and self.config.n_bins > 2: 
                    torch_sample = one_hot(torch_sample.long() , num_classes=self.config.n_bins).squeeze(1).float()
                else:
                    assert len(torch_sample.shape) == 2 , "Only support 1D tensor"

                graph = symmetric_matrix_to_pyg(
                    matrix=edge_matrix, 
                    node_features=torch_sample,
                    y=torch.tensor(self.train_label.loc[idx].values , dtype=torch.long),
                    edge_threshold=self.config.filter
                )
                train_graph.append(graph)
                pbar.update(1)
        
        logger.info(f"Generating graph data for testing...")
        test_graph = []
        with tqdm(total=self.test_data_ac.shape[0]) as pbar:
            for idx , sample in self.test_data_ac.iterrows():
                torch_sample = torch.tensor(sample.values , dtype=torch.float32).unsqueeze(-1)
                if self.config.discretized and self.config.n_bins > 2: 
                    torch_sample = one_hot(torch_sample.long() , num_classes=self.config.n_bins).squeeze(1).squeeze(1).float()
                else: 
                    assert len(torch_sample.shape) == 2 , "Only support 1D tensor"

                graph = symmetric_matrix_to_pyg(
                    matrix=edge_matrix, 
                    node_features=torch_sample ,
                    y=torch.tensor(self.test_label.loc[idx].values , dtype=torch.long),
                    edge_threshold=self.config.filter
                )
                test_graph.append(graph)
                pbar.update(1)
        
        os.makedirs("./artifacts/amogel" , exist_ok=True)
        torch.save(train_graph , "./artifacts/amogel/train_graph.pt")
        torch.save(test_graph , "./artifacts/amogel/test_graph.pt")
        
        try:
            logger.info(f"Node dimension: {test_graph[0].x.shape} , Edge dimension: {test_graph[0].edge_index.shape} , \
                    Edge attribute dimension: {test_graph[0].edge_attr.shape} , \
                    Edge max: {test_graph[0].edge_attr.max(dim=0).values} , \
                    Edge mean: {test_graph[0].edge_attr.mean(dim=0)} , \
                    Nonzero edge: {torch.count_nonzero(test_graph[0].edge_attr , dim=0)}")
        except Exception as e:
            logger.info(f"Node dimension: {test_graph[0].x.shape} , Edge dimension: {test_graph[0].edge_index.shape} , \
                    Edge attribute dimension: {test_graph[0].edge_attr.shape} , \
                    Edge mean: {test_graph[0].edge_attr.mean(dim=0)} , \
                    Nonzero edge: {torch.count_nonzero(test_graph[0].edge_attr , dim=0)}")
        
        mlflow.pytorch.autolog()
        mlflow.set_experiment("Graph Feature Selection")
        
        train_graph = torch.load("./artifacts/amogel/train_graph.pt")
        test_graph = torch.load("./artifacts/amogel/test_graph.pt")
        
        train_loader = DataLoader(train_graph , batch_size=32 , shuffle=True)
        test_loader = DataLoader(test_graph , batch_size=32 , shuffle=False)
        
        model = GCN(
            in_channels=train_graph[0].x.shape[1],
            hidden_channels=self.config.hidden_units,
            num_classes=self.train_label['label'].nunique(), 
            lr=self.config.learning_rate,
            drop_out=self.config.drop_out, 
            pooling_ratio=self.config.pooling_ratio
        )
        
        with mlflow.start_run():
            mlflow.log_params(self.config.__dict__)
            mlflow.log_params({
                "node_dim": train_graph[0].x.shape, 
                "edge_dim": train_graph[0].edge_index.shape,
                "edge_attr_dim": train_graph[0].edge_attr.shape,
                #"edge_attr_max": train_graph[0].edge_attr.max(dim=0).values,
                "nonzero_edge": torch.count_nonzero(train_graph[0].edge_attr , dim=0),
                "edge_attr_mean": train_graph[0].edge_attr.mean(dim=0)
            })
            mlflow.log_param("dataset" , self.dataset)
            trainer = Trainer(max_epochs=self.config.epochs)
            trainer.fit(model , train_loader , test_loader)
            
