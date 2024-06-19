# Implement KNN Classification on dataset
import pandas as pd
import os 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report , roc_auc_score
from amogel import logger
from amogel.model.DNN import DNN , OmicDataset
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

class OtherClassifier:
    
    def __init__(self, dataset="BRCA"):
        
        self.dataset = dataset 
        
        os.makedirs("./artifacts/compare/traditional" , exist_ok=True)

    def load_data(self):
        
        # load train data 
        train_data_omic_1 = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , f"1_tr.csv"), header=None)
        train_data_omic_2 = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , f"2_tr.csv"), header=None)
        train_data_omic_3 = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , f"3_tr.csv"), header=None)
        
        self.train_data = pd.concat([train_data_omic_1 ] , axis=1)
        
        # load train label 
        self.train_label = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , "labels_tr.csv") , header=None , names=["label"])
        self.num_classes = len(self.train_label['label'].unique())
        
        
        # load test data 
        test_data_omic_1 = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , f"1_te.csv"), header=None)
        test_data_omic_2 = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , f"2_te.csv"), header=None)
        test_data_omic_3 = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , f"3_te.csv"), header=None)
        
        self.test_data = pd.concat([test_data_omic_1 ] , axis=1)
        
        # load test label
        self.test_label = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , "labels_te.csv") , header=None , names=["label"])
        
        logger.info("Data dimension for training and testing")
        logger.info(f"Train data: {self.train_data.shape}")
        logger.info(f"Train label: {self.train_label.shape}")
        logger.info(f"Test data: {self.test_data.shape}")
        logger.info(f"Test label: {self.test_label.shape}")
        
    
    def train_and_evaluate_knn(self):
        # fit the model and evaluate using KNN from sklearn
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(self.train_data , self.train_label)
        pred = knn.predict(self.test_data)
        pred_prob = knn.predict_proba(self.test_data)
        output  = classification_report(self.test_label , pred , digits=4)
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