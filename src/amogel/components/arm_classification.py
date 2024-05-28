import pandas as pd 
import os
import pickle
import numpy as np
from sklearn.metrics import classification_report 
from amogel.entity.config_entity import ARMClassificationConfig
from amogel import logger
import warnings

warnings.filterwarnings("ignore")

class ARM_Classification():
    
    def __init__(self , config:ARMClassificationConfig ,  dataset:str , omic_type: int , topk: int = 50) -> None:
        
        self.config = config
        self.df_label = self._load_labels(dataset)
        self.df_test = self._load_test_data(dataset , omic_type)
        self.df_ac = self._load_arm(omic_type , dataset , topk)
            
    
    
    def _load_arm(self , omic_type:int , dataset:str , topk=50 ) -> pd.DataFrame: 
        """Load ARM

        Args:
            omic_type (int): _description_

        Returns:
            pd.DataFrame: _description_
        """
        
        filepath = os.path.join("./artifacts/data_preprocessing" , dataset , f"ac_rule_{omic_type}.tsv")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath , sep="\t" , header=None , names=["label" , "confidence" , "support" , "rules" , "interestingness"])

        grouped_top_tr = df.groupby("label").apply(lambda x: x.nlargest(topk , self.config.metric)).reset_index(drop=True)
        
        return grouped_top_tr
    
    def _print_summary(self , df_ac:pd.DataFrame)->None: 
        """Print Summary

        Args:
            df_ac (pd.DataFrame): _description_
        """
        
        # class summary 
        class_summary = {}
        
        for label in df_ac['label'].unique():
            filtered = df_ac[df_ac['label'] == label]
            
            for idx , row in filtered.iterrows():
                antecedents = [x.split(":")[0] for x in row['rules'].split(",")]
                
                for antecedent in antecedents:
                    if antecedent not in class_summary[label]:
                        class_summary[label][antecedent] = 1
                    else:
                        class_summary[label][antecedent] += 1
        
        # print summary
        for key in class_summary.keys():
            print(f"Class: {key} has {len(class_summary[key])} uniques antecedents")
            
    
    def _load_labels(self , dataset:str)->pd.DataFrame: 
        
        filepath = os.path.join(f"./artifacts/data_preprocessing" , dataset , "labels_te.csv" )
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath , sep="\t", header=None, names=["label"])
        
        return df        
    
    def _load_test_data(self , dataset:str , omic_type: int) -> pd.DataFrame:
        
        filepath = os.path.join(f"./artifacts/data_preprocessing" , dataset , f"{omic_type}_te.csv")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        df = pd.read_csv(filepath , header=None)
        
        discretized_filepath = os.path.join(f"./artifacts/data_preprocessing" , dataset , f"kbins_{omic_type}.joblib")
        if not os.path.exists(discretized_filepath):
            raise FileNotFoundError(f"File not found: {discretized_filepath}")
        with open(discretized_filepath , "rb") as f:
            est = pickle.load(f)
        
        df = pd.DataFrame(est.transform(df) , columns=df.columns)
        return df
    
    
    def test_arm(self) -> None: 
        
        testing_model = {}
        
        for label in self.df_ac['label'].unique():
            filtered = self.df_ac[self.df_ac['label'] == label]
            
            testing_model[label] = []
            
            for idx , row in filtered.iterrows():
                rule = set(row['rules'].split(","))
                testing_model[label].append(rule)
                
            
        classification_summary= []
        
        for idx , row in self.df_test.iterrows():
            # test each rules 
            sample = set([f"{x[0]}:{x[1]}" for x in list(zip(row.index , row.values))])
            summary = {}
            for label in testing_model.keys():
                summary[label] = []
                
                for rule in testing_model[label]:
                    insersection_set = rule.intersection(sample)
                    summary[label].append(len(insersection_set)/len(rule))
                    
                summary[label] = np.mean(summary[label])
            
            # select max value key
            summary['prediction'] = max(summary, key=summary.get)
            summary['sample'] = idx
            classification_summary.append(summary)
            
        df_prediction = pd.DataFrame(classification_summary)
        
        report = classification_report(self.df_label, df_prediction['prediction'], output_dict=True)
        return report['accuracy']
