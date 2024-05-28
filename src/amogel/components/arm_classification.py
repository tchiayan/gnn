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
        self.topk = topk
            
    
    
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
            
            class_summary[label] = {}
            for idx , row in filtered.iterrows():
                antecedents = [x for x in row['rules'].split(",")]
                
                for antecedent in antecedents:
                    if antecedent not in class_summary[label].keys():
                        class_summary[label][antecedent] = 1
                    else:
                        class_summary[label][antecedent] += 1
                        
        class_summary_distinct_set = {}
        for key in class_summary.keys():
            class_summary_distinct_set[key] = [set(class_summary[key].keys())]
        
        # find the distinct set compare with other class 
        for key in class_summary_distinct_set.keys():
            distinct_set = class_summary_distinct_set[key]
            for compared_key in class_summary_distinct_set.keys():
                if key != compared_key:
                    distinct_set = distinct_set.difference(class_summary_distinct_set[compared_key])
            print(f"Class: {key} has {len(distinct_set)} distinct antecedents compare to others")
        # print summary
        for key in class_summary.keys():
            print(f"Class: {key} has {len(class_summary[key])} uniques antecedents")
    
    def get_testing_model(self , distinct=False) -> dict[str , list[set]]:
        """Get Testing Model

        Returns:
            dict[str , list[set]]: _description_
        """
        
        testing_model = {}
        
        if not distinct:
            for label in self.df_ac['label'].unique():
                filtered = self.df_ac[self.df_ac['label'] == label]
                
                testing_model[label] = []
                
                for idx , row in filtered.iterrows():
                    rule = set(row['rules'].split(","))
                    testing_model[label].append(rule)
        else: 
            # class summary 
            class_summary = {}
            
            for label in self.df_ac['label'].unique():
                filtered = self.df_ac[self.df_ac['label'] == label]
                
                class_summary[label] = {}
                for idx , row in filtered.iterrows():
                    antecedents = row['rules'].split(",")
                    
                    for antecedent in antecedents:
                        if antecedent not in class_summary[label].keys():
                            class_summary[label][antecedent] = 1
                        else:
                            class_summary[label][antecedent] += 1
                            
            class_summary_distinct_set = {}
            for key in class_summary.keys():
                class_summary_distinct_set[key] = set([ k for k , v in class_summary[key].items() if v == self.topk])
            
            # find the distinct set compare with other class 
            for key in class_summary_distinct_set.keys():
                distinct_set = class_summary_distinct_set[key]
                for compared_key in class_summary_distinct_set.keys():
                    if key != compared_key:
                        distinct_set = distinct_set.difference(class_summary_distinct_set[compared_key])
                        
                testing_model[key] = [distinct_set]
        
        return testing_model
    
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
        
        testing_model = self.get_testing_model(distinct=True)
            
        classification_summary= []
        
        for idx , row in self.df_test.iterrows():
            # test each rules 
            sample = set([f"{x[0]}:{x[1]}" for x in list(zip(row.index , row.values))])
            summary = {}
            for label in testing_model.keys():
                summary[label] = []
                
                for rule in testing_model[label]:
                    insersection_set = rule.intersection(sample)
                    if len(rule) != 0:
                        summary[label].append(len(insersection_set)/len(rule))
                    else:
                        summary[label].append(0)
                    
                summary[label] = np.mean(summary[label])
            
            # select max value key
            summary['prediction'] = max(summary, key=summary.get)
            summary['sample'] = idx
            classification_summary.append(summary)
            
        df_prediction = pd.DataFrame(classification_summary)
        
        report = classification_report(self.df_label, df_prediction['prediction'], output_dict=True)
        
        
        # Generate union set of label 
        union_summary = []
        for label in testing_model.keys():
            union_set = set()
            for rule in testing_model[label]:
                union_set = union_set.union(rule)
            union_summary.append(len(union_set))
            
        return report['accuracy'] , union_summary