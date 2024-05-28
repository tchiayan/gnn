from amogel import logger 
from amogel.entity.config_entity import DataPreprocessingConfig
from amogel.utils.ac import generate_ac_to_file
import pandas as pd
from typing import List , Tuple
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold , SelectKBest , f_classif
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from pickle import dump
import warnings

warnings.filterwarnings("ignore")

import os 

class DataPreprocessing:
    
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
        
    def load_data(self , dataset: str)->Tuple[pd.DataFrame , pd.DataFrame , pd.DataFrame , pd.DataFrame]:
        """Load data from the given dataset

        Args:
            dataset (str): _description_

        Returns:
            Tuple[pd.DataFrame , pd.DataFrame , pd.DataFrame , pd.DataFrame]: Load miRNA, mRNA, DNA and label data
        """
        
        assert dataset in ['BRCA' , 'KIPAN'] , "Invalid dataset"
        
        logger.info(f"Loading dataset {dataset}")

        df_miRNA = self.load_data_from_path(self.config[dataset]['miRNA'])
        df_mRNA = self.load_data_from_path(self.config[dataset]['mRNA'])
        df_DNA = self.load_data_from_path(self.config[dataset]['DNA'])
        
        # inspect 
        miRNA_columns = df_miRNA.columns
        miRNA_patient_ids = ["-".join(x.split("-")[0:3]) for x in miRNA_columns]
        logger.info(f"Dataset: {dataset} | Total number of miRNA samples: {len(miRNA_patient_ids)} | Total number of unique miRNA samples: {len(set(miRNA_patient_ids))}")
        
        mRNA_columns = df_mRNA.columns
        mRNA_patient_ids = ["-".join(x.split("-")[0:3]) for x in mRNA_columns]
        logger.info(f"Dataset: {dataset} | Total number of mRNA samples: {len(mRNA_patient_ids)} | Total number of unique mRNA samples: {len(set(mRNA_patient_ids))}")
        
        DNA_columns = df_DNA.columns
        DNA_patient_ids = ["-".join(x.split("-")[0:3]) for x in DNA_columns]
        logger.info(f"Dataset: {dataset} | Total number of DNA samples: {len(DNA_patient_ids)} | Total number of unique DNA samples: {len(set(DNA_patient_ids))}")
        
        # loading label 
        df_label = self.load_label(self.config[dataset]['label'])
        
        return  df_miRNA , df_mRNA , df_DNA , df_label
    
    def load_label(self , label_path: Path):
        df_label = pd.read_csv(label_path , sep="\t")
        df_label = df_label.T 
        
        # set first row as header
        new_header = df_label.iloc[0]
        df_label = df_label[1:]
        df_label.columns = new_header
        df_label.reset_index(inplace=True)


        # to upper case 
        df_label['index'] = df_label['index'].apply(lambda x: x.upper())
        
        return df_label 
        # filter common samples
        df_label = df_label[df_label['index'].isin(common_samples)]
        df_label = df_label[['index' , 'histological_type']]
        df_label.set_index('index', inplace=True)
        df_label.sort_index(inplace=True)
        
        return df_label
        
    @staticmethod
    def load_data_from_path(datapath: str):
        """
        Load data from the given path
        """
        
        df_original = pd.read_csv(datapath, sep="\t" , header=0)
        df_original.drop(0 , inplace=True)
        
        # rename column 0 to gene
        column_0 = df_original.columns[0]
        df_original.rename(columns={column_0 : "gene"} , inplace=True)
        df_original.set_index("gene" , inplace=True)
        
        return df_original
    
    def data_cleaning(self , df_miRNA: pd.DataFrame , df_mRNA: pd.DataFrame , df_DNA: pd.DataFrame , df_label: pd.DataFrame , target: str)->Tuple[pd.DataFrame , pd.DataFrame , pd.DataFrame]:
        """Data cleaning

        Args:
            df_miRNA (pd.DataFrame): _description_
            df_mRNA (pd.DataFrame): _description_
            df_DNA (pd.DataFrame): _description_

        Returns:
            Tuple[pd.DataFrame , pd.DataFrame , pd.DataFrame]: _description_
        """
        
        # remove duplicate sample
        logger.info("Removing duplicate samples")
        df_miRNA = self.remove_duplicate_sample(df_miRNA)
        df_mRNA = self.remove_duplicate_sample(df_mRNA)
        df_DNA = self.remove_duplicate_sample(df_DNA)
        logger.info(f"Duplicate sample removal | Shape of miRNA: {df_miRNA.shape} | Shape of mRNA: {df_mRNA.shape} | Shape of DNA: {df_DNA.shape}")
        
        
        # convert data type to float 
        df_miRNA = df_miRNA.astype(float)
        df_mRNA = df_mRNA.astype(float)
        df_DNA = df_DNA.astype(float)
        
        
        
        # filter common samples
        logger.info("Selecting only common samples")
        common_samples = self.correlate_samples([df_miRNA , df_mRNA , df_DNA])
        
        logger.info(f"Common sample count: {len(common_samples)}")
        df_common_miRNA = df_miRNA[df_miRNA.index.isin(common_samples)]
        df_common_mRNA = df_mRNA[df_mRNA.index.isin(common_samples)]
        df_common_DNA = df_DNA[df_DNA.index.isin(common_samples)]
        logger.info(f"Common sample filtering | Shape of miRNA: {df_common_miRNA.shape} | Shape of mRNA: {df_common_mRNA.shape} | Shape of DNA: {df_common_DNA.shape}")
        
        
        df_label = df_label[df_label['index'].isin(common_samples)]
        df_label.set_index('index', inplace=True)
        df_label.sort_index(inplace=True)
        logger.info(f"Common sample filtering | Shape of label: {df_label.shape}")
        
        # Filter missing labels 
        logger.info("Filtering missing labels")
        df_label = self.filter_missing_labels(df_label , target)
        df_common_miRNA = df_common_miRNA[df_common_miRNA.index.isin(df_label.index)]  
        df_common_mRNA = df_common_mRNA[df_common_mRNA.index.isin(df_label.index)]
        df_common_DNA = df_common_DNA[df_common_DNA.index.isin(df_label.index)]
        
        # Filtering missing values
        logger.info("Missing value filtering")
        df_common_miRNA = self.filtering_missing_values(df_common_miRNA)
        df_common_mRNA = self.filtering_missing_values(df_common_mRNA)
        df_common_DNA = self.filtering_missing_values(df_common_DNA)
        
        # Sorting the data by index 
        df_common_miRNA.sort_index(inplace=True)
        df_common_mRNA.sort_index(inplace=True)
        df_common_DNA.sort_index(inplace=True)
        df_label.sort_index(inplace=True)
        
        # Variance filtering 
        logger.info("Variance filtering")
        df_common_miRNA = self.filtering_variance(df_common_miRNA , threshold=self.config.preprocessing.miRNA.variance)
        df_common_mRNA = self.filtering_variance(df_common_mRNA , threshold=self.config.preprocessing.mRNA.variance)
        df_common_DNA = self.filtering_variance(df_common_DNA , threshold=self.config.preprocessing.DNA.variance)
        
        # ANOVA-F filtering
        logger.info("ANOVA-F filtering")
        df_common_miRNA = self.annovaf_filtering(df_common_miRNA , df_label , target=target , threshold=self.config.preprocessing.miRNA.annova)
        df_common_mRNA = self.annovaf_filtering(df_common_mRNA , df_label , target=target , threshold=self.config.preprocessing.mRNA.annova)
        df_common_DNA = self.annovaf_filtering(df_common_DNA , df_label , target=target , threshold=self.config.preprocessing.DNA.annova)
        
        # scale the data to 0-1
        logger.info("Scaling the data")
        df_common_miRNA = df_common_miRNA / df_common_miRNA.max().max()
        df_common_mRNA = df_common_mRNA / df_common_mRNA.max().max()
        
        df_common_DNA = df_common_DNA / df_common_DNA.max().max()
        
        
        return df_common_miRNA , df_common_mRNA , df_common_DNA , df_label[[target]]
    
    def save_data(self , df_miRNA: pd.DataFrame , df_mRNA: pd.DataFrame , df_DNA: pd.DataFrame , df_label: pd.DataFrame , dataset: str):
        """Save data to the given path

        Args:
            df_miRNA (pd.DataFrame): _description_
            df_mRNA (pd.DataFrame): _description_
            df_DNA (pd.DataFrame): _description_
            df_label (pd.DataFrame): _description_
            dataset (str): _description_
        """
        
        assert dataset in ['BRCA' , 'KIPAN'] , "Invalid dataset"
        
        logger.info(f"Saving dataset {dataset}")
        
        root_dir = self.config.root_dir
        os.makedirs(os.path.join(root_dir , dataset) , exist_ok=True)
        
       
        
        # ordinal encoder 
        enc = OrdinalEncoder()
        label = enc.fit_transform(df_label.iloc[: , 0:1])
        df_label.iloc[: , 0] = label
        
        # train test split 
        logger.info(f"Train test split for dataset: {self.config.test_split}")
        X_train, X_test, y_train, y_test = train_test_split(df_label.index, df_label.iloc[: , 0:1], test_size=self.config.test_split, random_state=42)
        
        df_miRNA.loc[X_train,:].to_csv(os.path.join(root_dir, dataset , "3_tr.csv") , index=False , header=False)
        df_miRNA.loc[X_test,:].to_csv(os.path.join(root_dir, dataset , "3_te.csv") , index=False , header=False)
        pd.DataFrame(df_miRNA.columns).to_csv(os.path.join(root_dir , dataset ,"3_featname.csv") , index=False , header=False)
        
        df_mRNA.loc[X_train,:].to_csv(os.path.join(root_dir, dataset , "1_tr.csv") , index=False , header=False)
        df_mRNA.loc[X_test,:].to_csv(os.path.join(root_dir, dataset , "1_te.csv") , index=False , header=False)
        pd.DataFrame(df_mRNA.columns).to_csv(os.path.join(root_dir, dataset ,"1_featname.csv") , index=False , header=False)
        
        df_DNA.loc[X_train,:].to_csv(os.path.join(root_dir, dataset , "2_tr.csv") , index=False , header=False )
        df_DNA.loc[X_test,:].to_csv(os.path.join(root_dir, dataset , "2_te.csv") , index=False , header=False)
        pd.DataFrame(df_DNA.columns).to_csv(os.path.join(root_dir , dataset ,"2_featname.csv") , index=False , header=False)
        
        y_train.to_csv(os.path.join(root_dir, dataset , "labels_tr.csv") , index=False , header=False)
        y_test.to_csv(os.path.join(root_dir, dataset , "labels_te.csv") , index=False , header=False)
        
    def remove_duplicate_sample(self , df:pd.DataFrame):
        """Remove duplicate sample
        1. Remove duplicated samples
        2. Transpose the dataframe (genes as columns and samples as rows)

        Args:
            df (_type_): _description_
        """
        
        assert isinstance(df , pd.DataFrame) , "Invalid input"
        
        # modified patient id => ie. TCGA-KL-8323-01 -> TCGA-KL-8323
        columns_name = df.columns 
        rename_columns_name = {x:"-".join(x.split("-")[0:3]) for x in columns_name}
        df = df.rename(columns=rename_columns_name)
        
        # filter out duplicated columns 
        df = df.loc[:,~df.columns.duplicated()] # rows as genes and columns as samples
        
        return df.T
    
    def correlate_samples(self , dfs: List[pd.DataFrame]):
        
        assert isinstance(dfs , list) , "Invalid input"
        assert len(dfs) > 0 , "Empty input is not allowed"
        
        # Get common samples (columns names) from all the dataframes
        common_samples = set(dfs[0].index)
        for df in dfs[1:]:
            common_samples = common_samples.intersection(set(df.index))
        
        # common_samples = list(common_samples.intersection(set(label_df)))
        
        return common_samples
    
    def filter_missing_labels(self , df: pd.DataFrame, target:str)->List[str]:
            
        assert isinstance(df , pd.DataFrame) , "Invalid input"
        assert target in df.columns , f"Invalid target column name, not found in the label dataframe | {df.columns.tolist()}"
        
        missing_labels = df[df[target].isnull()].index.tolist()
        
        #print(missing_labels)
        filter_df = df[~df.index.isin(missing_labels)]
        
        return filter_df
    
    def filtering_missing_values(self , df: pd.DataFrame)->pd.DataFrame:
        """Filter missing features

        Args:
            df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        
        assert isinstance(df , pd.DataFrame) , "Invalid input"
        
        default_shape = df.shape
        df_filtered = df.dropna(axis=1)
        filter_shape = df_filtered.shape
        
        logger.info(f"Missing value filtering | Default shape: {default_shape} | Filtered shape: {filter_shape}")
        
        return df_filtered
    
    def filtering_variance(self , df: pd.DataFrame , threshold:float): 
        
        default_shape = df.shape 
        
        sel = VarianceThreshold(threshold=threshold)
        df_filtered  = pd.DataFrame(sel.fit_transform(df) , index=df.index , columns=df.columns[sel.get_support()])
        
        filter_shape = df_filtered.shape
        logger.info(f"Variance filtering | Default shape: {default_shape} | Filtered shape: {filter_shape}")
        
        return df_filtered
    
    def annovaf_filtering(self , df: pd.DataFrame , df_label: pd.DataFrame , target: str , threshold: float)->pd.DataFrame:
        
        assert isinstance(df , pd.DataFrame) , "Invalid input"
        assert isinstance(df_label , pd.DataFrame) , "Invalid input"
        assert target in df_label.columns , f"Invalid target column name, not found in the label dataframe | {df_label.columns.tolist()}" 
        default_shape = df.shape
        
        sel = SelectKBest(score_func=f_classif, k=threshold)
        df_filtered = pd.DataFrame(sel.fit_transform(df, df_label[target]), index=df.index , columns=df.columns[sel.get_support()])
        
        filter_shape = df_filtered.shape
        logger.info(f"ANOVA-F filtering | Default shape: {default_shape} | Filtered shape: {filter_shape}")
        
        return  df_filtered
    
    def feature_conversion(self)->None: 
        pass 
    
    def generate_ac(self, dataset:str)-> None: 
        
        label_path = os.path.join(self.config.root_dir , dataset , "labels_tr.csv")
        for i in range(1,4):
            logger.info(f"Generate AC rules for dataset {dataset} | ac_rule_{i}.tsv")
            data_filepath = os.path.join(self.config.root_dir , dataset , f"{i}_tr.csv")
            est = generate_ac_to_file(data_filepath , label_path , os.path.join(self.config.root_dir , dataset , f"ac_rule_{i}.tsv") , min_rule=True , min_rule_per_class=5000)
            
            logger.info(f"Store the KbinsDiscretizer for dataset {dataset} | kbins_{i}.joblib")
            dump(est , open(os.path.join(self.config.root_dir , dataset , f"kbins_{i}.joblib"), 'wb'))
            
        # generate ac rules for test data
        # label_path = os.path.join(self.config.root_dir , dataset , "labels_te.csv")
        # for i in range(1,4):
        #     logger.info(f"Generate AC rules for dataset {dataset} | ac_rule_{i}_te.tsv")
        #     data_filepath = os.path.join(self.config.root_dir , dataset , f"{i}_te.csv")
        #     generate_ac_to_file(data_filepath , label_path , os.path.join(self.config.root_dir , dataset , f"ac_rule_{i}_te.tsv") , min_rule=True , min_rule_per_class=500)