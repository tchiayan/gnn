import os
from box import ConfigBox 
from pathlib import Path
from amogel import logger   
import yaml
import torch
import numpy as np
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from scipy.sparse import coo_matrix
from typing import List
import pandas as pd

def read_yaml(path_to_yaml: Path) -> ConfigBox: 
    """reads yaml file and returns

    Args:
        path_to_yaml (Path): path like input

    Returns:
        ConfigBox: ConfigBox type
    """
    
    try: 
        
        with open(path_to_yaml , "r") as file: 
            logger.info(f"Reading config file from {path_to_yaml}")
            content = yaml.safe_load(file)
            logger.info("Config file read successfully")
            config = ConfigBox(content)
            return config
    except Exception as e:
        raise e
    

def coo_to_pyg_data(coo_matrix , node_features , y=None , extra_label=False):
    values = torch.FloatTensor(coo_matrix.data).unsqueeze(1)
    indices = torch.LongTensor(np.vstack((coo_matrix.row, coo_matrix.col)))
    size = torch.Size(coo_matrix.shape)

    indices , values = to_undirected(indices , values)
    
    if y is not None:
        if not extra_label:
            return Data(x=node_features, edge_index=indices, edge_attr=values, num_nodes=size[0] , y=y)
        else:
            return Data(x=node_features, edge_index=indices, edge_attr=values, num_nodes=size[0] , y=y , extra_label=torch.arange(node_features.size(0)))
    else:
        return Data(x=node_features, edge_index=indices, edge_attr=values, num_nodes=size[0])

def symmetric_matrix_to_pyg(matrix , node_features , y , edge_threshold=0.0):
    rows, cols = np.triu_indices_from(np.ones((matrix.shape[0] , matrix.shape[1])))
    
    data = matrix[rows, cols] # [number of edges , number of features]
    if len(data.shape) == 1:
        data = data.reshape(-1,1)
        
    indices , values = to_undirected(torch.LongTensor(np.vstack((rows, cols))) , torch.FloatTensor(data) , reduce="mean")
    
    ## Filter the edges with all features is more than 0 
    mask = torch.any(values > edge_threshold , dim=-1)
    indices = indices[:,mask]
    values = values[mask]
    
    return Data(x=node_features , edge_index=indices , edge_attr=values , num_nodes=node_features.shape[0] , y=y , extra_label=torch.arange(node_features.size(0)))
    

def symmetric_matrix_to_coo(matrix , threshold):
    # Find the nonzero entries in the upper triangle (including the main diagonal)
    rows, cols = np.triu_indices_from(matrix, k=0)

    # Extract the corresponding values from the upper triangle
    data = matrix[rows, cols]

    # Filter entries based on the threshold
    mask = abs(data) >= threshold
    rows = rows[mask]
    cols = cols[mask]
    data = data[mask]
    
    # Create a COO matrix
    coo = coo_matrix((data, (rows, cols)), shape=matrix.shape)

    return coo

def load_feature_conversion(dif:str , dataset:str): 
    
    feature_conversion_path = os.path.join(dif , f"{dataset}_kegg_go" , "featname_conversion.csv")
    
    if not os.path.exists(feature_conversion_path):
        raise FileNotFoundError(f"Feature conversion file not found at {feature_conversion_path}")
    
    feature_conversion = pd.read_csv(feature_conversion_path)
    feature_conversion['id'] = feature_conversion['id'].astype("Int64")
    
    return feature_conversion

def load_omic_features_name(dir:str , dataset:str , type:List[int]):
    """ 
    Load omic features name from the directory
    
    Args:
        dir: str : directory path
        dataset: str : dataset name
        type: List[int] : list of type of features
    
    Returns:
        pd.DataFrame : DataFrame with gene_loc and gene_name
    """
    df_features = []
    for i in type:
        feature_filepath = os.path.join(dir , dataset , f"{i}_featname.csv")
    
        df_feature = pd.read_csv(feature_filepath , header=None)
        
        if i == 1:
            df_feature['gene_name'] = df_feature[0].apply(lambda x: x.split("|")[0])
        elif i == 3:
            df_feature['gene_name'] = df_feature[0].str.replace(r'(hsa-|-)', '', regex=True)
            df_feature['gene_name'] = df_feature['gene_name'].apply(lambda x: "mir" + x if "let" in x else x)
        else:
            df_feature['gene_name'] = df_feature[0]
        df_features.append(df_feature)
        
    # merge all features
    df_features = pd.concat(df_features)
    df_features.reset_index(inplace=True)
    df_features['gene_idx'] = df_features.index.to_list()
    
    return df_features[["gene_idx" , "gene_name"]]
    
def load_ppi(dir:str, df_features_name:pd.DataFrame, protein_score:int=400):
    ppi_info_path = os.path.join(dir, f"protein_info.parquet.gzip")
    ppi_link_path = os.path.join(dir, f"protein_links.parquet.gzip")
    
    if not os.path.exists(ppi_info_path):
        raise FileNotFoundError(f"PPI info file not found at {ppi_info_path}")
    
    if not os.path.exists(ppi_link_path):
        raise FileNotFoundError(f"PPI link file not found at {ppi_link_path}")
    
    df_protein = pd.read_parquet(ppi_info_path)
    df_protein_link = pd.read_parquet(ppi_link_path)
    
    df_protein_merged = pd.merge(df_protein_link, df_protein[['#string_protein_id','preferred_name']], left_on="protein1", right_on="#string_protein_id")
    df_protein_merged.rename(columns={"preferred_name":"protein1_name"}, inplace=True)

    df_protein_merged = pd.merge(df_protein_merged, df_protein[['#string_protein_id','preferred_name']], left_on="protein2", right_on="#string_protein_id")
    df_protein_merged.rename(columns={"preferred_name":"protein2_name"}, inplace=True)

    # drop columns
    df_protein_merged.drop(columns=["#string_protein_id_x", "#string_protein_id_y", "protein1" , "protein2"], inplace=True)
    df_protein_merged.head()
    

    df_protein_merged = df_protein_merged.merge(df_features_name[['gene_idx' , 'gene_name']] , left_on="protein1_name", right_on="gene_name" , how="left")
    df_protein_merged.rename(columns={"gene_idx":"gene1_idx"}, inplace=True)

    df_protein_merged = df_protein_merged.merge(df_features_name[['gene_idx' , 'gene_name']] , left_on="protein2_name", right_on="gene_name" , how="left")
    df_protein_merged.rename(columns={"gene_idx":"gene2_idx"}, inplace=True)

    df_protein_merged.drop(columns=["gene_name_x", "gene_name_y"], inplace=True)
    
    # filter rows with only gene1_idx and gene2_idx
    df_filter_protein = df_protein_merged[df_protein_merged['gene1_idx'].notnull()][df_protein_merged['gene2_idx'].notnull()]
    
    if protein_score > 0:
        df_filter_protein = df_filter_protein[df_filter_protein['combined_score'] >= protein_score]
        
    return df_filter_protein
    
    