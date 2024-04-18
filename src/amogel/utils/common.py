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
    

def coo_to_pyg_data(coo_matrix , node_features , y=None):
    values = torch.FloatTensor(coo_matrix.data).unsqueeze(1)
    indices = torch.LongTensor(np.vstack((coo_matrix.row, coo_matrix.col)))
    size = torch.Size(coo_matrix.shape)

    indices , values = to_undirected(indices , values)
    
    if y is not None:
        return Data(x=node_features, edge_index=indices, edge_attr=values, num_nodes=size[0] , y=y)
    else:
        return Data(x=node_features, edge_index=indices, edge_attr=values, num_nodes=size[0])


def symmetric_matrix_to_coo(matrix , threshold):
    # Find the nonzero entries in the upper triangle (including the main diagonal)
    rows, cols = np.triu_indices_from(matrix, k=0)

    # Extract the corresponding values from the upper triangle
    data = matrix[rows, cols]

    # Filter entries based on the threshold
    mask = data >= threshold
    rows = rows[mask]
    cols = cols[mask]
    data = data[mask]
    
    # Create a COO matrix
    coo = coo_matrix((data, (rows, cols)), shape=matrix.shape)

    return coo