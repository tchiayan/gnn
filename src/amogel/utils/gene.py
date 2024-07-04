#### 
# Get DAVID gene conversion
####

from typing import List , Dict
from pathlib import Path
import pandas as pd
import warnings
import itertools
import numpy as np
import torch
import os
from amogel.utils.common import load_omic_features_name , load_feature_conversion
from tqdm import tqdm
from torch_geometric.utils import to_dense_adj
warnings.filterwarnings("ignore")


# url = 'https://davidbioinformatics.nih.gov/webservice/services/DAVIDWebService?wsdl'
# print ('url=%s' % url)

# #
# # create a service client using the wsdl.
# #
# client = Client(url)
# client.wsdl.services[0].setlocation('https://davidbioinformatics.nih.gov/webservice/services/DAVIDWebService.DAVIDWebServiceHttpSoap11Endpoint/')

# #authenticate user email 
# client.service.authenticate('chia.tan@monash.edu')


def get_all_gene_list(filepaths:List[Path] , configs:List[str])->List[str]:
    dfs = []
    for i , filepath in enumerate(filepaths):
        df = pd.read_csv(filepath , sep="\t" , header=0).iloc[: , 0:1] # gene id on first column
        df.columns = ['gene']
        df["file_index"] = i
        if configs[i] == "symbol_id":
            df['id'] = df['gene'].apply(lambda x: int(x.split("|")[1]))# get only the 
            df['gene'] = df['gene'].apply(lambda x: x.split("|")[0]) # get only the gene 
        elif configs[i] == 'symbol':
            df['id'] = np.nan 
        elif configs[i] == 'mir':
            df['gene'] = df['gene'].str.replace(r'(hsa-|-)', '', regex=True)
            df['gene'] = df['gene'].apply(lambda x: "mir" + x if "let" in x else x)
            df['id'] = np.nan
        
        duplicated_genes_count = df[df.duplicated()]['gene'].unique().shape[0]
        print(f"Duplicate genes count in {filepath.name}: {duplicated_genes_count}")
        dfs.append(df)
        
    for ( i , j ) in itertools.combinations([i for i in range(len(filepaths))] , 2):
        set_i = set(dfs[i]['gene'].unique().tolist())
        set_j = set(dfs[j]['gene'].unique().tolist())
        
        print(f"Overlapped gene count between {filepaths[i].name} and {filepaths[j].name}: {len(set_i.intersection(set_j))} | {len(set_i)} | {len(set_j)}")
        
    all_genes = pd.concat(dfs)
    
    # count duplicates genes 
    duplicate_gene_count = all_genes[all_genes.duplicated()]['gene'].unique().shape[0]
    print(f"Duplicate genes count in all path: {duplicate_gene_count}")

    # export to csv
    # all_genes.drop_duplicates(keep='first', subset=['gene']).to_csv("./gene_list.csv" , index=False)
    # should filter ? gene only combine after aggregation
    unknown_filter_gene = all_genes[all_genes['gene'] == "?"]
    known_filter_gene = all_genes[all_genes['gene'] != "?"]
    known_filter_gene = known_filter_gene.groupby(['gene']).agg({"id":"first"}).reset_index()
    combined_filter_gene = pd.concat([unknown_filter_gene[['gene' , 'id']] , known_filter_gene])
    combined_filter_gene['id'] = combined_filter_gene['id'].astype("Int64")
    combined_filter_gene.to_csv("./gene_list.csv" , index=False)
    #all_genes.groupby(['gene']).agg({"id":"first"}).reset_index().to_csv("./gene_list.csv" , index=False)
    
    return all_genes['gene'].unique().tolist()

def generate_edges_from_annotation(kegg_filepath: Path , features:pd.DataFrame , filter_p_value = None , topk=None ) -> torch.Tensor: 
    
    if not os.path.exists(kegg_filepath):
        raise FileNotFoundError(f"File {kegg_filepath} not found")
    
    kegg_df = pd.read_csv(kegg_filepath , sep="\t")[['Genes' , 'PValue']]
    
    if filter_p_value is not None:
        kegg_df = kegg_df[kegg_df['PValue'] <= filter_p_value]
        
    if topk is not None:
        kegg_df = kegg_df.sort_values(by='PValue' , ascending=True).head(topk)

    
    kegg_df['Genes'] = kegg_df['Genes'].apply(lambda x: [int(n) for n in x.split(",")])
    
    available_genes = set(features['id'].unique().tolist())
    edge_tensor = torch.zeros(features.shape[0] , features.shape[0])
    partial_match = 0
    with tqdm(total=kegg_df.shape[0]) as pbar:
        for gene_list in kegg_df['Genes']:
            # find full match
            path = set(gene_list)
            
            if len(path.intersection(available_genes)) > 1: # partial match
                partial_match += 1 
                
                # get all the features idx 
                gene_loc = features[features['id'].isin(path.intersection(available_genes))]['gene_idx'].tolist()
                for i , j in itertools.combinations(gene_loc , 2):
                    edge_tensor[i , j] = 1
                    edge_tensor[j , i] = 1
            pbar.update(1)
            
    assert (edge_tensor != edge_tensor.T).sum() == 0 , "Edge tensor is not symmetric"
    
    # calculate the number of edges
    tri_idx = np.triu_indices(n=features.shape[0] , k=0)
    non_zero_edges = (edge_tensor[tri_idx[0] , tri_idx[1]] > 0).sum()
    print(f"Partial match annotation: {partial_match} | Number of non-zero edges: {non_zero_edges}")
    return edge_tensor

def get_gene_attention(batches , edge_attn):
    edge_attn_layer = []
    
    for i , batch in enumerate(batches):
        dense_attn = to_dense_adj(edge_attn[i][0] , batch , edge_attr=edge_attn[i][1]).squeeze(dim=-1)
        edge_attn_layer.append(dense_attn)
    
    edge_attn_layer = torch.concat(edge_attn_layer , dim=0)
    return edge_attn_layer 

def biomarkers_selection(batch_file , edge_attn_files , topk=10):
    
    batches = torch.load(batch_file , map_location=torch.device('cpu'))
    edge_attn_layers = []
    
    for edge_attn_file in edge_attn_files:
        edge_attn = torch.load(edge_attn_file , map_location=torch.device('cpu'))
        edge_attn_layer = get_gene_attention(batches , edge_attn)
        edge_attn_layers.append(edge_attn_layer)
    
    edge_attn_layers = torch.stack(edge_attn_layers , dim=-1)
    
    biomarkers = edge_attn_layers.sum(dim=-1).mean(dim=0).sum(dim=-1).topk(topk)
    return biomarkers
  
if __name__ == "__main__":
    
    
    # feature_omic = load_omic_features_name(
    #     "./artifacts/data_preprocessing/", 
    #     "BRCA" , 
    #     [1 , 2 , 3]
    # )
    # feature_conversion = load_feature_conversion(
    #     "./artifacts/data_ingestion/unzip/",
    #     "BRCA",
    # )
    # features = feature_omic.merge(feature_conversion , left_on="gene_name" , right_on="gene" , how="left")
    # generate_edges_from_kegg(
    #     "./artifacts/data_ingestion/unzip/BRCA_kegg_go/KEGG_PATHWAY_BRCA.txt" , 
    #     features=features, 
    #     filter_p_value=0.05
    # )
    
    
    # test 
    filepaths = [
        Path("./artifacts/data_ingestion/unzip/LUSC/mRNA/LUSC.uncv2.mRNAseq_RSEM_all.txt") , 
        Path("./artifacts/data_ingestion/unzip/LUSC/DNA/LUSC.meth.by_mean.data.txt"),
        Path("./artifacts/data_ingestion/unzip/LUSC/miRNA/LUSC.miRseq_RPKM.txt")
    ]
    
    configs = [
        "symbol_id", 
        "symbol", 
        "mir",
    ]
    
    genes_symbol = get_all_gene_list(filepaths , configs)
    
    