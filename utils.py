import os
import pandas as pd
import torch 
from torch_geometric.data import Data
import torch_geometric.utils as geom_utils 
from typing import List
from tqdm import tqdm
from sklearn import feature_selection , preprocessing , scale
from sklearn import metrics
import itertools
import numpy as np

def read_features_file(path):
    df = pd.read_csv(path, header=None)
    return df

def generate_graph(df:pd.DataFrame , header_name:pd.DataFrame , labels:pd.DataFrame,  integration:str='PPI', threshold:int=300 , cached=None , rescale=False , use_quantile=False) -> List[Data]:
    
    if integration == 'PPI':
        
        
        # Get PPI data
        CACHED_PPI = r"PPI/ppi.parquet.gzip"
        
        if os.path.exists(CACHED_PPI):
            PPI_merge = pd.read_parquet(CACHED_PPI)
        else:
            PPI_filepath = r"PPI/9606.protein.links.v12.0.txt"
            PPI_annotation = r"PPI/9606.protein.info.v12.0.txt"

            ## Read PPI Annotation file 
            PPI_annotation = pd.read_csv(PPI_annotation , sep="\t")
            # print(PPI_annotation.head())

            ## Read PPI file
            PPI = pd.read_csv(PPI_filepath , sep=" ")
            # print(PPI.head())

            # convert protein id to gene name 
            PPI_merge = pd.merge(PPI , PPI_annotation , left_on="protein1" , right_on="#string_protein_id", how="left")
            PPI_merge.drop(columns=["#string_protein_id" , "protein_size" , "annotation"] , inplace=True , axis=1)
            PPI_merge.rename(columns={"preferred_name":"protein1_name"} , inplace=True)

            PPI_merge = pd.merge(PPI_merge , PPI_annotation , left_on="protein2" , right_on="#string_protein_id", how="left")
            PPI_merge.drop(columns=["#string_protein_id" , "protein_size" , "annotation"] , inplace=True , axis=1)
            PPI_merge.rename(columns={"preferred_name":"protein2_name"} , inplace=True)
            
            PPI_merge.to_parquet(CACHED_PPI , compression="gzip")
        
        ## Filter PPI Score > threshold (default : 600)
        PPI_merge = PPI_merge[PPI_merge["combined_score"] > threshold]
        
        GENE_NAME = [x.split("|")[0] for x in header_name[0].tolist()]
        edge_index = [[], []]
        edge_attr = []
        i = 0 
        
        pbar = tqdm(total=len(PPI_merge))
        pbar.set_description("Generate graph edge | Number of edges: {}".format(i))
        for idx  , row in PPI_merge.iterrows():
            if row["protein1_name"] in GENE_NAME and row["protein2_name"] in GENE_NAME:
                source_index = GENE_NAME.index(row["protein1_name"])
                target_index = GENE_NAME.index(row["protein2_name"])
                
                edge_index[0].append(source_index)
                edge_index[1].append(target_index)
                edge_index[0].append(target_index)
                edge_index[1].append(source_index)
                edge_attr.extend([row["combined_score"]] * 2)
        
                i += 2
                pbar.set_description("Number of edges: {}".format(i))
            pbar.update(1)
        pbar.close()
        
        ## Convert edge_index and edge_attr to tensor
        edge_index = torch.tensor(edge_index , dtype=torch.long)
        edge_attr = torch.tensor(edge_attr , dtype=torch.float).unsqueeze(1)
        
        # scale to 0 to 1 
        edge_min = torch.min(edge_attr)
        edge_max = torch.max(edge_attr)
        edge_attr = (edge_attr - edge_min) / ( edge_max - edge_min )
        
        edge_index , edge_attr = geom_utils.add_self_loops(edge_index=edge_index , edge_attr=edge_attr , fill_value=200. )
        
        assert edge_index.shape[1] == edge_attr.shape[0] , "Number of edges and edge attributes must be equal"
        
        graph_data  = []
        
        pbar = tqdm(total=len(df))
        # rescale 
        if rescale: 
            x = df.values 
            min_max_scalar = preprocessing.MinMaxScaler()
            x_scaled = min_max_scalar.fit_transform(x)
            df = pd.DataFrame(x_scaled)
            
        pbar.set_description("Generate graph data")
        for idx , [ index , row ] in enumerate(df.iterrows()):
            x = torch.tensor(row.values , dtype=torch.float).unsqueeze(1)
            assert x.shape[1] == 1 , "Feature dimension must be 1"
            graph = Data(x=x , edge_index=edge_index , edge_attr=edge_attr , y=torch.tensor(labels[idx] , dtype=torch.long))
            graph_data.append(graph)
            pbar.update(1)
        pbar.close()
        
        return graph_data
    
    elif integration == 'pearson' or integration == 'cosine_similarity':
        #coor = feature_selection.mutual_info_classif(df , labels)
        
        cosine_similarity = metrics.pairwise.cosine_similarity(df.T) if integration == 'cosine_similarity' else df.corr().to_numpy()
        
        if use_quantile:
            threshold = np.quantile(cosine_similarity , threshold)
        
        print(cosine_similarity.shape)
        c = 0 
        pbar = tqdm(total=len(cosine_similarity))
        pbar.set_description("Generate graph edge | Number of edges: {}".format(c))
        edge_index = [[], []]
        edge_attr = []
        for i in range(len(cosine_similarity)):
            filter = [ [j , cm]  for j , cm in enumerate(cosine_similarity[i]) if cm  >= threshold]
            for j , cm in filter: 
                edge_index[0].extend([i , j])
                edge_index[1].extend([j , i]) 
                edge_attr.extend([ cm ]*2)
            c += len(filter)*2
            pbar.set_description("Number of edges: {}".format(c))
            # for j in range(len(cosine_similarity)):
            #     if cosine_similarity[i][j] >= threshold:
            #         edge_index[0].extend([i , j])
            #         edge_index[1].extend([j , i]) 
            #         # edge_index[0].append(j)
            #         # edge_index[1].append(i)

            #         c += 2
            #         edge_attr.extend([ cosine_similarity[i][j] ]*2)
            #         pbar.set_description("Number of edges: {}".format(c))
            pbar.update(1)
        pbar.close()
        
        ## Convert edge_index and edge_attr to tensor
        edge_index = torch.tensor(edge_index , dtype=torch.long)
        edge_attr = torch.tensor(edge_attr , dtype=torch.float).unsqueeze(1)
        
        edge_index , edge_attr = geom_utils.add_self_loops(edge_index=edge_index , edge_attr=edge_attr , fill_value=1. )
        
        assert edge_index.shape[1] == edge_attr.shape[0] , "Number of edges and edge attributes must be equal"
        
        graph_data  = []
        
        if rescale: 
            x = df.values 
            # min_max_scalar = preprocessing.MinMaxScaler()
            # x_scaled = min_max_scalar.fit_transform(x)
            # df = pd.DataFrame(x_scaled)
            x_scaled = scale(x , with_std=True)
            
        pbar = tqdm(total=len(df))
        pbar.set_description("Generate graph data")
        for idx , [ index , row ] in enumerate(df.iterrows()):
            x = torch.tensor(row.values , dtype=torch.float).unsqueeze(1)
            assert x.shape[1] == 1 , "Feature dimension must be 1"
            graph = Data(x=x , edge_index=edge_index , edge_attr=edge_attr , y=torch.tensor(labels[idx] , dtype=torch.long))
            graph_data.append(graph)
            pbar.update(1)
        pbar.close()
        
        return graph_data
    
    elif integration == 'GO&KEGG':
        
        annotation_file_path = r'david/annotation_chart.tsv'
        anno_df = pd.read_csv(annotation_file_path , sep='\t')
        #print(anno_df.iloc[0:10 , 0: 10])
        header_file_path = r'david/3_featname_conversion.csv'
        header_name = pd.read_csv(header_file_path)
        header_name.dropna(inplace=True)
        #print(header_name.dtypes)
        
        header_name['gene id'] = header_name['gene id'].astype(int)
        header_name['gene id'] = header_name['gene id'].astype(str)
        header_name.reset_index(inplace=True)
        header_name.set_index('gene id' , inplace=True)
        #print(header_name)
        genes_id = header_name.index.to_list()
        
        # loop gene 
        pbar = tqdm(total=len(anno_df))
        i = 0
        pbar.set_description('Creating kegg/go gene connection {}/{}'.format(i , 0))
        gene_pair = {}
        for idx , row in anno_df.iterrows(): 
            genes = [ x.strip() for x in (row['Genes'].split(",")) ]
            # filter only related genes 
            genes = [ header_name.loc[x]['index'] for x in genes if x in genes_id ]
            genes.sort()
            
            if len(genes) > 0 :
                i += 1 
                for gene_paring in itertools.combinations(genes , 2):
                    if gene_paring not in gene_pair:
                        gene_pair[gene_paring] = 1 
                    else: 
                        gene_pair[gene_paring] += 1
                pbar.set_description('Creating kegg/go gene connection {}/{}'.format(i , idx))
            pbar.update(1)
        pbar.close()
        
        edge_index = [[], []]
        edge_attr = []
        
        for key , value in gene_pair.items():
            pairA , pairB = key
            edge_index[0].append(int(pairA))
            edge_index[1].append(int(pairB))
            edge_index[1].append(int(pairA))
            edge_index[0].append(int(pairB))
            edge_attr.extend([ int(value) ]*2)
            
        ## Convert edge_index and edge_attr to tensor
        edge_index = torch.tensor(edge_index , dtype=torch.long)
        edge_attr = torch.tensor(edge_attr , dtype=torch.float).unsqueeze(1)
        
        # scale to 0 to 1 
        edge_min = torch.min(edge_attr)
        edge_max = torch.max(edge_attr)
        edge_attr = (edge_attr - edge_min) / ( edge_max - edge_min )
        
        # print(edge_attr)
        # print(torch.min(edge_attr))
        # print(torch.max(edge_attr))
        edge_index , edge_attr = geom_utils.add_self_loops(edge_index=edge_index , edge_attr=edge_attr , fill_value=200. )
        
        assert edge_index.shape[1] == edge_attr.shape[0] , "Number of edges and edge attributes must be equal"
        
        graph_data  = []
        
        pbar = tqdm(total=len(df))
        # rescale 
        if rescale: 
            x = df.values 
            min_max_scalar = preprocessing.MinMaxScaler()
            x_scaled = min_max_scalar.fit_transform(x)
            df = pd.DataFrame(x_scaled)
            
        pbar.set_description("Generate graph data")
        for idx , [ index , row ] in enumerate(df.iterrows()):
            x = torch.tensor(row.values , dtype=torch.float).unsqueeze(1)
            assert x.shape[1] == 1 , "Feature dimension must be 1"
            graph = Data(x=x , edge_index=edge_index , edge_attr=edge_attr , y=torch.tensor(labels[idx] , dtype=torch.long))
            graph_data.append(graph)
            pbar.update(1)
        pbar.close()
        
        return graph_data
        
if __name__ == "__main__":
    
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "BRCA")
    
    # read labels
    labels = os.path.join(base_path, "labels_tr.csv")
    df_labels = read_features_file(labels) 

    ## mRNA Features
    feature1 = os.path.join(base_path, "1_tr.csv")
    df1 = read_features_file(feature1)
    name1 = os.path.join(base_path, "1_featname.csv")
    df1_header = read_features_file(name1)
    gp1 = generate_graph(df1 , df1_header , df_labels[0].tolist(), threshold=0.25, rescale=True, integration='pearson' , use_quantile=True)
    
    
    