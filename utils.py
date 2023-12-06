import os
import pandas as pd
import torch 
from torch_geometric.data import Data
import torch_geometric.utils as geom_utils 
from typing import List
from tqdm import tqdm
from sklearn import feature_selection , preprocessing 
from sklearn import metrics
import itertools
import numpy as np
from scipy.sparse import coo_matrix

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
            x_scaled = preprocessing.scale(x , with_std=True)
            
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

def get_PPI_info():
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
    
    return PPI_merge

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

# Function to convert COO matrix to PyTorch Geometric Data object
def coo_to_pyg_data(coo_matrix , node_features , label):
    values = torch.FloatTensor(coo_matrix.data).unsqueeze(1)
    indices = torch.LongTensor(np.vstack((coo_matrix.row, coo_matrix.col)))
    size = torch.Size(coo_matrix.shape)

    indices , values = geom_utils.to_undirected(indices , values)
    
    return Data(x=node_features, edge_index=indices, edge_attr=values, num_nodes=size[0] , y=label)

def get_omic_graph(feature_path , conversion_path , label_path):
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "BRCA")
    david_path = os.path.join(os.path.dirname(os.path.realpath(__file__)) , "david")

    feature1 = os.path.join(base_path, feature_path)
    df1 = read_features_file(feature1)
    name1 = os.path.join(david_path, conversion_path)
    df1_header = pd.read_csv(name1)
    labels = read_features_file(os.path.join(base_path, label_path))[0].values
    
    omic_1_name = df1_header['official gene symbol'].to_list()
    omic_1_id = df1_header['gene id'].astype('Int64').astype('str').to_list()
    omic_1 = np.zeros((len(omic_1_name) , len(omic_1_name)))
    
    # Generate ppi/kegg/go transaction
    ppi_info = get_PPI_info()
    filter_PPI = ppi_info[ppi_info['protein1_name'].isin(omic_1_name)]
    filter_PPI = filter_PPI[filter_PPI['protein2_name'].isin(omic_1_name)]
    
    vector_idx = np.array(list(zip([ omic_1_name.index(x) for x in filter_PPI['protein2_name'] ] , [ omic_1_name.index(x) for x in filter_PPI['protein1_name'] ])))
    ## Expectin vector_idx to have shape of [n , 2]
    if vector_idx.shape[0] > 0:
        omic_1[vector_idx[:,0] , vector_idx[:,1]] += 1
    
    kegg_go_df = pd.read_csv(os.path.join(david_path , "consol_anno_chart.tsv") , sep='\t')
    for idx ,  row in kegg_go_df.iterrows():
        related_genes = row['Genes'].split(", ")
        genes_idx = [ omic_1_id.index(x) for x in related_genes if x in omic_1_id]
        genes_idx.sort()
        if len(genes_idx) <= 1: 
            continue
        vector_idx = np.array([x for x in itertools.combinations(genes_idx , 2)])
        omic_1[vector_idx[:,0] , vector_idx[:,1]] += 1
        
    #coo = symmetric_matrix_to_coo(omic_1 , 0.1)
    #indices , values , size = coo_to_pyg_data(coo)
    #print("Generated len of edge: {}".format(values.shape[0]))
    
    mean_dict = torch.FloatTensor(df1.mean().to_list())
    graph_data  = []
    
    try: 
        
        
        pbar = tqdm(total=len(df1))
            
        pbar.set_description("Generate graph data")
        for idx , [ index , row ] in enumerate(df1.iterrows()):
            node_features = torch.FloatTensor(row.values)
            significant_node_indices = torch.nonzero(node_features > mean_dict, as_tuple=True)[0]
            
            subgraph_features = node_features[significant_node_indices]
            subgraph_adjacency = omic_1[significant_node_indices][: , significant_node_indices]
            subgraph_coo = symmetric_matrix_to_coo(subgraph_adjacency , 0.1)
            graph =coo_to_pyg_data(subgraph_coo , subgraph_features.unsqueeze(1) , torch.tensor(labels[idx] , dtype=torch.long))
            graph_data.append(graph)
            pbar.update(1)
        pbar.close()
        
        
    except Exception as e: 
        print("Error in generating graph")
        print(e)
        
    return graph_data
        
if __name__ == "__main__":
    
    
    # # read labels
    # labels = os.path.join(base_path, "labels_tr.csv")
    # df_labels = read_features_file(labels) 
    
    # ## mRNA Features
    print("Generating mRNA omic data graph")
    get_omic_graph('1_tr.csv' , '1_featname_conversion.csv' , 'labels_tr.csv')
    
    # ## miRNA Feature 
    # print("Generating miRNA omic data graph")
    # get_omic_graph('2_tr.csv' , '2_featname_conversion.csv')
    
    # # ## DNA Feature 
    # print("Generating DNA omic data graph")
    # get_omic_graph('3_tr.csv' , '3_featname_conversion.csv')
    # feature1 = os.path.join(base_path, "1_tr.csv")
    # df1 = read_features_file(feature1)
    # name1 = os.path.join(david_path, "1_featname_conversion.csv")
    # df1_header = pd.read_csv(name1)
    
    # omic_1_name = df1_header['official gene symbol'].to_list()
    # omic_1_id = df1_header['gene id'].astype('Int64').astype('str').to_list()
    
    # # Generate ppi/kegg/go transaction
    # kegg_go_df = pd.read_csv(os.path.join(david_path , "consol_anno_chart.tsv") , sep='\t')
    
    # omic_1 = np.zeros((1000 , 1000))
    
    # ppi_info = get_PPI_info()
    # filter_PPI = ppi_info[ppi_info['protein1_name'].isin(omic_1_name)]
    # filter_PPI = filter_PPI[filter_PPI['protein2_name'].isin(omic_1_name)]
    
    # vector_idx = np.array(list(zip([ omic_1_name.index(x) for x in filter_PPI['protein2_name'] ] , [ omic_1_name.index(x) for x in filter_PPI['protein1_name'] ])))
    # omic_1[vector_idx[:,0] , vector_idx[:,1]] += 1
    
    
    # i = 0
    # pbar = tqdm(total=len(kegg_go_df))
    # pbar.set_description("Generate graph data")
    
    # try: 
    #     for idx ,  row in kegg_go_df.iterrows():
    #         related_genes = row['Genes'].split(", ")
    #         genes_idx = [ omic_1_id.index(x) for x in related_genes if x in omic_1_id]
    #         genes_idx.sort()
    #         if len(genes_idx) <= 1: 
    #             continue
    #         vector_idx = np.array([x for x in itertools.combinations(genes_idx , 2)])
    #         omic_1[vector_idx[:,0] , vector_idx[:,1]] += 1
    #         i += 1 
    #         pbar.update(i)
    # except Exception as e: 
    #     print("Error occur")
    #     print(e)
    #     print(genes_idx)
    # finally: 
    #     print(omic_1)
    #     print(omic_1.sum())
    #     pbar.close()
        
    # coo = symmetric_matrix_to_coo(omic_1 , 0.1)
    # indice , values , size = coo_to_pyg_data(coo)
    # print(indice.shape)
    
    # gp1 = generate_graph(df1 , df1_header , df_labels[0].tolist(), threshold=0.25, rescale=True, integration='pearson' , use_quantile=True)
    
    # a = [0, 1 , 2 , 3]
    
    # combination = itertools.combinations(a , 2)
    # a = np.ones((4 ,4))
    # b = np.zeros_like(a)
    # c = np.array([list(x) for x in combination])
    # print(c)
    # b[c[:,0] , c[:,1]] = 1
    # print(b)
    # coo = symmetric_matrix_to_coo(b)
    # print(coo)
    
    
    