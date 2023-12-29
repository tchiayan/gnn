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
    
    return Data(x=node_features, edge_index=indices, edge_attr=values, num_nodes=size[0] , y=label , extra_label=torch.arange(node_features.size(0)))

def get_omic_graph(feature_path , conversion_path , ac_rule_path ,  label_path , weighted=True , filter_ppi = None , filter_p_value = None , significant_q = 0.5 , ppi=True , go_kegg=True , ac=True , correlation=False , k=50 , gene_info_only=False , remove_isolate_node=False , annotation_chart = './david/consol_anno_chart.tsv'):
    
    df1 = read_features_file(feature_path)
    df1_header = pd.read_csv(conversion_path)
    labels = read_features_file(label_path)[0].values
    
    omic_1_name = df1_header['official gene symbol'].to_list()
    omic_1_id = df1_header['gene id'].astype('Int64').astype('str').to_list()
    omic_1 = np.zeros((len(omic_1_name) , len(omic_1_name)))
    
    # Generate ppi/kegg/go transaction
    ppi_filter_genes = []
    if ppi:
        ppi_info = get_PPI_info()
        filter_PPI = ppi_info[ppi_info['protein1_name'].isin(omic_1_name)]
        filter_PPI = filter_PPI[filter_PPI['protein2_name'].isin(omic_1_name)]
        
        if filter_ppi is not None: 
            assert isinstance(filter_ppi , int)  , "PPI score must be integer type"
            filter_PPI = filter_PPI[filter_PPI['combined_score'] == filter_ppi]

        ppi_filter_genes.extend([ omic_1_name.index(x) for x in filter_PPI['protein1_name'] ])
        ppi_filter_genes.extend([ omic_1_name.index(x) for x in filter_PPI['protein2_name'] ])
        vector_idx = np.array(list(zip([ omic_1_name.index(x) for x in filter_PPI['protein2_name'] ] , [ omic_1_name.index(x) for x in filter_PPI['protein1_name'] ])))
        ## Expectin vector_idx to have shape of [n , 2]
        if vector_idx.shape[0] > 0:
            omic_1[vector_idx[:,0] , vector_idx[:,1]] += 1
    ppi_filter_genes = list(set(ppi_filter_genes))
    
    kegg_go_filter_genes = []
    if go_kegg:
        if annotation_chart.endswith('.tsv'):
            kegg_go_df = pd.read_csv(annotation_chart , sep='\t')
        else:
            kegg_go_df = pd.read_csv(annotation_chart , sep=',')
        
        if filter_p_value is not None:
            assert isinstance(filter_p_value , float) , "P value must be float type"
            kegg_go_df = kegg_go_df[kegg_go_df['PValue'] <= filter_p_value]
        
        for _ ,  row in kegg_go_df.iterrows():
            related_genes = row['Genes'].split(", ")
            genes_idx = [ omic_1_id.index(x) for x in related_genes if x in omic_1_id]
            kegg_go_filter_genes.extend([x for x in genes_idx])
            genes_idx.sort()
            if len(genes_idx) <= 1: 
                continue
            vector_idx = np.array([x for x in itertools.combinations(genes_idx , 2)])
            omic_1[vector_idx[:,0] , vector_idx[:,1]] += 1
    kegg_go_filter_genes = list(set(kegg_go_filter_genes))
    
    if not weighted: 
        omic_1 = (omic_1 > 0).astype(float)
        
    #coo = symmetric_matrix_to_coo(omic_1 , 0.1)
    #indices , values , size = coo_to_pyg_data(coo)
    #print("Generated len of edge: {}".format(values.shape[0]))
    
    mean_dict = torch.FloatTensor(df1.quantile(q=significant_q).to_list())
    # print(mean_dict)
    graph_data  = []
    node_per_graph = []
    edge_per_graph = []
    isolate_node_per_graph = []
    node_degree_per_graph = []
    
    ac_filter_genes = []
    if ac: 
        add_omic = {}
        rules = pd.read_csv(ac_rule_path,  names=['label' , 'support' , 'confidences' , 'itemset' , 'interestingness' ] , sep='\t')
        
        # Get each label 
        unique_labels = rules['label'].unique()
        
        #print(rules.head())
        #print(unique_labels)
        
        for label in unique_labels:
            add_omic[label] = np.zeros((len(omic_1_name) , len(omic_1_name)))
            
            sub_df = rules[rules['label'] == label]
            sub_df.sort_values(by='interestingness' , ascending=False)
            topk_df = sub_df.iloc[:k , :]
            
            related_genes = []
            for _ , row in topk_df.iterrows():
                gene_sets = [x.split(":")[0] for x in row['itemset'].split(",")]
                related_genes.extend(gene_sets)
            related_genes = [int(x) for x in list(set(related_genes))]
            ac_filter_genes.extend([ x for x in related_genes])
            related_genes.sort()
            
            #print(related_genes)
            
            vector_idx = np.array([x for x in itertools.combinations(related_genes , 2)])
            #print(vector_idx)
            #print(add_omic[label].shape)
            add_omic[label][vector_idx[:,0] , vector_idx[:,1]] += 1
            add_omic[label] = (add_omic[label] > 0).astype(float) # convert to 1 and zero only
            
            #print(topk_df.head())
    ac_filter_genes = list(set(ac_filter_genes))
    
    if correlation: 
        cosine_similarity  = df1.corr().to_numpy()
        
        # mask threshold < 0.80 to 0 
        mask = (cosine_similarity < 0.80).astype(float)
        cosine_similarity = cosine_similarity * mask
        print(cosine_similarity.shape)
        omic_1 += cosine_similarity
    
    try: 
        
        
        pbar = tqdm(total=len(df1))
        pbar.set_description("Generate graph data")
        for idx , [ index , row ] in enumerate(df1.iterrows()):
            node_features = torch.FloatTensor(row.values)
            significant_node_indices = torch.nonzero(node_features >= mean_dict, as_tuple=True)[0]
            
            
            subgraph_features = node_features[significant_node_indices]
            if ac and labels[idx] in add_omic: 
                new_omic = omic_1.copy() + add_omic[labels[idx]]
                subgraph_adjacency = new_omic[significant_node_indices][: , significant_node_indices]
            else:
                subgraph_adjacency = omic_1[significant_node_indices][: , significant_node_indices]
                
            subgraph_coo = symmetric_matrix_to_coo(subgraph_adjacency , 0.1)
            graph =coo_to_pyg_data(subgraph_coo , subgraph_features.unsqueeze(1) , torch.tensor(labels[idx] , dtype=torch.long))
            # caculate node degree 
            #print("Max value: " , torch.max(graph.edge_attr) )
            
            
            new_edge , new_attr , mask = geom_utils.remove_isolated_nodes(graph.edge_index , graph.edge_attr , num_nodes=graph.num_nodes)
            isolate_node_per_graph.append(graph.num_nodes - mask.sum().item())
            if remove_isolate_node:
                graph = Data(x=graph.x[mask], edge_index=new_edge, edge_attr=new_attr, num_nodes=graph.x[mask].size(0) , y=graph.y , extra_label=graph.extra_label[mask])
            
            node_per_graph.append(graph.num_nodes)
            edge_per_graph.append(graph.edge_index.shape[1])
            
            node_degree = geom_utils.degree(graph.edge_index[0] , graph.num_nodes)
            node_degree_per_graph.append(node_degree.float().mean().item())
            
            graph_data.append(graph)
            pbar.update(1)
        pbar.close()
        
        
    except Exception as e: 
        print("Error in generating graph")
        print(e)
    
    # print(node_per_graph)
    # print(edge_per_graph)
    avg_node_per_graph = np.mean(np.array(node_per_graph))
    avg_edge_per_graph = np.mean(np.array(edge_per_graph))
    # avg_nodedegree_per_graph = avg_edge_per_graph/avg_node_per_graph/avg_node_per_graph
    
    # remove nan value
    node_degree_per_graph = [x for x in node_degree_per_graph if not np.isnan(x)]
    avg_nodedegree = np.mean(np.array(node_degree_per_graph))
    
    avg_isolate_node_per_graph = np.mean(np.array(isolate_node_per_graph))
    
    if gene_info_only:
        # Visualize intersection of 3 filter genes 
        print("Number of PPI filter genes: {}".format(len(ppi_filter_genes)))
        print("Number of GO/KEGG filter genes: {}".format(len(kegg_go_filter_genes)))
        print("Number of AC filter genes: {}".format(len(ac_filter_genes)))
        print("Number of intersection between PPI and GO/KEGG: {}".format(len(set(ppi_filter_genes).intersection(set(kegg_go_filter_genes)))))
        print("Number of intersection between PPI and AC: {}".format(len(set(ppi_filter_genes).intersection(set(ac_filter_genes)))))
        print("Number of intersection between GO/KEGG and AC: {}".format(len(set(kegg_go_filter_genes).intersection(set(ac_filter_genes)))))
        print("Common genes between PPI and GO/KEGG and AC: {}".format(len(set(ppi_filter_genes).intersection(set(kegg_go_filter_genes)).intersection(set(ac_filter_genes)))))
    
    return graph_data , avg_node_per_graph , avg_edge_per_graph , avg_nodedegree , avg_isolate_node_per_graph  , (ppi_filter_genes , kegg_go_filter_genes , ac_filter_genes)
        
if __name__ == "__main__":
    
    
    # # read labels
    # labels = os.path.join(base_path, "labels_tr.csv")
    # df_labels = read_features_file(labels) 
    
    # ## mRNA Features
    # print("Generating mRNA omic data graph")
    # _  , avgnodepergraph , avgnoedge , avgnodedegree , avgisolatenode , _ = get_omic_graph('BRCA/1_tr.csv' , 'david/1_featname_conversion.csv' ,'david/ac_rule_1.tsv' , 'BRCA/labels_tr.csv' , weighted=False , filter_ppi=None , filter_p_value=None , significant_q=0 , ppi=None , go_kegg=None , ac=True , k=50 , remove_isolate_node=True)
    # print(f"Omic data type 1: avg node per graph - {avgnodepergraph:.2f} , avg edge per graph - {avgnoedge:.2f} , avg node degree per grap - {avgnodedegree:.2f} , avg isolate node per graph - {avgisolatenode:.2f}")
    
    
    # # ## miRNA Feature 
    # print("Generating miRNA omic data graph")
    # _  , avgnodepergraph , avgnoedge , avgnodedegree , avgisolatenode , _ = get_omic_graph('BRCA/2_tr.csv' , 'david/2_featname_conversion.csv' ,'david/ac_rule_2.tsv' , 'BRCA/labels_tr.csv' , weighted=False , filter_ppi=None , filter_p_value=None , significant_q=0 , ppi=None , go_kegg=None , ac=True , k=50 , remove_isolate_node=True)
    # print(f"Omic data type 2: avg node per graph - {avgnodepergraph:.2f} , avg edge per graph - {avgnoedge:.2f} , avg node degree per grap - {avgnodedegree:.2f} , avg isolate node per graph - {avgisolatenode:.2f}")
    
    # # # ## DNA Feature 
    # print("Generating DNA omic data graph")
    # _  , avgnodepergraph , avgnoedge , avgnodedegree , avgisolatenode , _ = get_omic_graph('BRCA/3_tr.csv' , 'david/3_featname_conversion.csv' ,'david/ac_rule_3.tsv' , 'BRCA/labels_tr.csv' , weighted=False , filter_ppi=None , filter_p_value=None , significant_q=0 , ppi=None , go_kegg=None , ac=True , k=50 , remove_isolate_node=True)
    # print(f"Omic data type 3: avg node per graph - {avgnodepergraph:.2f} , avg edge per graph - {avgnoedge:.2f} , avg node degree per grap - {avgnodedegree:.2f} , avg isolate node per graph - {avgisolatenode:.2f}")
    
    
    # ## mRNA Features
    print("Generating mRNA omic data graph")
    _  , avgnodepergraph , avgnoedge , avgnodedegree , avgisolatenode , _ = get_omic_graph('KIPAN/1_tr.csv' , 'KIPAN/1_featname_conversion.csv' ,'KIPAN/ac_rule_1.tsv' , 'KIPAN/labels_tr.csv' , weighted=False , filter_ppi=None , filter_p_value=None , significant_q=0 , ppi=None , go_kegg=None , ac=True , k=50 , remove_isolate_node=True , annotation_chart="KIPAN/consol_anno_chart.csv")
    print(f"Omic data type 1: avg node per graph - {avgnodepergraph:.2f} , avg edge per graph - {avgnoedge:.2f} , avg node degree per grap - {avgnodedegree:.2f} , avg isolate node per graph - {avgisolatenode:.2f}")
    
    
    # ## miRNA Feature 
    print("Generating miRNA omic data graph")
    _  , avgnodepergraph , avgnoedge , avgnodedegree , avgisolatenode , _ = get_omic_graph('KIPAN/2_tr.csv' , 'KIPAN/2_featname_conversion.csv' ,'KIPAN/ac_rule_2.tsv' , 'KIPAN/labels_tr.csv' , weighted=False , filter_ppi=None , filter_p_value=None , significant_q=0 , ppi=None , go_kegg=None , ac=True , k=50 , remove_isolate_node=True , annotation_chart="KIPAN/consol_anno_chart.csv")
    print(f"Omic data type 2: avg node per graph - {avgnodepergraph:.2f} , avg edge per graph - {avgnoedge:.2f} , avg node degree per grap - {avgnodedegree:.2f} , avg isolate node per graph - {avgisolatenode:.2f}")
    
    # # ## DNA Feature 
    print("Generating DNA omic data graph")
    _  , avgnodepergraph , avgnoedge , avgnodedegree , avgisolatenode , _ = get_omic_graph('KIPAN/3_tr.csv' , 'KIPAN/3_featname_conversion.csv' ,'KIPAN/ac_rule_3.tsv' , 'KIPAN/labels_tr.csv' , weighted=False , filter_ppi=None , filter_p_value=None , significant_q=0 , ppi=None , go_kegg=None , ac=True , k=50 , remove_isolate_node=True , annotation_chart="KIPAN/consol_anno_chart.csv")
    print(f"Omic data type 3: avg node per graph - {avgnodepergraph:.2f} , avg edge per graph - {avgnoedge:.2f} , avg node degree per grap - {avgnodedegree:.2f} , avg isolate node per graph - {avgisolatenode:.2f}")
    
    
    