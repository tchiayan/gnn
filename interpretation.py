from basic import MultiGraphClassification , PairDataset , collate
from utils import get_omic_graph
import torch
import lightning.pytorch as pl
import torch_geometric.utils as geom_utils 
import torch_geometric.data as geom_data
from torch_geometric.explain import GNNExplainer , Explainer , ModelConfig
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_regression
from sklearn.manifold import TSNE
import math

activation = []
def hook(model , input , output):
    activation.append(output.detach())

def main():
    batch_size = 20
    omic_type = "omic1"
    gene_count = 1000
    
    # Load datasets (BRCA)
    omic_1 , _ , _ , _ , _ , _ = get_omic_graph(r'BRCA/1_te.csv', r'david/1_featname_conversion.csv' , r'david/ac_rule_1.tsv', r'BRCA/labels_te.csv' , False , None , None , 0 , True , True , True , False, 50 , False , False , r'david/consol_anno_chart.tsv')
    omic_2 , _ , _ , _ , _ , _ = get_omic_graph(r'BRCA/2_te.csv', r'david/2_featname_conversion.csv' , r'david/ac_rule_2.tsv', r'BRCA/labels_te.csv' , False , None , None , 0 , True , True , True , False, 50 , False , False , r'david/consol_anno_chart.tsv')
    omic_3 , _ , _ , _ , _ , _ = get_omic_graph(r'BRCA/3_te.csv', r'david/3_featname_conversion.csv' , r'david/ac_rule_3.tsv', r'BRCA/labels_te.csv' , False , None , None , 0 , True , True , True , False, 50 , False , False , r'david/consol_anno_chart.tsv')
    
    # Load the gene expression data 
    gene_exp = pd.read_csv(r'BRCA/1_te.csv' , header=None)
    label = pd.read_csv(r'BRCA/labels_te.csv' , header=None, names=['label'])
    feature_name = pd.read_csv(r'BRCA/1_featname.csv' , header=None)
    
    # Create dataloader
    pair_dataset_te = PairDataset(omic_1 , omic_2, omic_3)
    dataloader_te = torch.utils.data.DataLoader(pair_dataset_te, batch_size=20, shuffle=False , collate_fn=collate, drop_last=True)
    
    # load model using checkpoint
    model = MultiGraphClassification.load_from_checkpoint(r'lightning_logs/version_365/checkpoints/epoch=99-step=3000.ckpt' , in_channels=1, hidden_channels=32 , num_classes=5)
    model.mlp[2].register_forward_hook(hook) # register hook to get the activation (Embedding of 3 omics layers)
    
    # print model summary 
    print(pl.utilities.model_summary.ModelSummary(model))
    print(model)
    
    # pl trainer 
    trainer = pl.Trainer()
    trainer.test(model=model, dataloaders=dataloader_te)
    
    # # explaination 
    # explainer_algo = GNNExplainer(epochs=100)
    # model_config = ModelConfig('
    #     mode='multiclass_classification' , 
    #     task_level='graph', 
    #     return_type='log_probs'
    # )
    # explain = Explainer(
    #     model=model, 
    #     algorithm=explainer_algo , 
    #     explanation_type='model',
    #     model_config=model_config, 
    #     node_mask_type='object'
    # )
    # for batch in dataloader_te:
    #     batch1 , batch2 , batch3 = batch 
        
    #     x1 , edge_index1 , edge_attr1 , batch1_idx ,  y1 = batch1.x , batch1.edge_index , batch1.edge_attr , batch1.batch ,  batch1.y
    #     x2 , edge_index2 , edge_attr2 , batch2_idx ,  y2 = batch2.x , batch2.edge_index , batch2.edge_attr , batch2.batch ,  batch2.y
    #     x3 , edge_index3 , edge_attr3 , batch3_idx ,  y3 = batch3.x , batch3.edge_index , batch3.edge_attr , batch3.batch ,  batch3.y
    #     #explain_feature = explain( x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx  , batch2_idx  , batch3_idx )
    #     explain(x1 , edge_index1 , target=)
    #     break;
    # Generate ranked genes 
    # perm1 , perm2 , score1, score2 , batch1 , batch2 = model.rank['omic1'] # last batch of epoch training
    # print("---- perm1 ----")
    # print(perm1.cpu())
    # print(perm1.cpu().shape)
    
    # print("---- score1 ----")
    # print(score1)
    # print(score1.shape)
    
    # print("---- batch1 ----")
    # print(batch1)
    # print(batch1.shape)
    
    genes1 = torch.zeros(batch_size*len(model.allrank[omic_type]) , gene_count)
    genes2 = torch.zeros(batch_size*len(model.allrank[omic_type]) , gene_count)
    batch_end_idx = batch_size*len(model.allrank[omic_type])
    for idx ,  ( perm1 , perm2 , score1 , score2 , batch1 , batch2 , attr_score1, attr_score2 ) in enumerate(model.allrank[omic_type]):
        
        # dense_batch , mask = geom_utils.to_dense_batch(perm1 , batch1 , batch_size=20 , max_num_nodes=500)
        
        # get batch of data from dataloader 
        
        sns.heatmap(attr_score1.cpu().numpy())
        plt.savefig('connection1.png')
        plt.clf()
        
        dense_graph = geom_data.Batch.from_data_list(omic_1[idx*20:(idx+1)*20]) # convert list of graph to batch
        rank1 = dense_graph.extra_label[perm1.cpu()].view(20 , 500) # batch , selected_genes_index
        
        dense_edge = geom_utils.to_dense_adj(dense_graph.edge_index , batch=dense_graph.batch , edge_attr=dense_graph.edge_attr)
        print(dense_edge.shape)
        
        score1 = score1.cpu().view(20 , math.floor(gene_count*0.5))

        #genes = torch.zeros(20 , 1000)
    
        # given ranked contain index of selected genes
        # we need to put score to original gene 

        for i in range(batch_size):
            genes1[(idx*batch_size)+i , rank1[i]] = score1[i] + attr_score1[i].sum(dim=-1).cpu()
        #print(attr_score1.shape)
        #print(attr_score2.shape)
    # discovery_genes , genes_score = genes.mean(dim=0).topk(20)
    # print(discovery_genes)
    # print(genes_score)

        sns.heatmap(attr_score2.cpu().numpy())
        plt.savefig('connection2.png')
        plt.clf()
        
        rank2 = dense_graph.extra_label[perm1.cpu()][perm2.cpu()].view(20 , math.floor(gene_count*0.5*0.5))
        score2 = score2.cpu().view(20 , math.floor(gene_count*0.5*0.5))

    
        for i in range(batch_size):
            genes2[(idx*batch_size)+i , rank2[i]] = score2[i] + attr_score2[i].sum(dim=-1).cpu()
        
    
    genes2 = genes2 + genes1 
    
    # Print heatmap of the ranked genes
    sns.heatmap(genes2.cpu().numpy())
    plt.savefig('heatmap.png')
    plt.clf()
    
    unique_label = label['label'].unique().tolist()
    gene_by_label = []
    sorted_genes = []
    for _label in unique_label:
        filtered_label = label.iloc[:batch_end_idx,:][label['label'] == _label].index.to_list() 
        gene_by_label.append(genes2[filtered_label, :])
        sorted_genes.append(genes2[filtered_label, : ].topk(k=250 , dim=1).indices) # sample , top_250 gene idx
    #print(sorted_genes)
    gene_by_label = torch.concat(gene_by_label , dim=0)
    sorted_genes = torch.concat(sorted_genes , dim=0)
    sns.heatmap(gene_by_label.cpu().numpy())
    plt.savefig('heatmap2.png')
    plt.clf()
    
    # loop sorted_genes
    with open("gene_list.csv", "w") as csv_gene:
        for i , x in enumerate(sorted_genes):
            csv_gene.write(",".join([ str(int(label.iloc[i , 0])) ] + [gene_name.split("|")[1] for gene_name in feature_name.iloc[x.numpy(),0].tolist()]) + "\n")
    
    
    
    genes_score , discovery_genes  = genes2.mean(dim=0).topk(20)
        
    ##  1. Get the top 20 genes from the ranked genes
    ##  2. Get the p-value of the top 20 genes using ANOVA test
    discovery_genes = discovery_genes.cpu().numpy()
    print("---- discovery_genes ----")
    print(discovery_genes)
    
    # Filter only the top 20 genes
    gene_exp = gene_exp.iloc[:,discovery_genes]
    
    # Calculate the p-value of the top 20 genes using ANOVA test
    f_statistic , p_value = f_regression(gene_exp , label)
    print(f_statistic)
    print(p_value)
    
    # visualize the first selected gene 
    # idx = discovery_genes[0]
    df = gene_exp.join(label)
    # plot subplot 5 x 4 
    fig , axis = plt.subplots(5, 4, figsize=(20, 20))
    plt.subplots_adjust(hspace=0.5)
    
    print("---- discovery_genes ----")
    print(f"discovery_genes: {[feature_name.iloc[discovery_genes[_idx] , 0] for _idx in range(20)]}")
    
    for row in range(5):
        for col in range(4):
            _idx = row * 4 + col
            ax = sns.boxplot(x='label' , y=discovery_genes[_idx] , data=df, color='gray', ax=axis[row , col])
            #ax = sns.swarmplot(x='label' , y=discovery_genes[_idx] , data=df , color='black')
            
            #axis[row , col].plot(ax)
            #feature_name.iloc[discovery_genes[_idx] , 0]
            axis[row , col].set_title(f'Gene {feature_name.iloc[discovery_genes[_idx] , 0]} (p-value={p_value[_idx]:.1E})')
            axis[row , col].set_xlabel('label')
            axis[row , col].set_ylabel('gene expression')
            
    # save the plot
    #plt.savefig('gene1.png')
    fig.savefig('gene1.png')
    
    print("---- activation ----")
    #print(activation)
    act = torch.cat(activation , dim=0).cpu().numpy()
    print(act.shape) # (number_of_samples , feature_embedding_dimension) => (260 , 32)
    
    # performed t-SNE visualization
    print(label.iloc[ :batch_end_idx , :])
    tsne = TSNE(n_components=2 , verbose=1 , random_state=123)
    z = tsne.fit_transform(act)
    df_tsne = pd.DataFrame()
    df_tsne['y'] = label.iloc[:batch_end_idx , 0]
    df_tsne['comp-1'] = z[: , 0]
    df_tsne['comp-2'] = z[: , 1]
    
    # remove previous plot
    plt.clf()
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='comp-1' , y='comp-2' , hue=df_tsne.y.to_list() , data=df_tsne, palette=sns.color_palette("hls", 5))
    plt.savefig('tsne.png')
    
    
if __name__ == '__main__':
    main()