import pandas as pd 
from sklearn import svm 
from sklearn.model_selection import train_test_split
import tqdm

def get_topk_ac(ac_filepath , data_filepath , label_filepath , visualize=False):
    
    df = pd.read_csv(ac_filepath, sep="\t", header=None , names=['class' , 'confidence' , 'support', 'rule' , 'interestingness'])

    if visualize:
        # visualize confidence histogram per class 
        df.hist(column='confidence', by='class', bins=50, figsize=(12,12))
        
        # visualize interestingness histogram per class
        df.hist(column='interestingness', by='class', bins=50, figsize=(12,12))
    
    # Get length of class 
    classes = df['class'].unique()
    
    # calculate the number of distinct genes per class for each top k rule
    topk_summary = pd.DataFrame(columns=['class','k' , 'union_genes_set', 'genes'] , index=range(100*len(classes)))
    for idx , label in enumerate(classes): 
        df_filter = df[df['class'] == label]
        
        topK = 1 # test until topK 100
        df_filter = df_filter.sort_values(by=['interestingness'], ascending=True) 
        # filter top k
        print(f"Generate top best K for class: {label}")
        pbar = tqdm.tqdm(total=100)
        while topK <= 100:
            
            df_filter_topk = df_filter.head(topK)
            pbar.update(1)
            genes = []
            # get distinct genes
            for gene_row in df_filter_topk['rule'].str.split(","):
                genes.extend([x.split(":")[0] for x in gene_row])
            genes = list(set(genes))
            # append to summary
            topk_summary.iloc[idx*100 + topK-1] = [label , topK , len(genes) , genes]
            topK += 1
        pbar.close()
    
    if visualize:
        # visualize the number of distinct genes per class for each top k rule
        topk_summary.pivot(index='k', columns='class', values='union_genes_set').plot(figsize=(12,12) , title="TopK vs Union Genes Set Size" )
    
    df_data = pd.read_csv(data_filepath, header=None)
    df_label = pd.read_csv(label_filepath, names=['class'])
    
    
    # convert df_data to high/low expression based on median 
    df_data = df_data.apply(lambda x: x > x.median() , axis=1)
    df_data = df_data.astype(int)
    
    topk_agg = topk_summary.groupby(['k']).agg({'genes': lambda x: set().union(*x)})
    topk_agg.reset_index(inplace=True)

    topk_agg['genes_count'] = topk_agg['genes'].apply(lambda x: len(x))
    topk_agg['svm_score'] = 0 
    
    for idx , row in topk_agg.iterrows():
        gene_idx = ([int(x) for x in row['genes']])
        gene_idx.sort()
        df_data_selection = df_data.iloc[:,gene_idx]
        X_train, X_test, y_train, y_test = train_test_split(df_data_selection, df_label, test_size=0.2, random_state=42)
            
        # SVM
        clf = svm.SVC()
        clf.fit(X_train, y_train.values.ravel())
        topk_agg.loc[idx , 'svm_score'] = clf.score(X_test, y_test)

    return topk_agg


if __name__ == "__main__":
    ac_filepath = r"./david/ac_rule_3.tsv"
    data_filepath = r'./BRCA/3_tr.csv'
    label_filepath = r'./BRCA/labels_tr.csv'
    
    topk = get_topk_ac(ac_filepath , data_filepath , label_filepath , False)
    print(topk)