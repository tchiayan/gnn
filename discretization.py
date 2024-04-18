from sklearn import preprocessing
import pandas as pd 
import os 
from fpgrowth import FPTree , information_gain , correlation
import itertools
import sys 
import argparse
import math
#from mlxtend.frequent_patterns import apriori
#from efficient_apriori import apriori
from fim import ista

def generate_ac_to_file(data_file , label_file , output_file , min_support=0.9 , min_confidence=0.1 ,  min_rule=False , min_rule_per_class=1000 , custom_support=None , low_memory=False):
    
    # Discretization
    df = pd.read_csv(data_file, header=None)
    est = preprocessing.KBinsDiscretizer(n_bins=2 , encode='ordinal' , strategy='uniform')
    est.fit(df)

    df = pd.DataFrame(est.transform(df))

    # Read label
    df_label = pd.read_csv(label_file, names=['class'])
    df_label['class'] = df_label['class'].astype(str)

    class_labels = df_label['class'].unique().tolist()
    class_labels.sort()
    print(f"Unique classes: {class_labels}")

    output = []
    for label_idx , label in enumerate(class_labels):
        print("_______________________________________________________________________")
        subdf = df.loc[df_label[df_label['class'] == label].index]
        print(f"Generate FPTree per class: {label} | {subdf.shape}")
        
        
        print(f"Build transaction | Data shape: {subdf.shape}")
        transactions = []
        for idx , row in subdf.iterrows():
            transaction = [f"{idx}:{i}" for idx , i in enumerate(row.values)]
            transactions.append(transaction)
            
        
        if min_rule:
            min_support = -(subdf.shape[0])
            rule_count = 0
            while rule_count < min_rule_per_class: 
                itemsets = ista(transactions , target='c' , supp=min_support , report='a')
                rule_count = len(itemsets)
                min_support += 1
            print(f"Len of frequent itemset: {len(itemsets)} | Optimised support: {-min_support} ({-min_support/subdf.shape[0]*100}%)")
                
        elif custom_support is not None: 
            min_support = -int(custom_support.split(",")[label_idx])
            print(f"Generate Frequent Itemsets (Support: {min_support})")
            itemsets = ista(transactions , target='c' , supp=min_support , report='a')
            print(f"Len of frequent itemset: {len(itemsets)}")
        else: 
            min_support = min_support
            
            
            print(f"Generate Frequent Itemsets (Support: {min_support})")
            itemsets = ista(transactions , target='c' , supp=min_support , report='a')
            print(f"Len of frequent itemset: {len(itemsets)}")
        
        generated_cars = 0
        for itemset in itemsets:
            antecedence , upper_support = itemset 
            lower_support = subdf.shape[0]
            confidence = float(upper_support) / lower_support 
            support = upper_support/len(df)
            
            if confidence >= min_confidence:
                generated_cars += 1
                output.append([str(label) , str(confidence), str(support) , ",".join(list(antecedence))])
                
        print(f"Len of generated CARs: {generated_cars}")
        

    print("Calculate information gain and correlation")
    merged_df = df.join(df_label)
    merged_df = merged_df.astype(str)

    class_column = merged_df.columns.to_list().index('class')
    merged_df.columns = [ x for x in range(len(merged_df.columns)) ]

    corr = correlation(merged_df , class_column)
    info_gain = information_gain(merged_df , class_column)

    for rule in output:
        items = rule[3].split(",")
        avg_ig = sum([ info_gain[int(x.split(":")[0])] for x  in items ])/len(items)
        avg_corr = sum([ corr[int(x.split(":")[0])] for x  in items ])/len(items)
        total_coor = 0 
        
        if avg_corr < 0:
            interestingess = 1/(math.log2(avg_ig) - math.log2(-avg_corr))
        else:
            interestingess = 1/(math.log2(avg_ig) + math.log2(avg_corr))
            
        rule.append(str(interestingess))

    output = sorted(output , key = lambda x : float(x[4]) , reverse=True) # sort by interestingness

    with open(output_file , 'w') as ac_file:
        string_row = [ "\t".join(x) for x in output ]
        ac_file.write("\n".join(string_row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Discretization")
    parser.add_argument("--min_support" , default=0.9, type=float , help="Min support")
    parser.add_argument("--min_confidence" , default=0.1, type=float , help="Min confidence")
    parser.add_argument("--percentage", action="store_true")
    parser.add_argument("--custom_support" , type=str , default=None)
    parser.add_argument("--low_memory" , action='store_true')
    parser.add_argument("--min_rule", action='store_true')
    parser.add_argument("--min_rule_per_class" , type=int , default=1000)
    parser.add_argument("--input", type=str , default="BRCA/1_tr.csv")
    parser.add_argument("--label" , type=str , default="BRCA/labels_tr.csv")
    parser.add_argument("--output" , type=str , default="AC_rules.tsv")
    args = parser.parse_args()  
    print(args)
    
    generate_ac_to_file(args.input , args.label , args.output , args.min_support , args.min_confidence , args.min_rule , args.min_rule_per_class , args.custom_support , args.low_memory)