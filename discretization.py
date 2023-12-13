from sklearn import preprocessing
import pandas as pd 
import os 
from fpgrowth import FPTree , information_gain , correlation
import itertools
import sys 
import argparse
import math
from mlxtend.frequent_patterns import apriori
from efficient_apriori import apriori

parser = argparse.ArgumentParser("Discretization")
parser.add_argument("--min_support" , default=0.9, type=float , help="Min support")
parser.add_argument("--min_confidence" , default=0.1, type=float , help="Min confidence")
parser.add_argument("--percentage", action="store_true")
parser.add_argument("--custom_support" , type=str , default=None)
parser.add_argument("--low_memory" , action='store_true')

args = parser.parse_args()

sys.setrecursionlimit(5000)
base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "BRCA")


df = pd.read_csv(os.path.join(base_path , "2_tr.csv"), header=None)
est = preprocessing.KBinsDiscretizer(n_bins=2 , encode='ordinal' , strategy='uniform')
est.fit(df)

df = pd.DataFrame(est.transform(df))
df.replace({1: True , 0: False} , inplace=True)

df_label = pd.read_csv(os.path.join(base_path , "labels_tr.csv"), names=['class'])
# df = df.join(df_label)

# for column in df.columns: 
#     df[column] = df[column].astype(str)
#     df[column] = "Feature " + str(column) + " : " + df[column]
# print(df.head())
# print(df.describe())
df_label['class'] = df_label['class'].astype(str)
class_labels = df_label['class'].unique().tolist()
print(f"Unique classes: {class_labels}")


custom_support = args.custom_support 

if custom_support is not None: 
    custom_support = { x.split(":")[0] : x.split(":")[1] for x in custom_support.split(";") }
    print(custom_support)


CARs = {}
output = []
for label in class_labels:
    print("_______________________________________________________________________")
    subdf = df.loc[df_label[df_label['class'] == label].index]
    print(f"Generate FPTree per class: {label} | {subdf.shape}")
    
    
    print(f"Build transaction | Data shape: {subdf.shape}")
    transactions = []
    for idx , row in subdf.iterrows():
        transaction = (f"{idx}" for idx , i in enumerate(row.values) if i )
        transactions.append(transaction)

    itemsets , rules = apriori(transactions , args.min_support , args.min_confidence)
    print(itemsets)
    #frequent_itemsets = apriori(subdf , min_support=args.min_support , low_memory=args.low_memory)
    #print(frequent_itemsets)
    # print("Generate FP Tree")
    # if custom_support is not None:
    #     min_support = int(custom_support[label])
    # else:
    #     min_support = args.min_support if not args.percentage else int(args.min_support * subdf.shape[0])
    # tree = FPTree(transactions , min_support , None , None ,)
    # print(f"Mine patterns [Support: {min_support}]")
    # patterns = tree.mine_patterns(min_support) #return dict with key: item-set , value: support score
    # print(f"Generated frequent item set [Class: {label}]: {len(patterns)}")
    
    # print("Generate CARs")
    # generated_cars = 0
    # for itemset in patterns.keys():
    #     upper_support = patterns[itemset]
        
    #     lower_support = subdf.shape[0]
    #     #print(f"{lower_support} | {upper_support}")
    #     confidence = float(upper_support)/lower_support
    #     support = upper_support/len(df)
        
    #     if confidence >= args.min_confidence:
    #         generated_cars += 1
    #         CARs[itemset] = ( label , confidence , support)
    #         output.append([str(label) , str(confidence) , str(support) , ",".join(list(itemset))])
    
    
    # print(f"Generated CARs [Class: {label}]: {generated_cars}")
    # print("--------------------------------------------------------------------")

# print("Calculate information gain and correlation")
# merged_df = df.join(df_label)
# merged_df = merged_df.astype(str)

# class_column = merged_df.columns.to_list().index('class')
# merged_df.columns = [ x for x in range(len(merged_df.columns)) ]

# corr = correlation(merged_df , class_column)
# info_gain = information_gain(merged_df , class_column)

# for rule in output:
#     items = rule[3].split(",")
#     avg_ig = sum([ info_gain[int(x)] for x  in items ])/len(items)
#     avg_corr = sum([ corr[int(x)] for x  in items ])/len(items)
#     total_coor = 0 
    
#     if avg_corr < 0:
#         interestingess = 1/(math.log2(avg_ig) - math.log2(-avg_corr))
#     else:
#         interestingess = 1/(math.log2(avg_ig) + math.log2(avg_corr))
        
#     rule.append(str(interestingess))

# output = sorted(output , key = lambda x : float(x[4]) , reverse=True) # sort by interestingness

# with open("AC_rules.tsv" , 'w') as ac_file:
#     string_row = [ "\t".join(x) for x in output ]
#     ac_file.write("\n".join(string_row))

