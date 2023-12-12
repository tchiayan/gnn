from sklearn import preprocessing
import pandas as pd 
import os 
from fpgrowth import FPTree
import itertools
import sys 

sys.setrecursionlimit(5000)
base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "BRCA")

data = [
    {"field1": 0.5}, 
    {"field1": 0.2}, 
    {"field1": 0.9}
]

df = pd.read_csv(os.path.join(base_path , "1_tr.csv"), header=None)
est = preprocessing.KBinsDiscretizer(n_bins=2 , encode='ordinal' , strategy='uniform')
est.fit(df)

print(df.head())
df = pd.DataFrame(est.transform(df))
# df.replace({1: True , 0: False} , inplace=True)

df_label = pd.read_csv(os.path.join(base_path , "labels_tr.csv"), names=['class'])
# df = df.join(df_label)

# for column in df.columns: 
#     df[column] = df[column].astype(str)
#     df[column] = "Feature " + str(column) + " : " + df[column]
# print(df.head())
# print(df.describe())
df_label['class'] = df_label['class'].astype(str)
class_label = df_label['class'].unique().tolist()
print(f"Unique classes: {class_label}")

print("Build transaction")
transactions = []
for idx , row in df.iloc.iterrows():
    transaction = [ f"Feature {idx}" for idx , i in enumerate(row.values) if i == 1 ]
    transactions.append(transaction)
for idx , row in df_label.iterrows():
    transactions[idx].append(row[0])
    
# frequent_itemsets = fpgrowth(df , min_support=0.5 , use_colnames=True)

print("Generate FP Tree")
tree = FPTree(transactions , 5 , None , None ,)
print("Mine patterns")
patterns = tree.mine_patterns(5) #return dict with key: item-set , value: support score


print("Generate CARs")
CARs = {}
output = []
    
for itemset in patterns.keys():
    upper_support = patterns[itemset]
    
    for i in range(1 , len(itemset)):
        # Generate possible combination of itemsets (Subset of frequent itemset is frequet itemset)
        for antecedent in itertools.combinations(itemset , i):
            antecedent = tuple(sorted(antecedent))
            consequent = tuple(sorted(set(itemset) - set(antecedent)))
            
            # obtaining the rules where only the consequent has the class label
            if len(consequent) == 1 and consequent[0] in class_label:
                if antecedent in patterns:
                    lower_support = patterns[antecedent]
                    confidence = float(upper_support) / lower_support
                    support = upper_support/len(df)

                    # filtering the rules where the confidence of the rules does not satisfy the minimum threshold
                    if confidence >= 0.3:
                        CARs[antecedent] = (consequent, confidence, support)
                        output.append("\t".join([str(consequent[0]) , str(confidence) , str(support) , ",".join(list(antecedent))]))

print(f"Generated CARs length: {len(CARs)}")               
with open("AC_rules.tsv" , 'w') as ac_file:
    ac_file.write("\n".join(output))

