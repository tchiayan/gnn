from sklearn import preprocessing 
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd 
import argparse
import math
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm
from fim import ista
from pathlib import Path 

def information_gain(data, class_column):
    """
    This function calculates the information gain (IG) of each feature with the target in the dataset, it is used in preprocessing
    of the dataset, which is rule ranking and feature selection. It also removes redundant features in the dataset which has an information
    gain lower than 0.

    Args:
        data (pd.DataFrame): The dataset which has been preprocessed to contain only categorical attributes
        class_column (int): The index of the target in the dataset

    Returns:
        dict: A dictionary where the key is the index of the column and the value is the information gain.
    """
    assert class_column != None, "Class column is not provided"
    assert isinstance(data, pd.DataFrame), "Data in wrong format, it should be in pandas dataframe"
    assert isinstance(class_column, int), "Class column provided is not of type int"
    assert class_column < len(data.columns) | class_column >= 0, "Invalid class column"

    target = data[class_column]
    features = data.drop(columns=class_column, axis=1)
    feature_columns = list(features.columns)

    # calculating information gain of the features with the target
    # print(features.values)
    
    # feature_values =  [ [ col.split(":")[1].strip() for col in row ] for row in features.values.tolist()]
    # ordinal_encoder = OrdinalEncoder()
    # features = ordinal_encoder.fit_transform(features)
    # features = pd.DataFrame(features)
    
    #print(features.values[0:2 , :])
    #print(features.values)
    information_gain = mutual_info_classif(features.values , target, discrete_features=True)

    # make a dictionary and obtain the columns of the features to be removed from the dataset
    info_gain = {}
    columns_removed = []
    for index in range(len(information_gain)):
        if information_gain[index] > 0:
            info_gain[feature_columns[index]] = information_gain[index]
        else:
            columns_removed.append(feature_columns[index])

    #print("Remove redundant features (0 IG): {}".format(columns_removed))
    #print("Information gain: {}".format(info_gain))
    # remove the redundant features
    data.drop(columns=columns_removed, axis=1, inplace=True)
    return info_gain


def correlation(data, class_column):
    """
    This function calculates how correlated the attribute is to the class column. It is used together with information gain for rule
    ranking and feature selection

    Args:
        data (pd.DataFrame): The dataset which has been preprocessed to contain only categorical attributes
        class_column (int): The index of the target in the dataset

    Returns:
        list : a list containing the correlation value of each attribute to the class column
    """
    assert class_column != None, "Class column is not provided"
    assert isinstance(data, pd.DataFrame), "Data in wrong format, it should be in pandas dataframe"
    assert isinstance(class_column, int), "Class column provided is not of type int"
    assert class_column < len(data.columns) | class_column >= 0, "Invalid class column"

    # ordinal_encoder = OrdinalEncoder()
    # data = ordinal_encoder.fit_transform(data)
    # data = pd.DataFrame(data)
    corr = data.corr()[class_column].values.tolist()   # obtain the correlation of attributes with the class label
    # print("Correlation:" , corr)
    return corr


def generate_ac_to_file(data_file:Path , label_file:Path , output_file , min_support=0.9 , min_confidence=0.0 , min_rule_per_class=1000 , n_bins=2):
    
    # Discretization
    df = pd.read_csv(data_file, header=None)
    est = preprocessing.KBinsDiscretizer(n_bins=n_bins , encode='ordinal' , strategy='quantile')
    
    # Get discretized threshold
    df = pd.DataFrame(est.fit_transform(df))

    # Read label
    df_label = pd.read_csv(label_file, names=['class'])
    df_label.columns = ['class']
    df_label['class'] = df_label['class'].astype(str)

    class_labels = df_label['class'].unique().tolist()
    class_labels.sort()
    print(f"Unique classes: {class_labels}")

    output = []
    rule_summary = []
    
    # build transaction 
    transactions = {}
    for label_idx , label in enumerate(class_labels):
        subdf = df.loc[df_label[df_label['class'] == label].index]
        transactions[label] = []
        for idx , row in subdf.iterrows():
            transaction = set([f"{idx}:{i}" for idx , i in enumerate(row.values)])
            transactions[label].append(transaction)
            
    with tqdm(total=len(class_labels)) as pbar:
        
        for label_idx , label in enumerate(class_labels):
            subdf = df.loc[df_label[df_label['class'] == label].index]
            arm_summary = {}
            
            arm_summary['data_shape'] = subdf.shape
            min_support = -(subdf.shape[0])
            rule_count = 0
            while rule_count < min_rule_per_class: 
                itemsets = ista(transactions[label] , target='c' , supp=min_support , report='a')
                rule_count = len(itemsets)
                min_support += 1
            #print(f"Len of frequent itemset: {len(itemsets)} | Optimised support: {-min_support} ({-min_support/subdf.shape[0]*100}%)")
            arm_summary['itemset_length'] = len(itemsets)
            arm_summary['support'] = -min_support
            arm_summary['support_percentage'] = -min_support/subdf.shape[0]*100
            
            generated_cars = 0
            for itemset in itemsets:
                antecedence , upper_support = itemset 
                lower_support = subdf.shape[0]
                # confidence = float(upper_support) / lower_support 
                support = upper_support / lower_support
                
                # measure confidence 
                match_transaction_within_class = 0 
                match_transaction_outside_class = 0
                for transaction_label , _transactions in transactions.items():
                    if label == transaction_label:
                        match_transaction_within_class += sum([ 1 for x in _transactions if set(antecedence).issubset(x)])
                    else:
                        match_transaction_outside_class += sum([ 1 for x in _transactions if set(antecedence).issubset(x)])
                if match_transaction_within_class == 0:
                    raise Exception("Error. Match transaction within class is 0.")
                confidence = match_transaction_within_class / (match_transaction_within_class + match_transaction_outside_class)
                    
                
                if confidence >= min_confidence:
                    generated_cars += 1
                    if len(antecedence) > 1 :
                        output.append([str(label) , str(confidence), str(support) , ",".join(list(antecedence))])
                    
            #print(f"Len of generated CARs: {generated_cars}")
            arm_summary['cars_length'] = generated_cars
            rule_summary.append(arm_summary)
            pbar.update(1)
    print("ARM summary")
    print(pd.DataFrame(rule_summary))
        

    print("Calculate information gain and correlation")
    merged_df = df.join(df_label)
    merged_df = merged_df.astype(str)

    class_column = merged_df.columns.to_list().index('class')
    merged_df.columns = [ x for x in range(len(merged_df.columns)) ]

    corr = correlation(merged_df , class_column)
    info_gain = information_gain(merged_df , class_column)
    for rule in output:
        items = rule[3].split(",")
        support = rule[2]
        confidence = rule[1]
        genes = [ int(x.split(":")[0]) for x in items ]
        avg_ig = sum([ info_gain[x] for x  in genes if x in info_gain.keys() ])/len(items)
        avg_corr = abs(sum([ corr[x] for x in genes if not math.isnan(corr[x]) ])/len(items)) + 0.0000001
        
        try:
            interestingess = 1/math.log2(avg_ig) + 1/math.log2(avg_corr) + 1/math.log2(float(confidence)-0.00000001)
            if math.isnan(interestingess):
                raise Exception("Error")
        except: 
            print( [corr[x] for x in genes ])
            print(f"Error: {avg_ig} | {avg_corr} | {confidence}")
            raise Exception("Error")
        
        #print(f"IG: {avg_ig} | Corr: {avg_corr} | Interestingness: {interestingess}")
        rule.append(str(interestingess))

    # x[0] = class , x[1] = confidence , x[2] = support , x[3] = antecedence , x[4] = interestingness
    output = sorted(output , key = lambda x : float(x[4]) , reverse=True) # sort by interestingness

    with open(output_file , 'w') as ac_file:
        string_row = [ "\t".join(x) for x in output ]
        ac_file.write("\n".join(string_row))
        
    return est


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