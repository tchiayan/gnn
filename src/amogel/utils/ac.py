from sklearn import preprocessing 
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd 
import argparse
import math
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report
from tqdm import tqdm
from fim import ista
from pathlib import Path 
import numpy as np
import torch
import itertools
import os
from collections import Counter
from amogel.model.DNN import DNN , OmicDataset
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

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
    assert isinstance(class_column, str), "Class column provided is not of type str"
    assert class_column in data.columns.to_list(), f"Invalid class column [{class_column}]: {data.columns}"
    #assert class_column < len(data.columns) | class_column >= 0, "Invalid class column"

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
    assert isinstance(class_column, str), "Class column provided is not of type str"
    assert class_column in data.columns.to_list(), f"Invalid class column [{class_column}]: {data.columns}"
    #assert class_column < len(data.columns) | class_column >= 0, "Invalid class column"

    # ordinal_encoder = OrdinalEncoder()
    # data = ordinal_encoder.fit_transform(data)
    # data = pd.DataFrame(data)
    columns = data.columns.to_list()
    corr = data.corr()[class_column].values.tolist()   # obtain the correlation of attributes with the class label
    # print("Correlation:" , corr)
    return { columns[idx]:_corr for idx , _corr in enumerate(corr)}


def generate_ac_to_file(data_file, label_file , output_file , min_support=0.9 , min_confidence=0.0 , min_rule_per_class=1000 , n_bins=2 , filter=[]):
    
    # Discretization
    if isinstance(data_file , Path) or isinstance(data_file , str):
        df = pd.read_csv(data_file, header=None)
    elif isinstance(data_file , pd.DataFrame):
        df = data_file
        
    est = preprocessing.KBinsDiscretizer(n_bins=n_bins , encode='ordinal' , strategy='quantile')
    # Get discretized threshold
    df = pd.DataFrame(est.fit_transform(df) , columns=df.columns)
    # Filtering
    if len(filter) > 0:
        df = df[filter]

    # Read label
    if isinstance(label_file , Path) or isinstance(label_file , str):
        df_label = pd.read_csv(label_file, names=['class'])
    elif isinstance(label_file , pd.DataFrame):
        df_label = label_file
    df_label.columns = ['class']
    df_label['class'] = df_label['class'].astype(str)

    class_labels = df_label['class'].unique().tolist()
    class_labels.sort()

    output = []
    rule_summary = []
    
    # build transaction 
    transactions = {}
    for label_idx , label in enumerate(class_labels):
        subdf = df.loc[df_label[df_label['class'] == label].index]
        column_idx = subdf.columns.to_list()
        transactions[label] = []
        for idx , row in subdf.iterrows():
            transaction = set([f"{column_idx[idx]}:{i}" for idx , i in enumerate(row.values)])
            transactions[label].append(transaction)
            
    with tqdm(total=len(class_labels)) as pbar:
        
        for label_idx , label in enumerate(class_labels):
            subdf = df.loc[df_label[df_label['class'] == label].index]
            arm_summary = {}
            
            arm_summary['data_shape'] = subdf.shape
            min_support = -(subdf.shape[0])
            rule_count = 0
            pbar.set_description("Generate frequent itemset for class {}".format(label))
            while rule_count < min_rule_per_class: 
                itemsets = ista(transactions[label] , target='c' , supp=min_support , report='a')
                rule_count = len(itemsets)
                min_support += 1
            #print(f"Len of frequent itemset: {len(itemsets)} | Optimised support: {-min_support} ({-min_support/subdf.shape[0]*100}%)")
            arm_summary['itemset_length'] = len(itemsets)
            arm_summary['support'] = -min_support
            arm_summary['support_percentage'] = -min_support/subdf.shape[0]*100
            
            generated_cars = 0
            pbar.set_description("Generate CARs for class {}".format(label))
            avg_confidence = 0
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
                avg_confidence += confidence
                
                if confidence >= min_confidence:
                    generated_cars += 1
                    if len(antecedence) > 1 :
                        output.append([str(label) , str(confidence), str(support) , ",".join(list(antecedence))])
                    
            #print(f"Len of generated CARs: {generated_cars}")
            arm_summary['cars_length'] = generated_cars
            arm_summary['avg_confidence'] = avg_confidence / len(itemsets)
            rule_summary.append(arm_summary)
            pbar.update(1)
    print(f"------ ARM summary [Gene Filter: {len(filter)}]---------")
    print(pd.DataFrame(rule_summary))
        

    merged_df = df.join(df_label)
    merged_df = merged_df.astype(str)

    #class_column = merged_df.columns.to_list().index('class')
    #merged_df.columns = [ x for x in range(len(merged_df.columns)) ]
    corr = correlation(merged_df , 'class')
    info_gain = information_gain(merged_df , 'class')
    feature_selection = set([])
    for rule in output:
        items = rule[3].split(",")
        support = rule[2]
        confidence = rule[1]
        genes = [ int(x.split(":")[0]) for x in items ]
        avg_ig = sum([ info_gain[x] for x  in genes if x in info_gain.keys() ])/len(items)
        avg_corr = abs(sum([ corr[x] for x in genes if not math.isnan(corr[x]) ])/len(items)) + 0.0000001
        
        feature_selection = feature_selection.union(set(genes))
        try:
            interestingness_1 = math.log2(avg_ig) + math.log2(avg_corr) + math.log2(float(confidence)+0.0000001)
            interestingness_2 = 1/math.log2(avg_ig) + 1/math.log2(avg_corr) + 1/(float(confidence)+0.0000001)
            interestingness_3 = math.log2(avg_ig) + math.log2(avg_corr) + math.log2(float(support)*float(confidence)+0.0000001)
            if math.isnan(interestingness_1) or math.isnan(interestingness_2) or math.isnan(interestingness_3):
                raise Exception("Error")
        except: 
            print( [corr[x] for x in genes ])
            print(f"Error: {avg_ig} | {avg_corr} | {confidence}")
            raise Exception("Error")
        
        #print(f"IG: {avg_ig} | Corr: {avg_corr} | Interestingness: {interestingess}")
        rule.append(str(interestingness_1))
        rule.append(str(interestingness_2))
        rule.append(str(interestingness_3))

    # x[0] = class , x[1] = confidence , x[2] = support , x[3] = antecedence , x[4] = interestingness_1 , x[5] = interestingness_2
    output = sorted(output , key = lambda x : float(x[4]) , reverse=True) # sort by interestingness_1

    #print(f"Before feature selection: {len(feature_selection)}")
    #feature_selection = [ gene for gene in feature_selection if corr[gene] > 0.1]
    print(f"Selected gene corr: {len(feature_selection)}")
    
    with open(output_file , 'w') as ac_file:
        string_row = [ "\t".join(x) for x in output ]
        ac_file.write("\n".join(string_row))
    
    return est , list(feature_selection)


def generate_ac_feature_selection(data_file, label_file , output_file  , min_support=0.9 , min_confidence=0.0 , min_rule_per_class=1000 , n_bins=2 , filter=[] , fixed_k=None , df_test_data:pd.DataFrame = None , df_test_label:pd.DataFrame = None):
    
    # Discretization
    if isinstance(data_file , Path) or isinstance(data_file , str):
        df = pd.read_csv(data_file, header=None)
    elif isinstance(data_file , pd.DataFrame):
        df = data_file
        
    est = preprocessing.KBinsDiscretizer(n_bins=n_bins , encode='ordinal' , strategy='quantile')
    # Get discretized threshold
    df = pd.DataFrame(est.fit_transform(df) , columns=df.columns)
    # Filtering
    if len(filter) > 0:
        df = df[filter]

    # Read label
    if isinstance(label_file , Path) or isinstance(label_file , str):
        df_label = pd.read_csv(label_file, names=['class'])
    elif isinstance(label_file , pd.DataFrame):
        df_label = label_file.copy(deep=True)
    df_label.columns = ['class']
    df_label['class'] = df_label['class'].astype(str)

    class_labels = df_label['class'].unique().tolist()
    class_labels.sort()

    output = []
    rule_summary = []
    
    # build transaction 
    transactions = {}
    for label_idx , label in enumerate(class_labels):
        subdf = df.loc[df_label[df_label['class'] == label].index]
        column_idx = subdf.columns.to_list()
        transactions[label] = []
        for idx , row in subdf.iterrows():
            transaction = set([f"{column_idx[idx]}:{i}" for idx , i in enumerate(row.values)])
            transactions[label].append(transaction)
            
    with tqdm(total=len(class_labels)) as pbar:
        
        for label_idx , label in enumerate(class_labels):
            subdf = df.loc[df_label[df_label['class'] == label].index]
            arm_summary = {}
            
            arm_summary['data_shape'] = subdf.shape
            min_support = -(subdf.shape[0])
            rule_count = 0
            pbar.set_description("Generate frequent itemset for class {}".format(label))
            while rule_count < min_rule_per_class: 
                itemsets = ista(transactions[label] , target='c' , supp=min_support , report='a')
                rule_count = len(itemsets)
                min_support += 1
            #print(f"Len of frequent itemset: {len(itemsets)} | Optimised support: {-min_support} ({-min_support/subdf.shape[0]*100}%)")
            arm_summary['itemset_length'] = len(itemsets)
            arm_summary['support'] = -min_support
            arm_summary['support_percentage'] = -min_support/subdf.shape[0]*100
            
            generated_cars = 0
            pbar.set_description("Generate CARs for class {}".format(label))
            avg_confidence = 0
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
                avg_confidence += confidence
                
                if confidence >= min_confidence:
                    generated_cars += 1
                    if len(antecedence) > 1 :
                        output.append([str(label) , str(confidence), str(support) , ",".join(list(antecedence))])
                    
            #print(f"Len of generated CARs: {generated_cars}")
            arm_summary['cars_length'] = generated_cars
            arm_summary['avg_confidence'] = avg_confidence / len(itemsets)
            rule_summary.append(arm_summary)
            pbar.update(1)
    print(f"------ ARM summary [Gene Filter: {len(filter)}]---------")
    print(pd.DataFrame(rule_summary))
        

    merged_df = df.join(df_label)
    merged_df = merged_df.astype(str)

    #class_column = merged_df.columns.to_list().index('class')
    #merged_df.columns = [ x for x in range(len(merged_df.columns)) ]
    corr = correlation(merged_df , 'class')
    info_gain = information_gain(merged_df , 'class')
    #feature_selection = set([])
    for rule in output:
        items = rule[3].split(",")
        support = rule[2]
        confidence = rule[1]
        genes = [ int(x.split(":")[0]) for x in items ]
        avg_ig = sum([ info_gain[x] for x  in genes if x in info_gain.keys() ])/len(items)
        avg_corr = abs(sum([ corr[x] for x in genes if not math.isnan(corr[x]) ])/len(items)) + 0.0000001
        
        #feature_selection = feature_selection.union(set(genes))
        try:
            interestingness_1 = math.log2(avg_ig) + math.log2(avg_corr) + math.log2(float(confidence)+0.0000001)
            interestingness_2 = 1/math.log2(avg_ig) + 1/math.log2(avg_corr) + 1/(float(confidence)+0.0000001)
            interestingness_3 = math.log2(avg_ig) + math.log2(avg_corr) + math.log2(float(support)*float(confidence)+0.0000001)
            if math.isnan(interestingness_1) or math.isnan(interestingness_2) or math.isnan(interestingness_3):
                raise Exception("Error")
        except: 
            print( [corr[x] for x in genes ])
            print(f"Error: {avg_ig} | {avg_corr} | {confidence}")
            raise Exception("Error")
        
        #print(f"IG: {avg_ig} | Corr: {avg_corr} | Interestingness: {interestingess}")
        rule.append(str(interestingness_1))
        rule.append(str(interestingness_2))
        rule.append(str(interestingness_3))

    # x[0] = class , x[1] = confidence , x[2] = support , x[3] = antecedence , x[4] = interestingness_1 , x[5] = interestingness_2
    # output = sorted(output , key = lambda x : float(x[4]) , reverse=True) # sort by interestingness_1

    #print(f"Before feature selection: {len(feature_selection)}")
    #feature_selection = [ gene for gene in feature_selection if corr[gene] > 0.1]
    df_ac = pd.DataFrame(output , columns=['class' , 'confidence' , 'support' , 'rules' , 'interestingness_1' , 'interestingness_2' , 'interestingness_3'])
    df_ac['interestingness_1'] = df_ac['interestingness_1'].astype(float)
    df_ac['interestingness_2'] = df_ac['interestingness_2'].astype(float)
    df_ac['interestingness_3'] = df_ac['interestingness_3'].astype(float)
    
    
    if fixed_k == "CBA":
        # build CBA classifier 
        df_ac = df_ac.sort_values(by='interestingness_1' , ascending=False) # sort by interestingness_1 from highest to lowest
        classifier = []
        default_classes = [] 
        rules_error = []
        total_errors = []
        # convert dataframe to list of set
        datasets = [ set([f"{x[0]}:{x[1]}" for x in list(zip(row.index , row.values))]) for idx , row in df.copy(deep=True).iterrows()]
        labels = df_label['class'].values.tolist()
        print(len(datasets) , len(labels))
        with tqdm(total=len(df_ac)) as pbar:
            for idx , row in df_ac.iterrows():
                rule_antecedents = set([ x for x in row['rules'].split(",")])
                rule_consequent = row['class']
                rule_marked = False
                
                if len(datasets) <= 0: 
                    break
                
                temp_label = []
                temp_len = 0 
                temp_satistifes_conseq_cnt = 0 
                
                for didx , datacase in enumerate(datasets):
                    if rule_antecedents <= datacase: # if rule is subset of datacase
                        temp_label.append(didx)
                        temp_len += 1
                        
                        if rule_consequent == labels[didx]:
                            temp_satistifes_conseq_cnt += 1
                            rule_marked = True
                    
                if rule_marked: 
                    classifier.append(rule_antecedents)
                    datasets = [ x for idx , x in enumerate(datasets) if idx not in temp_label]
                    labels = [ x for idx , x in enumerate(labels) if idx not in temp_label]
                    
                    # get the default class by majority remaining class 
                    try:
                        most_label_item = Counter(labels).most_common(1)[0]
                        most_common_label = most_label_item[0]
                        most_common_label_count = most_label_item[1]
                    except: 
                        most_common_label = "None"
                        most_common_label_count = 0
                        print("End of most_common_label")
                    default_classes.append(most_common_label)
                    
                    rules_error.append(temp_len - temp_satistifes_conseq_cnt)
                    dflt_class_err = len(datasets) - most_common_label_count 
                    total_errors.append(dflt_class_err + sum(rules_error))
                    
                pbar.update(1)
        
        idx_to_cut = total_errors.index(min(total_errors))
        final_classifier = classifier[:idx_to_cut+1]
        default_class = default_classes[idx_to_cut]
        
        feature_selection = set()
        for rule in final_classifier:
            feature_selection = feature_selection.union(set([ int(x.split(":")[0]) for x in list(rule) ]))
        print(f"Number of rules: {len(final_classifier)}")
            
        # build network graph for selected k 
        corr_array = torch.tensor([abs(corr[x]) if x in corr.keys() else 0 for x in range(0 , df.shape[1])] , dtype=torch.float32)
        infogain_array = torch.tensor([info_gain[x] if x in info_gain.keys() else 0 for x in range(0 , df.shape[1])] , dtype=torch.float32)
        edge_tensor = torch.zeros(df.shape[1] , df.shape[1])    
        #df_filter = df_ac.groupby('class').apply(lambda x: x.nlargest(selected_k , 'interestingness_1')).reset_index(drop=True)
        for idx , row in enumerate(final_classifier):
            gene_idx = [int(x.split(":")[0]) for x in list(row)]
            combination = np.array([list(x) for x in itertools.combinations(gene_idx , 2)])
            edge_tensor[combination[:,0] , combination[:,1]] = (infogain_array[combination[:,0]] + infogain_array[combination[:,1]] + corr_array[combination[:,0]] + corr_array[combination[:,1]])/4
            edge_tensor[combination[:,1] , combination[:,0]] = (infogain_array[combination[:,0]] + infogain_array[combination[:,1]] + corr_array[combination[:,0]] + corr_array[combination[:,1]])/4
            
        return est , list(feature_selection) , edge_tensor
    elif fixed_k == "DNN":
        
        test_k = [  5 , 10 , 20 , 30 , 40 , 50 , 100 , 150 , 200 , 500 , 1000 , 1500 , 2000  ]
        # test_k = [ 5 , 10 ]
        selected_k = 5
        best_acc = 0 
        
        for k in test_k:
            feature_selection = set()
            df_filter = df_ac.groupby('class').apply(lambda x: x.nlargest(k , 'interestingness_1')).reset_index(drop=True)
            for idx , row in df_filter.iterrows():
                items = row['rules'].split(",")
                genes = [ int(x.split(":")[0]) for x in items ]
                feature_selection = feature_selection.union(set(genes))
                
            # load data and train the model using DNN
            feature_selection = list(feature_selection)
            feature_selection.sort()
            
            train_data = data_file.loc[: , list(feature_selection)]
            train_label = label_file
            test_data = df_test_data.loc[: , list(feature_selection)]
            test_label = df_test_label
            
            train_dataset = OmicDataset(train_data , train_label)
            test_dataset = OmicDataset(test_data , test_label)
            
            # dataloader 
            train_loader = DataLoader(train_dataset , batch_size=32 , shuffle=True)
            test_loader = DataLoader(test_dataset , batch_size=32 , shuffle=False)
            
            num_classes = len(df_label['class'].unique())
            model = DNN(input_dimension=train_data.shape[1] , num_classes=num_classes)
            
            trainer = Trainer(max_epochs=100 , enable_progress_bar=False , enable_checkpointing=False)
            trainer.fit(model , train_loader )
            print(f"Testing with topk: {k}")
            output = trainer.test(model , test_loader)
            
            if output[0]['test_acc'] > best_acc:
                best_acc = output[0]['test_acc']
                selected_gene = feature_selection
                selected_k = k
            
        # build network graph for selected k 
        corr_array = torch.tensor([abs(corr[x]) if x in corr.keys() else 0 for x in range(0 , df.shape[1])] , dtype=torch.float32)
        infogain_array = torch.tensor([info_gain[x] if x in info_gain.keys() else 0 for x in range(0 , df.shape[1])] , dtype=torch.float32)
        edge_tensor = torch.zeros(df.shape[1] , df.shape[1])    
        df_filter = df_ac.groupby('class').apply(lambda x: x.nlargest(selected_k , 'interestingness_1')).reset_index(drop=True)
        for idx , row in df_filter.iterrows():
            gene_idx = [int(x.split(":")[0]) for x in row['rules'].split(",")]
            combination = np.array([list(x) for x in itertools.combinations(gene_idx , 2)])
            edge_tensor[combination[:,0] , combination[:,1]] = (infogain_array[combination[:,0]] + infogain_array[combination[:,1]] + corr_array[combination[:,0]] + corr_array[combination[:,1]])/4
            edge_tensor[combination[:,1] , combination[:,0]] = (infogain_array[combination[:,0]] + infogain_array[combination[:,1]] + corr_array[combination[:,0]] + corr_array[combination[:,1]])/4
            
        print("Best K: {} | Best Acc: {:.4f} | Total selected gene: {}".format(selected_k , best_acc, len(selected_gene)))
        return est , list(selected_gene) , edge_tensor
    
    elif fixed_k is None:
        test_k = [  5 , 10 , 20 , 30 , 40 , 50 , 100 , 150 , 200 , 500 , 1000 , 1500 , 2000  ]
    else: 
        test_k  = [ fixed_k ]
    
    selected_k = 1 
    best_acc = 0
    selected_gene = []
    for k in test_k:
        feature_selection = set()
        df_filter = df_ac.groupby('class').apply(lambda x: x.nlargest(k , 'interestingness_1')).reset_index(drop=True)
        for idx , row in df_filter.iterrows():
            items = row['rules'].split(",")
            genes = [ int(x.split(":")[0]) for x in items ]
            feature_selection = feature_selection.union(set(genes))
            
        # generate testing model 
        model_summary = {}
        for label in df_label['class'].unique():
            df_sub = df_filter[df_filter['class'] == label]
            model_summary[label] = []
            for idx , row in df_sub.iterrows(): 
                rule = [ x for x in row['rules'].split(",")]
                model_summary[label].append(rule)
        
        # prediction 
        class_summary = []
        for idx , row in df.iterrows():
            sample = set([f"{x[0]}:{x[1]}" for x in list(zip(row.index , row.values))])
            summary = {}
            for label in model_summary.keys():
                summary[label] = []
                
                for rule in model_summary[label]:
                    intersection_set = sample.intersection(set(rule))
                    if len(rule) != 0:
                        summary[label].append(len(intersection_set)/len(rule))
                    else:
                        summary[label].append(0)
                summary[label] = sum(summary[label])/len(summary[label])
            
            # select the class with highest score
            summary['predict'] = max(summary , key=summary.get) 
            summary['sample'] = idx 
            class_summary.append(summary)
        
        df_prediction = pd.DataFrame(class_summary)
        report = classification_report(df_label['class'] , df_prediction['predict'] , output_dict=True , digits=4)
        print(f"Top {k} | {report['accuracy']:.4f} | Lens gene: {len(feature_selection)}")
        
        if report['accuracy'] > best_acc:
            best_acc = report['accuracy']
            selected_gene = feature_selection
            selected_k = k
    
    # build network graph for selected k 
    corr_array = torch.tensor([abs(corr[x]) if x in corr.keys() else 0 for x in range(0 , df.shape[1])] , dtype=torch.float32)
    infogain_array = torch.tensor([info_gain[x] if x in info_gain.keys() else 0 for x in range(0 , df.shape[1])] , dtype=torch.float32)
    edge_tensor = torch.zeros(df.shape[1] , df.shape[1])    
    df_filter = df_ac.groupby('class').apply(lambda x: x.nlargest(selected_k , 'interestingness_1')).reset_index(drop=True)
    for idx , row in df_filter.iterrows():
        gene_idx = [int(x.split(":")[0]) for x in row['rules'].split(",")]
        combination = np.array([list(x) for x in itertools.combinations(gene_idx , 2)])
        edge_tensor[combination[:,0] , combination[:,1]] = (infogain_array[combination[:,0]] + infogain_array[combination[:,1]] + corr_array[combination[:,0]] + corr_array[combination[:,1]])/4
        edge_tensor[combination[:,1] , combination[:,0]] = (infogain_array[combination[:,0]] + infogain_array[combination[:,1]] + corr_array[combination[:,0]] + corr_array[combination[:,1]])/4
        
    print("Best K: {} | Best Acc: {:.4f}".format(selected_k , best_acc))
    return est , list(selected_gene) , edge_tensor

if __name__ == "__main__":
    print(os.listdir("../../../"))
    generate_ac_feature_selection(
        "./artifacts/data_preprocessing/KIPAN/1_tr.csv", 
        "./artifacts/data_preprocessing/KIPAN/labels_tr.csv",
        "", 
        fixed_k=50
    )