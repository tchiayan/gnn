""" Example for MOGONET classification
"""
from train_test import train_test
from amogel.config.configuration import ConfigurationManager

if __name__ == "__main__":    
    dataset = ConfigurationManager().get_dataset()
    data_folder = f'./artifacts/data_preprocessing/{dataset}'
    view_list = [1,2,3]
    num_epoch_pretrain = 500
    num_epoch = 3000
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3
    
    if data_folder == './artifacts/data_preprocessing/KIPAN':
        num_class = 3
    if data_folder == './artifacts/data_preprocessing/BRCA':
        num_class = 5
    
    train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch)             