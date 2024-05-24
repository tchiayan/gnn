from amogel import logger 
from amogel.entity.config_entity import PPIConfiguration 
import requests
import os
from zipfile import ZipFile
import gzip
import pandas as pd

class PPIPreparation:
    
    def __init__(self, config: PPIConfiguration):
        self.config = config
        
    
    def download_data(self):
        """
        Download data from URL
        """
        
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # Download protein link file (homosapien)
        url = self.config.protein_link_src  # Replace with your desired URL
        logger.info(f"Downloading data from {url}")
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.content
            # Save the data to a file
            save_path = os.path.join(self.config.save_dir , "protein_links.txt.gzip")
            logger.info(f"Saving data to {save_path}")
            with open(save_path, "wb") as file:
                file.write(data)
        else:
            logger.error("Failed to download data from URL")
            
        # Download protein info file (homosapien)
        url = self.config.protein_info_src
        logger.info(f"Downloading data from {url}")
        response = requests.get(url)
    
        if response.status_code == 200:
            data = response.content
            # Save the data to a file
            save_path = os.path.join(self.config.save_dir , "protein_info.txt.gzip")
            logger.info(f"Saving data to {save_path}")
            with open(save_path, "wb") as file:
                file.write(data)
        else:
            logger.error("Failed to download data from URL") 
            
    def unzip_data(self):
        """
        Unzip the data file (gzip format)
        """
        
        # create unzip folder
        unzip_folder = os.path.join(self.config.save_dir , "unzip")
        os.makedirs(unzip_folder, exist_ok=True)
        
        # unzip protein link file
        protein_link_path = os.path.join(self.config.save_dir , "protein_links.txt.gzip")
        protein_link_unzip_path = os.path.join(unzip_folder , "protein_links.parquet.gzip")
        logger.info(f"Unzipping {protein_link_path}")
        with gzip.open(protein_link_path, 'rb') as f_in:
            pd.read_csv(f_in , sep="\s" , engine="python").to_parquet(protein_link_unzip_path , compression="gzip")
            # with open(protein_link_unzip_path, 'wb') as f_out:
            #     f_out.write(f_in.read())
                
        # unzip protein info file
        protein_info_path = os.path.join(self.config.save_dir , "protein_info.txt.gzip")
        protein_info_unzip_path = os.path.join(unzip_folder , "protein_info.parquet.gzip")
        logger.info(f"Unzipping {protein_info_path}")
        with gzip.open(protein_info_path, 'rb') as f_in:
            pd.read_csv(f_in , sep="\t").to_parquet(protein_info_unzip_path , compression="gzip")
            # with open(protein_info_unzip_path, 'wb') as f_out:
            #     f_out.write(f_in.read())
        
        
        
        