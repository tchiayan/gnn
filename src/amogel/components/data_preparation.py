import os 
import zipfile 
import gdown    
from amogel import logger    
from amogel.entity.config_entity import DataPreparationConfig

class DataPreparation: 
    
    def __init__(self , config: DataPreparationConfig):
        self.config = config 
    
    def download_data(self): 
        '''
        Fetch data from google drive 
        ''' 
        
        try:
            
            prefix = "https://drive.google.com/uc?/export=download&id="
            
            # create directory if not exist 
            os.makedirs(self.config.root_dir , exist_ok=True)
             
            # download BRCA datasets 
            BRCA_ID = self.config.datasets.BRCA
            
            
            brca_download_path = os.path.join(self.config.root_dir , "BRCA.zip")
            logger.info(f"Downloading BRCA data from {prefix + BRCA_ID} to {brca_download_path}")
            gdown.download(prefix + BRCA_ID , brca_download_path)
            
            # download KIPAN datasets
            KIPAN_ID = self.config.datasets.KIPAN
            kipan_download_path = os.path.join(self.config.root_dir , "KIPAN.zip")
            logger.info(f"Downloading KIPAN data from {prefix + KIPAN_ID} to {kipan_download_path}")
            gdown.download(prefix + KIPAN_ID , kipan_download_path)
            
            # download BRCA KEGG Pathway and GO annotation data
            brca_kegg_go_download_path = os.path.join(self.config.root_dir , "BRCA_kegg_go.zip")
            logger.info(f"Downloading BRCA KEGG_GO data from {self.config.kegg_go.BRCA} to {brca_kegg_go_download_path}")
            gdown.download(prefix + self.config.kegg_go.BRCA , brca_kegg_go_download_path)
            
            # download KIPAN KEGG Pathway and GO annotation data
            kipan_kegg_go_download_path = os.path.join(self.config.root_dir , "KIPAN_kegg_go.zip")
            logger.info(f"Downloading KIPAN KEGG_GO data from {self.config.kegg_go.KIPAN} to {kipan_kegg_go_download_path}")
            gdown.download(prefix + self.config.kegg_go.KIPAN , kipan_kegg_go_download_path)
            
            
        except Exception as e:
            raise e 

    def extract_zip_file(self):
        """
        Extracts the zip files
        """
        
        # make directory if not exist
        os.makedirs(self.config.unzip_dir , exist_ok=True)
        
        # extract BRCA data 
        brca_zip_path = os.path.join(self.config.root_dir , "BRCA.zip")
        brca_unzip_path  = os.path.join(self.config.unzip_dir , "BRCA")
        logger.info(f"Extracting BRCA data from {brca_zip_path} to {brca_unzip_path}")
        with zipfile.ZipFile(brca_zip_path , 'r') as zip_ref:
            zip_ref.extractall(brca_unzip_path)
        
        # extract KIPAN data
        kipan_zip_path = os.path.join(self.config.root_dir , "KIPAN.zip")
        kipan_unzip_path = os.path.join(self.config.unzip_dir , "KIPAN")
        
        logger.info(f"Extracting KIPAN data from {kipan_zip_path} to {kipan_unzip_path}")
        with zipfile.ZipFile(kipan_zip_path , 'r') as zip_ref:
            zip_ref.extractall(kipan_unzip_path)
            
        # extract BRCA KEGG_GO data
        brca_kegg_go_zip_path = os.path.join(self.config.root_dir , "BRCA_kegg_go.zip")
        brca_kegg_go_unzip_path = os.path.join(self.config.unzip_dir , "BRCA_kegg_go")
        logger.info(f"Extracting BRCA KEGG_GO data from {brca_kegg_go_zip_path} to {brca_kegg_go_unzip_path}")
        with zipfile.ZipFile(brca_kegg_go_zip_path , 'r') as zip_ref:
            zip_ref.extractall(brca_kegg_go_unzip_path)
            
        # extract KIPAN KEGG_GO data
        kipan_kegg_go_zip_path = os.path.join(self.config.root_dir , "KIPAN_kegg_go.zip")
        kipan_kegg_go_unzip_path = os.path.join(self.config.unzip_dir , "KIPAN_kegg_go")
        logger.info(f"Extracting KIPAN KEGG_GO data from {kipan_kegg_go_zip_path} to {kipan_kegg_go_unzip_path}")
        with zipfile.ZipFile(kipan_kegg_go_zip_path , 'r') as zip_ref:
            zip_ref.extractall(kipan_kegg_go_unzip_path)