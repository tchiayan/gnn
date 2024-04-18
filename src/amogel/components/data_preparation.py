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