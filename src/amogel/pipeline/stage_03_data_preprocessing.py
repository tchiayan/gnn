from amogel.config.configuration import ConfigurationManager
from amogel.components.data_preprocessing import DataPreprocessing
from amogel import logger

STAGE_NAME = "Data Preprocessing Stage"

class DataPreprocessingPipeline:
    
    def __init__(self) -> None:    
        pass
    
    def main(self):
        config = ConfigurationManager() 
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = DataPreprocessing(data_preprocessing_config)
        
        # load BRCA dataset
        miRNA , mRNA , DNA , label = data_preprocessing.load_data("BRCA")
        miRNA , mRNA , DNA , label = data_preprocessing.data_cleaning(miRNA , mRNA , DNA , label, target="BRCA_Subtype_PAM50")
        
        data_preprocessing.save_data(miRNA , mRNA , DNA , label , "BRCA")
        data_preprocessing.generate_ac("BRCA")
        
        

if __name__ == "__main__":
    try: 
        logger.info(f"-------- Running {STAGE_NAME} --------")
        obj = DataPreprocessingPipeline()
        obj.main()
        logger.info(f"-------- Completed {STAGE_NAME} --------")
    except Exception as e:
        logger.error(f"Error in {STAGE_NAME} : {e}")
        raise e