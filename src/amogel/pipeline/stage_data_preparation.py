from amogel.config.configuration import ConfigurationManager 
from amogel.components.data_preparation import DataPreparation
from amogel import logger    

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionPipeline:
    
    def __init__(self) -> None:    
        pass
    
    def main(self):
        config = ConfigurationManager() 
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataPreparation(data_ingestion_config)
        data_ingestion.download_data()
        data_ingestion.extract_zip_file()
        

if __name__ == "__main__":
    try: 
        logger.info(f"-------- Running {STAGE_NAME} --------")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f"-------- Completed {STAGE_NAME} --------")
    except Exception as e:
        logger.error(f"Error in {STAGE_NAME} : {e}")
        raise e
    
    