from amogel.config.configuration import ConfigurationManager
from amogel.components.ppi_preparation import PPIPreparation
from amogel import logger

STAGE_NAME = "PPI Data Preparation Stage"

class PPIPreparationPipeline:
    
    def __init__(self) -> None:    
        pass
    
    def main(self):
        config = ConfigurationManager() 
        ppi_config = config.get_ppi_config()
        ppi_data_preparation = PPIPreparation(ppi_config)
        ppi_data_preparation.download_data()
        ppi_data_preparation.unzip_data()
        

if __name__ == "__main__":
    try: 
        logger.info(f"-------- Running {STAGE_NAME} --------")
        obj = PPIPreparationPipeline()
        obj.main()
        logger.info(f"-------- Completed {STAGE_NAME} --------")
    except Exception as e:
        logger.error(f"Error in {STAGE_NAME} : {e}")
        raise e