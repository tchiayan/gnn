from amogel import logger
from amogel.config.configuration import ConfigurationManager
from amogel.components.model_training import ModelTraining

STAGE_NAME = "Model Training"

class ModelTrainingPipeline():
        
    def __init__(self) -> None:
        pass
    
    def run(self):
        config = ConfigurationManager()
        training_config = config.get_model_training_config()
        
        model_training = ModelTraining(training_config)
        model_training.training()
        if training_config.enable_testing:
            model_training.testing()
        
if __name__ == "__main__":
    
    logger.info(f"-------- Running {STAGE_NAME} --------")
    
    try:
        main = ModelTrainingPipeline()
        main.run()
    except Exception as e:
        logger.error(f"Error in {STAGE_NAME} : {e}")
        raise e
    logger.info(f"-------- Completed {STAGE_NAME} --------")
    