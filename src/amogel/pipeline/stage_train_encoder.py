from amogel import logger
from amogel.components.train_multi_embedding import MultiEmbeddingTrainer
from amogel.config.configuration import ConfigurationManager
import os 
from pathlib import Path

STAGE_NAME = "Encoder Training"
class TrainEncoderPipeline():
    
    def __init__(self) -> None:    
        pass
        
    def run(self):
        config = ConfigurationManager()
        encoder_trainer_config = config.get_encoder_training_config()
        
        for i in [ 3 , 2 ,1]:
            # train omic embedding 2 
            embeddingTrainer = MultiEmbeddingTrainer(
                out_channels=encoder_trainer_config.output_channel , 
                epochs=encoder_trainer_config.learning_epoch , 
                lr=encoder_trainer_config.learning_rate , 
                omic_type=i , 
                dataset="BRCA" , 
                config=encoder_trainer_config
            )
            embeddingTrainer.run()
            embeddingTrainer.save_model()


if __name__ == "__main__": 
    try: 
        logger.info(f"-------- Running {STAGE_NAME} --------")
        obj = TrainEncoderPipeline()
        obj.run()
        logger.info(f"-------- Completed {STAGE_NAME} --------")
    except Exception as e:
        logger.error(f"Error in {STAGE_NAME} : {e}")
        raise e