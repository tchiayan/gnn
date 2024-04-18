from amogel import logger
from amogel.components.train_embedding import EmbeddingTrainer
from amogel.config.configuration import ConfigurationManager
import os 
from pathlib import Path

STAGE_NAME = "Embedding Training"
class TrainEmbeddingPipeline():
    
    def __init__(self) -> None:    
        pass
        
    def run(self):
        config = ConfigurationManager()
        embedding_trainer_config = config.get_embedding_trainer_config()
        
        # train omic embedding 2 
        # embeddingTrainer = EmbeddingTrainer(
        #     out_channels=embedding_trainer_config.output_channel , 
        #     epochs=embedding_trainer_config.learning_epoch , 
        #     lr=embedding_trainer_config.learning_rate , 
        #     omic_type=2 , 
        #     dataset="BRCA" , 
        #     config=embedding_trainer_config
        # )
        # embeddingTrainer.run()
        # embeddingTrainer.save_embedding()
        
        # train omic embedding 3 
        embeddingTrainer = EmbeddingTrainer(
            out_channels=embedding_trainer_config.output_channel , 
            epochs=embedding_trainer_config.learning_epoch , 
            lr=embedding_trainer_config.learning_rate , 
            omic_type=3 , 
            dataset="BRCA" , 
            config=embedding_trainer_config
        )
        embeddingTrainer.run()
        embeddingTrainer.save_embedding()
        
        # # train omic embedding 1 
        # embeddingTrainer = EmbeddingTrainer(
        #     out_channels=embedding_trainer_config.output_channel , 
        #     epochs=embedding_trainer_config.learning_epoch , 
        #     lr=embedding_trainer_config.learning_rate , 
        #     omic_type=1 , 
        #     dataset="BRCA" , 
        #     config=embedding_trainer_config
        # )
        # embeddingTrainer.run()
        # embeddingTrainer.save_embedding()
        
        
        
        
        
        
        
if __name__ == "__main__": 
    try: 
        logger.info(f"-------- Running {STAGE_NAME} --------")
        obj = TrainEmbeddingPipeline()
        obj.run()
        logger.info(f"-------- Completed {STAGE_NAME} --------")
    except Exception as e:
        logger.error(f"Error in {STAGE_NAME} : {e}")
        raise e