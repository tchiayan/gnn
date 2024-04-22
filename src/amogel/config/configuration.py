from amogel.constants import * 
import os
from amogel import *
from amogel.utils.common import read_yaml
from amogel.entity.config_entity import DataPreparationConfig , PPIConfiguration , DataPreprocessingConfig , EmbeddingTrainerConfig , KnowledgeGraphConfig

class ConfigurationManager: 
    def __init__(
        self, 
        config_filepath = CONFIG_FILE_PATH , 
        param_filepath = PARAM_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(param_filepath)
    
    def get_data_ingestion_config(self) -> DataPreparationConfig:
        config = self.config.data_preparation 
        
        data_ingestion_config = DataPreparationConfig(
            datasets=config.datasets,
            root_dir=config.root_dir,
            unzip_dir=config.unzip_dir
        )
        
        return data_ingestion_config
    
    def get_ppi_config(self) -> PPIConfiguration:
        config = self.config.ppi_data_preparation
        
        ppi_config = PPIConfiguration(
            root_dir=config.root_dir,
            protein_link_src=config.protein_link_src, 
            protein_info_src=config.protein_info_src,
            save_dir=config.save_dir, 
        )
        
        return ppi_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig: 
        config = self.config.data_preprocessing 
        
        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=config.root_dir,
            BRCA={
                    'miRNA': config.BRCA.miRNA,
                    'mRNA': config.BRCA.mRNA,
                    'DNA': config.BRCA.DNA, 
                    'label': config.BRCA.label
                },
            KIPAN={
                    'miRNA': config.KIPAN.miRNA,
                    'mRNA': config.KIPAN.mRNA,
                    'DNA': config.KIPAN.DNA, 
                    'label': config.BRCA.label
                }
        )   
        
        return data_preprocessing_config
    
    
    def get_embedding_trainer_config(self) -> EmbeddingTrainerConfig:
        config = self.config.train_embedding
        params = self.params.embedding_training
        
        embedding_trainer_config = EmbeddingTrainerConfig(
            root_dir=config.root_dir,
            data_preprocessing_dir=config.data_preprocessing_dir, 
            learning_rate=params.learning_rate,
            output_channel=params.output_channel,
            learning_epoch=params.learning_epoch
        )
        
        return embedding_trainer_config
    
    def get_knowledge_graph_config(self) -> KnowledgeGraphConfig: 
        config  = self.config 
        
        knowledge_graph_config = KnowledgeGraphConfig(
            root_dir=config.root_dir,
            embedding_dir=config.embedding_dir,
            ppi_dir=config.ppi_dir,
            data_dir=config.data_dir
        )
        
        return knowledge_graph_config