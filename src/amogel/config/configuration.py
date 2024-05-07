from amogel.constants import * 
import os
from amogel import *
from amogel.utils.common import read_yaml
from amogel.entity.config_entity import (
    DataPreparationConfig , 
    PPIConfiguration , 
    DataPreprocessingConfig , 
    EmbeddingTrainerConfig , 
    KnowledgeGraphConfig , 
    ModelTrainingConfig , 
    EncoderTrainingConfig
)

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
            kegg_go=config.kegg_go,
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
        params = self.params.data_preprocessing
        
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
                }, 
            preprocessing=params
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
    
    def get_encoder_training_config(self) -> EncoderTrainingConfig: 
        config = self.config.train_encoder 
        params = self.params.train_encoder 
        
        encoder_training_config = EncoderTrainingConfig(
            root_dir=config.root_dir, 
            learning_epoch=params.learning_epoch,
            learning_rate=params.learning_rate,
            output_channel=params.output_channel,
            print_interval=params.print_interval, 
            model=params.model,
            data_preprocessing_dir=config.data_preprocessing_dir, 
        )
        
        return encoder_training_config
    
    def get_knowledge_graph_config(self) -> KnowledgeGraphConfig: 
        config = self.config.knowledge_graph
        params = self.params.knowledge_graph
        
        knowledge_graph_config = KnowledgeGraphConfig(
            root_dir=config.root_dir,
            embedding_dir=config.embedding_dir,
            ppi_dir=config.ppi_dir,
            data_dir=config.data_dir, 
            combined_score=params.ppi_combined_score
        )
        
        return knowledge_graph_config
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        
        params = self.params.model_training 
        
        model_training_config = ModelTrainingConfig(
            hidden_units=params.hidden_units,
            learning_rate=params.learning_rate,
            learning_epoch=params.learning_epoch, 
            drop_out=params.drop_out, 
            combined_score=self.params.knowledge_graph.ppi_combined_score, 
            model=params.model , 
            dataset=params.dataset , 
            batch_size=params.batch_size, 
            weight=params.weight,
            enable_validation=params.enable_validation,
            enable_testing=params.enable_testing, 
            enable_training=params.enable_training, 
            alpha=params.alpha
        )
        
        return model_training_config