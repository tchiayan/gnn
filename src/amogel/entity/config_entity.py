from dataclasses import dataclass 
from pathlib import Path 
from typing import List , Optional

@dataclass(frozen=True)
class Datasets:
    BRCA: str 
    KIPAN: str 

@dataclass(frozen=True)
class KeggGo: 
    BRCA: str 
    KIPAN: str
    
@dataclass(frozen=True)
class DataPreparationConfig: 
    datasets: Datasets 
    kegg_go: KeggGo
    root_dir: Path 
    unzip_dir: Path
    
@dataclass(frozen=True)
class PPIConfiguration:
    root_dir: Path
    protein_link_src: str
    protein_info_src: str 
    save_dir: Path
    
@dataclass(frozen=True)
class OmicsRawDataPaths: 
    miRNA: Path 
    mRNA: Path
    DNA: Path 
    label: Path

@dataclass(frozen=True)
class OmicFilteringConfig: 
    variance: float 
    annova: int 

@dataclass(frozen=True)
class OmicTypeWithFilteringConfig: 
    miRNA: OmicFilteringConfig
    mRNA: OmicFilteringConfig
    DNA: OmicFilteringConfig

@dataclass(frozen=True)
class DataPreprocessingConfig: 
    root_dir: Path 
    BRCA: OmicsRawDataPaths
    KIPAN: OmicsRawDataPaths
    preprocessing: OmicTypeWithFilteringConfig
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    
@dataclass(frozen=True)
class EmbeddingTrainerConfig: 
    root_dir: Path
    data_preprocessing_dir: Path
    learning_rate: float 
    output_channel: int 
    learning_epoch: int
    
@dataclass(frozen=True)
class KnowledgeGraphConfig: 
    root_dir: Path 
    embedding_dir: Path 
    ppi_dir: Path 
    data_dir: Path
    combined_score: int
    ppi: bool 
    kegg_go: bool 
    synthetic: bool
    dataset: str
    edge_threshold: float
    discretized: bool
    topk: int
    
@dataclass(frozen=True)
class ModelTrainingConfig: 
    hidden_units: int 
    learning_rate: float
    learning_epoch: int
    drop_out: float
    combined_score: int
    model: str
    dataset: str
    batch_size: int
    enable_validation: bool 
    enable_testing: bool
    enable_training: bool
    alpha: float
    binary: bool
    multihead: int
    multihead_concat: bool
    weight: Optional[List[float]] = None
    

@dataclass(frozen=True)
class EncoderTrainingConfig: 
    root_dir: Path 
    learning_rate: float
    data_preprocessing_dir: Path
    output_channel: int
    learning_epoch: int
    print_interval: int
    model: str