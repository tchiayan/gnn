from dataclasses import dataclass 
from pathlib import Path 


@dataclass(frozen=True)
class Datasets:
    BRCA: str 
    KIPAN: str 
    
    
@dataclass(frozen=True)
class DataPreparationConfig: 
    datasets: Datasets 
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
class DataPreprocessingConfig: 
    root_dir: Path 
    BRCA: OmicsRawDataPaths
    KIPAN: OmicsRawDataPaths
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    
@dataclass(frozen=True)
class EmbeddingTrainerConfig: 
    root_dir: Path
    data_preprocessing_dir: Path
    learning_rate: float 
    output_channel: int 
    learning_epoch: int