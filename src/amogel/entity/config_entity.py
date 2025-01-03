from dataclasses import dataclass 
from pathlib import Path 
from typing import List , Optional

@dataclass(frozen=True)
class Datasets:
    BRCA: str 
    KIPAN: str 
    BLCA: str 
    LUSC: str

@dataclass(frozen=True)
class KeggGo: 
    BRCA: str 
    KIPAN: str
    BLCA: str
    LUSC: str
    
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
    BLCA: OmicsRawDataPaths
    LUSC: OmicsRawDataPaths
    preprocessing: OmicTypeWithFilteringConfig
    test_split: float 
    n_bins: int 
    min_rules: int
    fold_change: bool
    discretize_level: int
    dataset: str
    random_state: Optional[int] = None 
    
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
    metric: str
    n_bins: int
    ppi_normalize: str 
    kegg_normalize: str
    kegg_sort: str 
    kegg_topk: int
    ac_normalize: str 
    corr: bool
    corr_filter: float
    kegg_filter: str
    ppi_filter: Optional[str] = None
    
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
    optimizer: str
    decay: float 
    gat_dropout: float 
    num_layer: int 
    num_block: int 
    pooling_rate: float 
    pooling: str
    momentum: float
    weight: Optional[List[float]] = None, 
    

@dataclass(frozen=True)
class EncoderTrainingConfig: 
    root_dir: Path 
    learning_rate: float
    data_preprocessing_dir: Path
    output_channel: int
    learning_epoch: int
    print_interval: int
    model: str
    
@dataclass(frozen=True)
class ARMClassificationConfig: 
    data_path: Path
    topk: List[int]
    dataset: str
    metric: str
    strategy: str
    
@dataclass(frozen=True)
class CompareOtherConfig:
    dnn: bool 
    gnn: bool
    corr_threshold: float
    epochs: int 
    hidden_units: int 
    learning_rate: float
    batch_size: int 
    drop_out: float
    negative_corr: bool
    pooling_ratio: float
    ppi: bool
    ppi_score: int 
    corr: bool
    ppi_edge: str
    discretized: bool
    n_bins: int
    select_k: str
    information: bool
    kegg: bool 
    kegg_filter: float 
    kegg_topk: int
    go: bool 
    go_filter: float
    go_topk: int
    filter: float
    info_mean: bool
    scale_edge: bool
    decay: float