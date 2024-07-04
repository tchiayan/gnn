from amogel.config.configuration import ConfigurationManager
from amogel.utils.gene import biomarkers_selection
from amogel.utils.common import load_omic_features_name
from amogel import logger 
import torch

class BiomarkersPipeline:
    
    def __init__(self):
        self.config = ConfigurationManager()
        self.dataset = self.config.get_dataset()
        self.topk = self.config.get_topk()
        self.top_gene = self.config.get_biomarkers()

    def run(self):
        batch_file = f"./artifacts/amogel/batches_{self.topk}.pt"
        edge_attn_file = [
            f"./artifacts/amogel/edge_attn_l1_{self.topk}.pt", 
            f"./artifacts/amogel/edge_attn_l2_{self.topk}.pt"
        ]
        biomarkers = biomarkers_selection(
            batch_file=batch_file , 
            edge_attn_files=edge_attn_file , 
            topk=self.top_gene
        )
        
        # load genes name 
        omic_types = [ 1 , 2 , 3 ]
        feature_names = load_omic_features_name(
            "artifacts/data_preprocessing/" , self.dataset , [1,2,3]
        )
        
        # ac selected genes 
        ac_genes = torch.load("artifacts/ac_genes/gene.pt" , map_location=torch.device("cpu"))
        selected_features = feature_names[feature_names['gene_idx'].isin(list(ac_genes))].iloc[biomarkers.indices.numpy(),:]
        
        for i in omic_types:
            print(f"Top genes from omic type {i}") 
            for idx , row in selected_features[selected_features['omic_type'] == i].iterrows():
                print(row['gene_name'])
                
            print("------ end of list ------")
        
if __name__ == "__main__":
    pipeline = BiomarkersPipeline()
    pipeline.run()