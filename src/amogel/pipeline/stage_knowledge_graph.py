from amogel import logger 
from amogel.components.knowledge_graph import KnowledgeGraph 
from amogel.config.configuration import ConfigurationManager

STAGE_NAME = 'Knowledge Graph Generation'

class KnowledgeGraphPipeline():
        
    def __init__(self) -> None:
        pass
    
    def run(self):
        config = ConfigurationManager()
        knowledge_graph_config = config.get_knowledge_graph_config()
        
        for i in [1,2,3]:
            # generate knowledge graph for omic type 
            knowledge_graph = KnowledgeGraph(
                config=knowledge_graph_config , 
                omic_type=i , 
                dataset=config.get_dataset()
            )
            
            if knowledge_graph_config.dataset == 'triplet_multigraph':
                knowledge_graph.generate_triplet_multigraph( ppi=knowledge_graph_config.ppi , kegg_go=knowledge_graph_config.kegg_go , synthetic=knowledge_graph_config.synthetic)
            elif knowledge_graph_config.dataset == 'binarylearning_multigraph':
                knowledge_graph.generate_binaryclassifier_multigraph( ppi=knowledge_graph_config.ppi , kegg_go=knowledge_graph_config.kegg_go , synthetic=knowledge_graph_config.synthetic )
            elif knowledge_graph_config.dataset == 'common_multigraph':
                knowledge_graph.generate_common_graph( ppi=knowledge_graph_config.ppi , kegg_go=knowledge_graph_config.kegg_go , synthetic=knowledge_graph_config.synthetic )
            elif knowledge_graph_config.dataset == 'multiedges_multigraph':
                knowledge_graph.generate_multiedges_graph( ppi=knowledge_graph_config.ppi , kegg_go=knowledge_graph_config.kegg_go , synthetic=knowledge_graph_config.synthetic )
            elif knowledge_graph_config.dataset == 'unified_multigraph_test':
                knowledge_graph.generate_unified_test_multigraph( ppi=knowledge_graph_config.ppi , kegg_go=knowledge_graph_config.kegg_go , synthetic=knowledge_graph_config.synthetic )
            elif knowledge_graph_config.dataset == 'discretized_common_multigraph':
                knowledge_graph.generate_discretized_common_graph( ppi=knowledge_graph_config.ppi , kegg_go=knowledge_graph_config.kegg_go , synthetic=knowledge_graph_config.synthetic )
            elif knowledge_graph_config.dataset == 'discretized_multiedges_multigraph':
                knowledge_graph.generate_discretized_multiedges_graph( ppi=knowledge_graph_config.ppi , kegg_go=knowledge_graph_config.kegg_go , synthetic=knowledge_graph_config.synthetic, corr=knowledge_graph_config.corr )
            else:
                raise ValueError(f"Invalid dataset : {knowledge_graph_config.dataset}")
            # knowledge_graph.generate_knowledge_graph( ppi=True  , kegg_go=True )
            # knowledge_graph.generate_unified_graph( ppi=True , kegg_go=True , synthetic=True)
            # knowledge_graph.generate_unified_multigraph( ppi=True , kegg_go=True , synthetic=True)
            # knowledge_graph.generate_contrastive_multigraph( ppi=True , kegg_go=True , synthetic=True)
            # knowledge_graph.generate_correlation_graph( ppi=True , kegg_go=True , synthetic=True)
            # knowledge_graph.generate_binaryclassifier_multigraph( ppi=True , kegg_go=True , synthetic=True)
            # knowledge_graph.generate_triplet_multigraph( ppi=knowledge_graph_config.ppi , kegg_go=knowledge_graph_config.kegg_go , synthetic=knowledge_graph_config.synthetic)
            # knowledge_graph.summary()
        
if __name__ == "__main__":
    
    logger.info(f"-------- Running {STAGE_NAME} --------")
    
    try:
        main = KnowledgeGraphPipeline()
        main.run()
    except Exception as e:
        logger.error(f"Error in {STAGE_NAME} : {e}")
        raise e
    logger.info(f"-------- Completed {STAGE_NAME} --------")