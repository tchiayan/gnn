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
        
        # generate knowledge graph for omic type 1
        knowledge_graph = KnowledgeGraph(
            config=knowledge_graph_config , 
            omic_type=1 , 
            dataset="BRCA"
        )
        knowledge_graph.generate_graph()
        
        # generate knowledge graph for omic type 2
        # knowledge_graph = KnowledgeGraph(
        #     config=knowledge_graph_config , 
        #     omic_type=2 , 
        #     dataset="BRCA"
        # )
        # knowledge_graph.generate_graph()
        
        # # generate knowledge graph for omic type 3
        # knowledge_graph = KnowledgeGraph(
        #     config=knowledge_graph_config , 
        #     omic_type=3 , 
        #     dataset="BRCA"
        # )
        knowledge_graph.generate_graph()