from amogel.components.classifier import OtherClassifier
from amogel.config.configuration import ConfigurationManager
from amogel import logger 

class OtherClassificationPipeline:
    
    def __init__(self):
        self.dataset = ConfigurationManager().get_dataset()
    
    def run(self):
        config = ConfigurationManager().get_compare_other_configurations()
        classifier = OtherClassifier(config=config , dataset=self.dataset)
        logger.info("Loading datasets")
        classifier.load_data()
        logger.info("Training and evaluating DNN with AC model")
        classifier.train_and_evaluate_dnn_feature_selection_ac()
        
if __name__ == "__main__":
    pipeline = OtherClassificationPipeline()
    pipeline.run()