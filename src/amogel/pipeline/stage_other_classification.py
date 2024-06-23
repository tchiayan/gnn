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
        classifier.load_data(select_k=config.select_k)
        # logger.info("Training and evaluating KNN model")
        # classifier.train_and_evaluate_knn()
        # logger.info("Training and evaluating SVM model")
        # classifier.train_and_evaluate_svm()
        # logger.info("Training and evaluating Naive Bayes model")
        # classifier.train_and_evaluate_nb()
        # logger.info("Training and evaluating DNN model")
        # classifier.train_and_evaluate_dnn()
        # logger.info("Training and evaluating with DNN feature selection model")
        # classifier.train_and_evaluate_dnn_feature_selection()
        # logger.info("Training and evaluating with DNN feature selection model")
        # classifier.train_and_evaluate_dnn_feature_selection_ac()
        logger.info("Training and evaluating with Graph feature selection model")
        classifier.train_and_evaluate_graph_feature_selection_ac()
        
if __name__ == "__main__":
    pipeline = OtherClassificationPipeline()
    pipeline.run()