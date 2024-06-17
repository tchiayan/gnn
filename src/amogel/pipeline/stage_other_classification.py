from amogel.components.classifier import OtherClassifier
from amogel import logger 

class OtherClassificationPipeline:
    
    def __init__(self):
        pass 
    
    def run(self):
        classifier = OtherClassifier()
        logger.info("Loading datasets")
        classifier.load_data()
        logger.info("Training and evaluating KNN model")
        classifier.train_and_evaluate_knn()
        logger.info("Training and evaluating SVM model")
        classifier.train_and_evaluate_svm()
        logger.info("Training and evaluating Naive Bayes model")
        classifier.train_and_evaluate_nb()
        logger.info("Training and evaluating DNN model")
        classifier.train_and_evaluate_dnn()
        
        
if __name__ == "__main__":
    pipeline = OtherClassificationPipeline()
    pipeline.run()