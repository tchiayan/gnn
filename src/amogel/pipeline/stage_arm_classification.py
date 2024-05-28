from amogel.components.arm_classification import ARM_Classification
from amogel.config.configuration import ConfigurationManager
import pandas as pd
from amogel import logger

class Stage_ARM_Classification():
    
    def __init__(self) -> None:
        pass 
    
    
    def run():
        config = ConfigurationManager()
        arm_config = config.get_arm_classification_config()
        
        summaries = []
        for topk in arm_config.topk:
            logger.info(f"Test ARM Classification topk: {topk}")
            summary = {'topk': topk}
            for i in range(1 , 4):
                arm_class = ARM_Classification(arm_config.dataset , i , topk=topk)
                accuracy = arm_class.test_arm()
                summary[i] = round(accuracy*100,2)
                
            summaries.append(summary)
        df = pd.DataFrame(summaries)
        # print without index 
        print(df.to_string(index=False))
                

if __name__ == "__main__":
    
    Stage_ARM_Classification.run()
            
    