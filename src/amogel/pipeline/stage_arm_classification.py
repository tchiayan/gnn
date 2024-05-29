from amogel.components.arm_classification import ARM_Classification
from amogel.config.configuration import ConfigurationManager
import pandas as pd
from amogel import logger
import os
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
                arm_class = ARM_Classification(arm_config , arm_config.dataset , i , topk=topk)
                accuracy , info = arm_class.test_arm()
                summary[i] = round(accuracy*100,2)
                summary[f"{i}_info"] = info
                
            summaries.append(summary)
        df = pd.DataFrame(summaries)
        df['avg'] = df[[1,2,3]].mean(axis=1)
        print(df.to_string(index=False))
        
        os.makedirs('./artifacts/arm_classification', exist_ok=True)
        df.to_csv(f'./artifacts/arm_classification/report.csv', index=False)

if __name__ == "__main__":
    
    Stage_ARM_Classification.run()
            
    