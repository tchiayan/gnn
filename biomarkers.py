import argparse 
from pathlib import Path
import os
import time
import shutil


def copy(dataset , prior = True): 
    """
    Copy generated graph edges from dvc artifact to biomarkers directory
    """ 
    
    assert dataset in ["KIPAN" , "BRCA"] , "Dataset not supported"
    
    source_file = Path("artifacts/biomarkers/graph_edges.pt")
    
    _prior = "prior" if prior else "no_prior"
    target_dir = Path(f"biomarkers/{dataset}/{_prior}")
    current_datetime = time.strftime("%Y%m%d_%H%M%S")
    target_filename = f"graph_{current_datetime}.pt"  # filename with date and time
    target_file = os.path.join(target_dir , target_filename)
    
    # make directory if not exists
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # copy file
    shutil.copy(source_file, target_file)
    
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Biomarker Helper')
    
    
    parser.add_argument("--tool", type=str , choices=["copy"] , help="Tool to use")
    parser.add_argument("--noprior" , help="Use prior information" , action="store_true")
    parser.add_argument("--dataset" , type=str , choices=["KIPAN" , "BRCA"] , help="Dataset to use")
    
    args = parser.parse_args()
    
    if args.tool == "copy": 
        copy(args.dataset , not args.noprior)
    
    