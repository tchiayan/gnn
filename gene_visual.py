import pandas as pd 
import matplotlib.pyplot as plt 

def main():
    
    # read generated gene list 
    df = pd.read_csv("gene_list.csv", header=None)
    
    # rename header 
    columns = df.columns.tolist()
    columns[0] = "label"
    df.columns = columns
    
    print(df.info())
    ax = df.hist(column=[1,2,3,4,5] , by=['label'])
    
    
    print(ax)
    fig = ax.get_figure()
    fig.save("gene_hist.png")
if __name__ == "__main__":
    
    main()
    
    