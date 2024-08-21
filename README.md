# Python Libary Installation
## Create seperate environment for python using conda
Create seperate environment for python using the following command: `conda create -n gnn python=3.9` and activate the newly create environment `conda activate gnn`

## Install pytorch
Install latest pytorch lightning with GPU capability using command `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`. Please refer to <a href="https://pytorch.org/get-started/locally/">official website</a> for more information.

## Install pytorch lightning
Install lightning module for pytorch using using the command `conda install lightning -c conda-forge`. Please refer to <a href="https://lightning.ai/docs/pytorch/stable/starter/installation.html">lightning official website</a> for more information.

## Install pytorch geometric
Pytorch geometric is the high level module for pytorch to build the graph neural network. Install the module using command: `conda install pyg -c pyg`. `scikit-learn` python module will be installed together with the installtion

## Install other module dependancy
Install other python dependancies using the command: `pip install -r requirements.txt`

## Generate AC Rule with interestingness value 
mRNA `python discretization.py --custom_support 77,87,29,302,100 --min_confidence 0.95`
miRNA `Need to regenerate`
DNA `python discretization.py --custom_support 77,87,28,302,99 --min_confidence 0.95`


# D3.js 
```
.attr("fill" , d => d.data.top_10 ? "#f00" : "#000")

.attr("stroke" , (d) => (d[0].data.top_10 && d[1].data.top_10) ? "#ffb8b8" : colornone)

colornone = "#fafafa"
```