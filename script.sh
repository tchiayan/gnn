python gnpool.py --model dmongraph_pool --build_graph PPI --edge_threshold 300 --batch_size 10 --hidden_embedding 32 --lr 0.00001 --max_epoch 2000 --runkfold 5 --disable_early_stopping 
python gnpool.py --model dmongraph_pool --build_graph PPI --edge_threshold 200 --batch_size 5 --hidden_embedding 32 --lr 0.00001 --max_epoch 2000 --runkfold 5 --disable_early_stopping 
python gnpool.py --model dmongraph_pool --build_graph pearson --edge_threshold 0.95 --batch_size 15 --hidden_embedding 32 --lr 0.00001 --max_epoch 2000 --runkfold 5 --disable_early_stopping 
python gnpool.py --model dmongraph_pool --build_graph pearson --edge_threshold 0.8 --batch_size 15 --hidden_embedding 32 --lr 0.00001 --max_epoch 2000 --runkfold 5 --disable_early_stopping 
python gnpool.py --model dmongraph_pool --build_graph pearson --edge_threshold 0.6 --batch_size 10 --hidden_embedding 32 --lr 0.00001 --max_epoch 2000 --runkfold 5 --disable_early_stopping
python gnpool.py --model dmongraph_pool --build_graph PPI --edge_threshold 400 --batch_size 15 --hidden_embedding 32 --lr 0.000005 --max_epoch 2000 --runkfold 5 --disable_early_stopping 
python gnpool.py --model dmongraph_pool --build_graph PPI --edge_threshold 300 --batch_size 10 --hidden_embedding 32 --lr 0.000005 --max_epoch 2000 --runkfold 5 --disable_early_stopping 
python gnpool.py --model dmongraph_pool --build_graph PPI --edge_threshold 200 --batch_size 5 --hidden_embedding 32 --lr 0.000005 --max_epoch 2000 --runkfold 5 --disable_early_stopping 
python gnpool.py --model dmongraph_pool --build_graph pearson --edge_threshold 0.95 --batch_size 15 --hidden_embedding 32 --lr 0.000005 --max_epoch 2000 --runkfold 5 --disable_early_stopping 
python gnpool.py --model dmongraph_pool --build_graph pearson --edge_threshold 0.8 --batch_size 15 --hidden_embedding 32 --lr 0.000005 --max_epoch 2000 --runkfold 5 --disable_early_stopping 
python gnpool.py --model dmongraph_pool --build_graph pearson --edge_threshold 0.6 --batch_size 10 --hidden_embedding 32 --lr 0.000005 --max_epoch 2000 --runkfold 5 --disable_early_stopping  
python gnpool.py --model singlegraph_diffpool --max_epoch 500 --lr 1e-5 --edge_threshold 0.95 --convolution GCNConv --build_graph pearson --runkfold 5
python gnpool.py --model singlegraph_diffpool --max_epoch 500 --lr 1e-5 --edge_threshold 0.8 --convolution GCNConv --build_graph pearson --runkfold 5
python gnpool.py --model singlegraph_diffpool --max_epoch 500 --lr 1e-5 --edge_threshold 0.6 --convolution GCNConv --build_graph pearson --runkfold 5
python gnpool.py --model singlegraph_diffpool --max_epoch 500 --lr 1e-5 --edge_threshold 0.95 --convolution GraphConv --build_graph pearson --runkfold 5
python gnpool.py --model singlegraph_diffpool --max_epoch 500 --lr 1e-5 --edge_threshold 0.8 --convolution GraphConv --build_graph pearson --runkfold 5
python gnpool.py --model singlegraph_diffpool --max_epoch 500 --lr 1e-5 --edge_threshold 0.6 --convolution GraphConv --build_graph pearson --runkfold 5
python gnpool.py --model singlegraph_diffpool --max_epoch 500 --lr 1e-5 --edge_threshold 0.95 --convolution GATConv --build_graph pearson --runkfold 5
python gnpool.py --model singlegraph_diffpool --max_epoch 500 --lr 1e-5 --edge_threshold 0.8 --convolution GATConv --build_graph pearson --runkfold 5
python gnpool.py --model singlegraph_diffpool --max_epoch 500 --lr 1e-5 --edge_threshold 0.6 --convolution GATConv --build_graph pearson --runkfold 5
python gnpool.py --model singlegraph_diffpool --max_epoch 500 --lr 1e-5 --edge_threshold 0.95 --convolution SAGEConv --build_graph pearson --runkfold 5
python gnpool.py --model singlegraph_diffpool --max_epoch 500 --lr 1e-5 --edge_threshold 0.8 --convolution SAGEConv --build_graph pearson --runkfold 5
python gnpool.py --model singlegraph_diffpool --max_epoch 500 --lr 1e-5 --edge_threshold 0.6 --convolution SAGEConv --build_graph pearson --runkfold 5