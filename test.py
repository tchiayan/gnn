import torch
from torch_geometric.nn import TopKPooling
from torch_geometric.data import Data , Batch

# Create a sample graph
edge_index = torch.tensor([[0, 1, ], [1, 0, ]], dtype=torch.long)
x = torch.rand((3, 16))  # 3 nodes with 16 features

data = Data(x=x, edge_index=edge_index)

# Create TopKPooling layer
pooling_layer = TopKPooling(in_channels=16, ratio=0.5)

# Apply TopKPooling
x, edge_index, top_node, batch, perm , score = pooling_layer(x, edge_index, None, None)

# Get the indices of the top-k nodes

print("Original graph nodes:", torch.arange(data.num_nodes))
print("Top-K nodes indices:", edge_index)
print("Top-K nodes features:", x.size())
print("Top-k nodes perm: " , perm)
print("Top-K nodes scores:", score)


import torch
from torch_geometric.utils import remove_isolated_nodes
from torch_geometric.data import Data

# Create a sample graph with isolated nodes
edge_index = torch.tensor([[0, 1, 1, 2, 1], [1, 0, 2, 1, 4]], dtype=torch.long)
x = torch.rand((5, 16))  # 5 nodes with 16 features

data = Data(x=x, edge_index=edge_index)
print(data)

# Print the original graph
print("Original graph:")
print("Edge indices:", data.edge_index.t())
print("Node features:", data.x)

# Remove isolated nodes
edge_index, edge_attr , mapping = remove_isolated_nodes(data.edge_index, num_nodes=data.num_nodes)

# Update node features accordingly
x = x[mapping]

# Create a new Data object with the updated graph
data = Data(x=x, edge_index=edge_index , edge_attr=edge_attr, num_nodes=x.size(0))

# Print the graph after removing isolated nodes
print("\nGraph after removing isolated nodes:")
print("Edge indices:", data.edge_index.t())
print("Node features:", data.x)
print(data)

# Create a sample graph with multiple batches
edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long)
x = torch.rand((5, 16))  # 5 nodes with 16 features
batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)  # Batch information

data = Data(x=x, edge_index=edge_index, batch=batch)

# Create a Batch object
batched_data = Batch.from_data_list([data , data])
print("===== batched_data =====")
print(batched_data)

# Create TopKPooling layer
pooling_layer = TopKPooling(in_channels=16, ratio=0.5)

# Apply TopKPooling to the batched data
x_pooled, edge_index_pooled, _, batch_pooled, perm , score = pooling_layer(batched_data.x, batched_data.edge_index, None, batched_data.batch)
print(perm)
print(score)
print(batch_pooled)

# Get a dense batch of selected nodes using perm
dense_batch = torch.index_select(batched_data.x, dim=0, index=perm)
print(dense_batch)

# Recover the original node indices in the dense batch
original_indices = torch.arange(batched_data.num_nodes)[perm]

print("Original node indices in dense batch:")
print(original_indices)