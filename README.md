# ENGAGE
This repository hosts the code for our ECML'23 paper 'ENGAGE: Explanation Guided Data Augmentation for Graph Representation Learning' by Yucheng Shi, Kaixiong Zhou, Ninghao Liu.

## Dependencies
* torch 1.10.1+cu113 
* torch-cluster 1.5.9 
* torch-geometric 2.0.3 
* torch-scatter 2.0.9
* torch-sparse 0.6.12
* faiss-cpu 1.7.2

If you have trouble in installing `torch-geometric`, you may find help in its [official website](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

## Training & Evaluation
### Graph-level
For SimCLR model:
```
python Simclr_graph_self_guided.py --dataset DD
```
For Simsiam model:
```
python Simsiam_graph_self_guided.py --dataset DD
```
### Node-level
For SimCLR model:
```
python Simclr_self_guided.py --dataset Cora --model GAT
```
For Simsiam model:
```
python Simsiam_self_guided.py --dataset Cora --model GCN
```
## Acknowledgements
Parts of implementation are reference to [GRACE](https://github.com/CRIPAC-DIG/GRACE), [GraphCL](https://github.com/Shen-Lab/GraphCL), and [Simsiam](https://github.com/PatrickHua/SimSiam).
