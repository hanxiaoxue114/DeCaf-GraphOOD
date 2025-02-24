## DeCaf 
This is the source code for paper ''[DeCaf: A Causal Decoupling Framework for OOD Generalization on Node Classification](https://arxiv.org/abs/2410.20295)'' to appear In Proceedings of the 28th International Conference on Artificial Intelligence and Statistics (AISTATS 2025). 

[Xiaoxue Han](https://hanxiaoxue114.github.io/), Huzefa Rangwala, [Yue Ning](https://yue-ning.github.io/)

## Prerequisites
The code has been successfully tested in the following environment. (For older versions, you may need to modify the code)
- Python 3.8.13
- PyTorch 1.12.1+cu11.6
- pygsp 0.5.1
- sklearn 1.1.2


## Getting Started

### Training and testing
Please run following commands for training and testing under the `src` folder. We take the dataset `kindle` with GCN as backbone GNN model as the example.


**Evaluate the TACO model**
```python
python -W ignore train.py --dataset cora --gnn GCN --ood_type label --hidden 64
```

## Cite
Please cite our paper if you find this code useful for your research:


**BibTeX**

```
@misc{han2024decafcausaldecouplingframework,
      title={DeCaf: A Causal Decoupling Framework for OOD Generalization on Node Classification}, 
      author={Xiaoxue Han and Huzefa Rangwala and Yue Ning},
      year={2024},
      eprint={2410.20295},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.20295}, 
}
```
