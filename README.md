## A Hybrid GNN Approach for Improved Molecular
Property Prediction


In this work, we propose a hybrid approach that incorporates different graph-based methods to combine their strengths and mitigate their limitations to accurately predict molecular properties. The proposed approach consists in a multi-layered hybrid GNN architecture that integrates multiple GNN frameworks to compute graph embeddings for molecular property prediction. 

Furthermore, we conduct extensive experiments on multiple benchmark datasets to demonstrate that our hybrid approach significantly outperforms the state-of-the-art graph-based models.

![ScreenShot](Figures/Model_Architecture.png?raw=true)

This repository provides the source code and datasets for the proposed work.

Contact Information: uc2019119752@student.uc.pt, if you have any questions about this work.


## Data Collection and Processing

To evaluate our hybrid approach, we chose to train and test the developed
models with three well-known benchmarks for molecular property prediction.
These benchmarks contain molecules described as SMILES with a binary la-
bel, each dataset labels the molecule according to a molecular property. These benchmarks are the HIV and BACE of the biophysics category and Tox21 of the physiology category of MoleculeNet [Wu et al. (2018b)] (https://doi.org/10.1039/c7sc02664a).


## System Requirements

We used the following packages for code implementation:

```
-torch = 1.9.0
-cuda = 10.2.0
-torch-cluster = 1.5.9
-torch-geometric = 2.1.0
-torch-scatter = 2.0.8
-torch-sparse = 0.6.11
-torch-spline-conv = 1.2.1
-torchvision = 0.10.0
-scikit-learn = 1.1.1
-seaborn = 0.12.0
-scipy = 1.7.3
-numpy = 1.21.5
-matplotlib = 3.5.3
-pandas = 1.4.3
-networkx = 2.8.6
-rdkit = 2022.03.5
```


## References

[1]Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., Leswing, K., & Pande, V. (2018b). Moleculenet: A benchmark for molecular machine learning. Chemical Science, 9 . doi:10.1039/c7sc02664a.

```
@article{molnet,
   author = {Zhenqin Wu and Bharath Ramsundar and Evan N. Feinberg and Joseph Gomes and Caleb Geniesse and Aneesh S. Pappu and Karl Leswing and Vijay Pande},
   doi = {10.1039/c7sc02664a},
   title = {MoleculeNet: A benchmark for molecular machine learning},
   year = {2018},
}
```
