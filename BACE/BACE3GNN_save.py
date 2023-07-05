# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:19:03 2022

@author: pedro
"""


import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn import Linear

from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from sklearn.metrics import accuracy_score , roc_auc_score , precision_score , recall_score
import seaborn as  sns
import shutil
import random
import warnings


df = pd.read_csv('bace.csv')


df['Class'].value_counts() 
df['Class'].value_counts().plot.pie(autopct='%.2f')

X = list(df['mol'])
y = list(df['Class'])



def one_hot_encoding(x, permitted_list):
    """
    This implementation was adapted from: https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding

def get_atom_features(atom, 
                      use_chirality = True, 
                      hydrogens_implicit = True):
    """
    This implementation was adapted from: https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """
    # list of permitted atoms
    
    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
    
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    # atom features
    
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    
    is_in_a_ring_enc = [int(atom.IsInRing())]
    
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
    
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
    
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
                                    
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc
    return np.array(atom_feature_vector)

def get_bond_features(bond, 
                      use_stereochemistry = True):
    """
    This implementation was adapted from: https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
    Takes an RDKit bond object as input and gives a 1d array of bond features as output.
    """
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc
    return np.array(bond_feature_vector)

def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y):
    """
    This implementation was adapted from: https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
    Inputs:
    
    x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
    y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)
    
    Outputs:
    
    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs 
    
    """
    
    data_list = []
    
    for (smiles, y_val) in zip(x_smiles, y):
        
        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles)
        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2*mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)
            
        X = torch.tensor(X, dtype = torch.float)
        
        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0)
        
        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))
        
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        
        EF = torch.tensor(EF, dtype = torch.float)
        
        # construct label tensor
        y_tensor = torch.tensor(np.array([y_val]), dtype = torch.float)
        
        # construct Pytorch Geometric data object and append to data list
        data_list.append(Data(x = X, edge_index = E, edge_attr = EF, y = y_tensor))
    return data_list

data_list = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(X, y)

random.shuffle(data_list)
data = DataLoader(dataset = data_list, batch_size = 64)


def save_ckp(state, is_best, checkpoint_dir, best_model_dir, filename, best_model):
    f_path = checkpoint_dir + filename
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + best_model
        shutil.copyfile(f_path, best_fpath)
        
def optimizer_to(optim, device):
    # move optimizer to device
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def load_ckp(checkpoint_fpath, model, optimizer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))
    checkpoint = torch.load(checkpoint_fpath, map_location = device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.to(device)

    optimizer_to(optimizer, device)
    return model, optimizer, checkpoint['epoch']

#-----------------------------------------------------------------------------------------  
embedding_size = 400
#-----------------------------------------------------------------------------------------  

CUDA_LAUNCH_BLOCKING=1

class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        # GIN layers - comment the layers that are not going to be used

        self.mlp1 = nn.Sequential(nn.Linear(79, embedding_size),
                                  nn.BatchNorm1d(embedding_size), nn.ReLU(),
                                  nn.Linear(embedding_size, embedding_size), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Linear(embedding_size, embedding_size),
                                  nn.BatchNorm1d(embedding_size), nn.ReLU(),
                                  nn.Linear(embedding_size, embedding_size), nn.ReLU())
        self.ginconv1 = GINConv(self.mlp1 , eps=0.00005, train_eps=True)
        self.out_mlp1 = nn.Linear(embedding_size , embedding_size)
        self.ginconv2 = GINConv(self.mlp2 , eps=0.00005, train_eps=True)
        self.out_mlp2 = nn.Linear(embedding_size , embedding_size)
        self.ginconv3 = GINConv(self.mlp2 , eps=0.00005, train_eps=True)
        self.out_mlp3 = nn.Linear(embedding_size , embedding_size)


        # GCN layers - comment the layers that are not going to be used
        
        #self.initial_conv = GCNConv(79, embedding_size)
        #self.conv1 = GCNConv(embedding_size, embedding_size)
        #self.conv2 = GCNConv(embedding_size, embedding_size)
        
        
        #Graph Sage layers -  comment the layers that are not going to be used
        
        #self.initial_sage = SAGEConv(79, embedding_size)
        #self.sage1 = SAGEConv(embedding_size , embedding_size)
        #self.sage2 = SAGEConv(embedding_size , embedding_size)
        
        
        # GAT layers - comment the layers that are not going to be used
        
        #self.initial_att = GATConv(79,embedding_size)
        #self.att1 = GATConv(embedding_size,embedding_size)
        #self.att2 = GATConv(embedding_size,embedding_size)

        #------------------------------------------------------------------
        self.drop = nn.Dropout(p=0.25)
        #---------------------------------------------------------------------------

        # Output layer
        self.out = Linear(embedding_size*2, 1)

    def forward(self, x, edge_index, batch_index):

#------------------------- Layers ---------------------------#            
        
        #GIN
        hidden = self.ginconv1(x, edge_index)
        hidden = self.out_mlp1(hidden)
        hidden = F.tanh(hidden)
        hidden = self.ginconv2(hidden, edge_index)
        hidden = self.out_mlp2(hidden)
        hidden = F.tanh(hidden)
        hidden = self.ginconv3(hidden, edge_index)
        hidden = self.out_mlp3(hidden)
        hidden = F.tanh(hidden)

        #GCN
        #hidden = self.initial_conv(x, edge_index)
        #hidden = F.tanh(hidden)
        #hidden = self.conv1(hidden, edge_index)
        #hidden = F.tanh(hidden)
        #hidden = self.conv2(hidden, edge_index)
        #hidden = F.tanh(hidden)
        
        #Sage
        #hidden = self.initial_sage(x, edge_index)
        #hidden = F.tanh(hidden)
        #hidden = self.sage1(hidden, edge_index)
        #hidden = F.tanh(hidden)
        #hidden = self.sage2(hidden, edge_index)
        #hidden = F.tanh(hidden) 

        #GAT 
        #hidden = self.initial_att(x, edge_index)
        #hidden = F.tanh(hidden)
        #hidden = self.att1(hidden, edge_index)
        #hidden = F.tanh(hidden)
        #hidden = self.att2(hidden, edge_index)
        #hidden = F.tanh(hidden)

        hidden = self.drop(hidden)
        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)   
        
        # Apply a final (linear) classifier.
        hidden = self.out(hidden)

        return hidden 

model = GNN()
print(model)

warnings.filterwarnings("ignore")

optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)

# Use GPU for training
device = torch.device("cuda:0")# if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Wrap data in a data loader
data_size = len(data_list)

#-----------------------------------------------------------------------------------------   
#batch size:
NUM_GRAPHS_PER_BATCH = 64
#Loss function weight:
pos_weight = torch.FloatTensor([1]).to(device)
criterion=torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#-----------------------------------------------------------------------------------------  

loader = DataLoader(data_list[:int(data_size * 0.85)], 
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True, drop_last=True)
test_loader = DataLoader(data_list[int(data_size * 0.85):], 
                         batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True, drop_last = True)

def train(loader):
    # Enumerate over the data
    for batch in loader:
      # Use GPU
      batch.to(device)  
      # Reset gradients
      optimizer.zero_grad() 
      # Passing the node features and the connection info
      pred = model(batch.x.float(), batch.edge_index, batch.batch)

      pred=torch.reshape(pred,(NUM_GRAPHS_PER_BATCH,))

      loss = criterion(pred,batch.y)
      
      # Calculating the loss and gradients
      loss.backward()  

      # Update using the gradients
      optimizer.step()   
    return loss,  optimizer

def test(loader):
    model.eval()
    y_score=[]
    y_true=[]
    for batch in loader:
      batch.to(device)  
      pred = model(batch.x.float(), batch.edge_index, batch.batch) 
      pred=torch.sigmoid(pred)
      
      pred = torch.where(pred>0.5, torch.ones_like(pred), pred)
      pred = torch.where(pred<=0.5, torch.zeros_like(pred), pred)

      y_score.append(pred)
      y_true.append(batch.y)

    y_true = torch.cat(y_true, dim = 0).cpu().detach().numpy() 
    y_score = torch.cat(y_score, dim = 0).cpu().detach().numpy()

    y_true = [int(x) for x in y_true]
    y_score = [int(x) for x in y_score]
    accuracy_score(y_true , y_score)
    roc_list=[]
    roc_list.append(roc_auc_score(y_true, y_score))
    acc = sum(roc_list)/len(roc_list)

    precision = precision_score(y_true, y_score)
    specificity = recall_score(y_true, y_score)

    cm=confusion_matrix(y_true=y_true,y_pred=y_score)
    return acc, precision, specificity, cm

print("Starting training...")
losses = [0]
acc_list = [0]
precision_list=[0]
specificity_list=[0]


for epoch in range(10000):
    best_epoch = False
    loss, optimizer = train(loader)
    losses.append(loss)
    roc_auc, precision, specificity, cm = test(test_loader)
    
    if roc_auc >= max(acc_list):
        best_epoch = True

    acc_list.append(roc_auc)
    precision_list.append(precision)
    specificity_list.append(specificity)
    
    checkpoint_gnn = {'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}
   
#----------------------------------------------------------------------------------------------
#             Give a name to the file 
    NAME = "NAME_THIS"
    checkpoint_dir = "checkpoints_bace/" + NAME  # Give a name to the file 
    model_dir = "checkpoints_bace/" + NAME + "_model" # Give a name to the file 
    
    #name = "results_bace/" + NAME + ".txt"        # Give a name to the file 
    #file = open(name,"a")
    #file.write("roc_auc: ")
    #file.write(str(roc_auc))
    #file.write("\n")
    #file.close()

    save_ckp(checkpoint_gnn, best_epoch, checkpoint_dir, model_dir, "/checkpoints_" + NAME + "_bace.pt", "/model_" + NAME + "_bace.pt" )


    if epoch % 20==0:
      print(max(acc_list))
      print(f"Epoch {epoch} | Train Loss {loss} | roc auc score {roc_auc}")
      sns.set(style="darkgrid")
      acc_indices = [i for i,l in enumerate(acc_list)]
      grafico = sns.lineplot(x=acc_indices, y=acc_list)
      plt.show()
      