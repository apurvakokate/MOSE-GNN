import numpy as np
import torch
# RDkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
# from torch_geometric.utils.smiles import from_smiles Adds many atoms
from torch_geometric.data import Data, Dataset
import pandas as pd
import pickle
import pdb

def get_setup_files_with_folds(dataset_name, date_tag, fold, algorithm):
    algorithm = 'RBRICS' if algorithm == 'None' else algorithm
    least_count_dict = {'Mutagenicity':{'RBRICS':3, 'MGSSL':3}, 
                       'hERG':{'RBRICS':10,'MGSSL':15}, 
                       'BBBP':{'RBRICS':6,'MGSSL':15},
                       'Lipophilicity':{'RBRICS':10,'MGSSL':15},
                       'tox21':{'RBRICS':5,'MGSSL':3},
                       'esol':{'RBRICS':5,'MGSSL':5}}
    # print(algorithm)
    least_count = least_count_dict[dataset_name][algorithm]
    
    path = "../DICTIONARY"
    
    with open(f'{path}/FOLDS/{dataset_name}_{algorithm}_fold_{fold}_leastcount_{least_count}_{date_tag}_graph_lookup.pickle', 'rb') as file:
        lookup = pickle.load(file)
    with open(f'{path}/FOLDS/{dataset_name}_{algorithm}_fold_{fold}_leastcount_{least_count}_{date_tag}_motif_list.pickle', 'rb') as file:
        motif_list = list(pickle.load(file))
    with open(f'{path}/FOLDS/{dataset_name}_{algorithm}_fold_{fold}_leastcount_{least_count}_{date_tag}_motif_counts.pickle', 'rb') as file:
        motif_counts = pickle.load(file)
    with open(f'{path}/FOLDS/{dataset_name}_{algorithm}_fold_{fold}_leastcount_{least_count}_{date_tag}_motif_length.pickle', 'rb') as file:
        motif_lengths = pickle.load(file)
    with open(f'{path}/FOLDS/{dataset_name}_{algorithm}_fold_{fold}_leastcount_{least_count}_{date_tag}_motif_class.pickle', 'rb') as file:
        motif_class_count = pickle.load(file)
    with open(f'{path}/FOLDS/{dataset_name}_{algorithm}_fold_{fold}_leastcount_{least_count}_{date_tag}_graph_motifidx.pickle', 'rb') as file:
        graph_to_motifs = pickle.load(file)
    with open(f'{path}/FOLDS/{dataset_name}_{algorithm}_fold_{fold}_leastcount_{least_count}_{date_tag}_test_graph_lookup.pickle', 'rb') as file:
        # Serialize and save the object to the file
        test_data_lookup = pickle.load(file)
    with open(f'{path}/FOLDS/{dataset_name}_{algorithm}_fold_{fold}_leastcount_{least_count}_{date_tag}_test_graph_motifidx.pickle', 'rb') as file:
        test_graph_to_motifs = pickle.load(file)
    with open(f'{path}/FOLDS/{dataset_name}_{algorithm}_fold_{fold}_leastcount_{least_count}_{date_tag}_test_dataset_masked.pickle', 'rb') as file:
        test_mask_data = pickle.load(file)
    with open(f'{path}/FOLDS/{dataset_name}_{algorithm}_fold_{fold}_leastcount_{least_count}_{date_tag}_train_dataset_masked.pickle', 'rb') as file:
        train_mask_data = pickle.load(file)
    with open(f'{path}/FOLDS/{dataset_name}_{algorithm}_fold_{fold}_leastcount_{least_count}_{date_tag}_validation_dataset_masked.pickle', 'rb') as file:
        val_mask_data = pickle.load(file)
        
    return lookup, motif_list, motif_counts, motif_lengths, motif_class_count, graph_to_motifs, test_data_lookup, test_graph_to_motifs, train_mask_data, val_mask_data, test_mask_data


def get_setup_files(dataset_name, date_tag):
    with open(f'dictionary/{dataset_name}_graph_lookup_{date_tag}.pickle', 'rb') as file:
        lookup = pickle.load(file)
    with open(f'dictionary/{dataset_name}_motif_list_{date_tag}.pickle', 'rb') as file:
        motif_list = list(pickle.load(file))
    with open(f'dictionary/{dataset_name}_motif_counts_{date_tag}.pickle', 'rb') as file:
        motif_counts = pickle.load(file)
    with open(f'dictionary/{dataset_name}_motif_class_{date_tag}.pickle', 'rb') as file:
        motif_class_count = pickle.load(file)
    with open(f'dictionary/{dataset_name}_graph_motifidx_{date_tag}.pickle', 'rb') as file:
        graph_to_motifs = pickle.load(file)
    with open(f'dictionary/{dataset_name}_test_graph_lookup_{date_tag}.pickle', 'rb') as file:
        # Serialize and save the object to the file
        test_data_lookup = pickle.load(file)
    with open(f'dictionary/{dataset_name}_test_graph_motifidx_{date_tag}.pickle', 'rb') as file:
        test_graph_to_motifs = pickle.load(file)
    with open(f'dictionary/{dataset_name}_test_dataset_masked_{date_tag}.pickle', 'rb') as file:
        test_mask_data = pickle.load(file)
    with open(f'dictionary/{dataset_name}_train_dataset_masked_{date_tag}.pickle', 'rb') as file:
        train_mask_data = pickle.load(file)
    with open(f'dictionary/{dataset_name}_validation_dataset_masked_{date_tag}.pickle', 'rb') as file:
        val_mask_data = pickle.load(file)
        
    return lookup, motif_list, motif_counts, motif_class_count, graph_to_motifs, test_data_lookup, test_graph_to_motifs, train_mask_data, val_mask_data, test_mask_data

def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """

    if x not in permitted_list:
        return None
    #     x = permitted_list[-1]

    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

    return binary_encoding
def get_atom_features(atom, 
                      use_chirality = True, 
                      hydrogens_implicit = False):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """

    # define list of permitted atoms
    
    permitted_list_of_atoms =  ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I',
                               'Cu', 'Bi', 'B', 'Zn', 'Hg', 'Ti', 'Fe', 'Au', 'Mn', 'Tl',
                               'As','Ca', 'Si', 'Co', 'Al','Na', 'Ni', 'K', 'Sn', 'Cr', 'Dy', 'Zr', 'Sb', 'In', 'Yb', 'Nd',
                               'Be', 'Se', 'Cd', 'Li', 'Mg', 'Pt', 'Gd', 'V', 'Ge', 'Mo', 'Ag', 'Ba', 'Pb', 'Sr', 'Pd']
    
    # if hydrogens_implicit == False:
    permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    # compute atom features
    
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    if atom_type_enc is None:
        return None

    atom_feature_vector = atom_type_enc 
                                    
    # if use_chirality == True:
    #     chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
    #     atom_feature_vector += chirality_type_enc

    return np.array(atom_feature_vector)

def get_bond_features(bond, 
                      use_stereochemistry = True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """

    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc #+ bond_is_conj_enc + bond_is_in_ring_enc
    if bond_feature_vector is None:
        return None
    
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        if stereo_type_enc is None:
            return bond_feature_vector
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)
def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(smiles, y_val):
    """
    Inputs:
    
    x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
    y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)
    
    Outputs:
    
    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
    
    """   
    # convert SMILES to RDKit mol object
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        # input(smiles)
        return None  # Ignore invalid SMILES

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
        if get_atom_features(atom) is not None:
            X[atom.GetIdx(), :] = get_atom_features(atom)
        else:
            input(f"Error processing {smiles} at atom {atom.GetSymbol()}. Verify before continuing")
        

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
    
#     print(y_val)
#     print(type(y_val))
#     input(y_val.dtype)  # Check the data type of the numpy array

    # construct label tensor
    y_tensor = torch.tensor(np.array([y_val]), dtype = torch.float)

    # construct Pytorch Geometric data object and append to data list
    return Data(x = X, edge_index = E, edge_attr = EF, y = y_tensor, smiles=smiles)

class MolDataset(Dataset):
    def __init__(self, root, csv_file,split, label_col, transform=None, pre_transform=None, task_type='BinaryClass', normalize = False, mean = None, std = None):
        self.df = pd.read_csv(csv_file)
        self.dataset_name = csv_file.split('.csv')[0]
        self.label_col = label_col
        self.labels = self.df[label_col].values.flatten()  # Flatten to ensure array shape compatibility
        self.normalize = normalize
        
        super(MolDataset, self).__init__(root, transform, pre_transform)
        
        # Determine the number of unique classes in the label column
        if task_type == 'MultiClass':
            self._num_classes = self.df[label_col[0]].nunique()
        if task_type == 'Regression':
            self._num_classes = 1
        elif task_type == 'BinaryClass':
            self._num_classes = 2
        else:
            self._num_classes = len(label_col)
        self.split = split
        
        # Split the dataset based on the 'group' column
        if self.split == 'training':
            self.df = self.df[self.df['group'] == 'training']
        elif self.split == 'valid':
            self.df = self.df[self.df['group'] == 'valid']
        elif self.split == 'test':
            self.df = self.df[self.df['group'] == 'test']
        self.df.reset_index(drop=True, inplace=True)    
        
        # Compute mean and std from the training split if normalization is enabled
        if self.normalize:
            if mean is None and std is None:
                self.mean = self.df[self.label_col].mean().iloc[0]
                self.std = self.df[self.label_col].std().iloc[0]
            else:
                self.mean = mean
                self.std = std
        
        

    def len(self):
        return len(self.df)
    
    @property
    def num_classes(self):
        return self._num_classes  # Access the private attribute

    @num_classes.setter
    def num_classes(self, value):
        self._num_classes = value  # Allows setting num_classes externally if needed

    def get(self, idx):
        smiles = self.df.loc[idx, 'smiles']
        if len(self.label_col) > 1:
               label = torch.tensor([self.df.loc[idx, col] for col in self.label_col], dtype=torch.float).unsqueeze(0)
        else:
               label = torch.tensor([self.df.loc[idx, col] for col in self.label_col], dtype=torch.float)
        if self.normalize:
            label = (label - torch.tensor(self.mean, dtype=torch.float)) / torch.tensor(self.std, dtype=torch.float)
        data = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(smiles, label)
        # Check if data is None
        if data is not None:
            data.y = label
            data.smiles = smiles  # Add smiles to the data object for transform use
            
            if self.transform:
                data = self.transform(data)
            return data
        
        else:
            return None