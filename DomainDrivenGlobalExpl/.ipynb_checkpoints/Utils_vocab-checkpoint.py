import numpy as np
from collections import defaultdict
from collections import Counter
from rdkit import Chem
from rBRICS_public import *
from rdkit.Chem import BRICS
import pdb
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
def sanitize_molecule(mol):
    """
    Sanitize the molecule after breaking bonds to fix aromaticity and valence issues.
    """
    try:
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception as e:
        print(f"Sanitization failed: {e}")
        return None
    return mol

def calculate_bond_energies(mol):
    """
    Approximate bond dissociation energies (BDEs) based on bond type and atom environment.
    """
    bond_energies = {}
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        bond_type = bond.GetBondType()
        
        # Approximate BDE based on bond type (values in kcal/mol)
        # These are rough estimates and should be replaced with quantum chemical calculations for accuracy
        bond_energy = 0
        if bond_type == Chem.BondType.SINGLE:
            bond_energy = 90  # Single bond average
        elif bond_type == Chem.BondType.DOUBLE:
            bond_energy = 170  # Double bond average
        elif bond_type == Chem.BondType.TRIPLE:
            bond_energy = 230  # Triple bond average
        
        # Adjust for electronegativity differences (proxy for bond strength)
        bond_energy += abs(begin_atom.GetAtomicNum() - end_atom.GetAtomicNum()) * 5
        
        # Store bond energy
        bond_idx = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        bond_energies[bond_idx] = bond_energy
    return bond_energies

def energy_based_fragmentation(mol, energy_threshold=200):
    """
    Fragment a molecule by breaking bonds with BDE less than the energy threshold.
    """
    
    bond_energies = calculate_bond_energies(mol)
    
    # Select bonds to break based on energy threshold
    bonds_to_break = [bond for bond, energy in bond_energies.items() if energy < energy_threshold]
    
    # Perform fragmentation
    editable_mol = Chem.RWMol(mol)
    for bond in bonds_to_break:
        editable_mol.RemoveBond(*bond)
        
    # Sanitize molecule to fix aromaticity issues
    sanitized_mol = sanitize_molecule(editable_mol)
    if sanitized_mol is None:
        return [mol], 0
    
    
    # Convert to a sanitized molecule and extract fragments
    fragments = []
    for frag in Chem.GetMolFrags(editable_mol, asMols=True):
        Chem.SanitizeMol(frag)  # Ensure molecule consistency
        fragments.append(frag)
    
    return fragments, bond_energies


def get_lookup_tables(training_data, validation_data, test_data, algorithm):
    # Clean datasets
    training_data = remove_bad_mols(training_data)
    validation_data = remove_bad_mols(validation_data)
    test_data = remove_bad_mols(test_data)

    # Initialize motif dictionary and process datasets
    lookup = MotifDictionary()

    process_dataset(training_data + validation_data, lookup,algorithm=algorithm)
    process_dataset(test_data, lookup, is_test=True, algorithm=algorithm)

    # Count motifs and normalize by motif length
    data_lookup = dict(lookup.data)
    test_data_lookup = dict(lookup.test_data)
    
    return data_lookup, test_data_lookup, lookup

def reindex_data(data_lookup, motif_list, value_counts, least_count, is_test=False):
    graph_to_motifs = defaultdict(set)
    
    node_coverages = []

    for graph_str, graph_data in data_lookup.items():
        covered_nodes = 0
        total_nodes = len(graph_data)
        for node_id, motif_str in graph_data.items():
            if is_test:
                try:
                    data_lookup[graph_str][node_id] = (motif_str, motif_list.index(motif_str))
                    graph_to_motifs[graph_str].add(motif_list.index(motif_str))
                    covered_nodes += 1
                except:
                    data_lookup[graph_str][node_id] = (motif_str, None)
            elif value_counts[motif_str] > least_count:
                data_lookup[graph_str][node_id] = (motif_str, motif_list.index(motif_str))
                graph_to_motifs[graph_str].add(motif_list.index(motif_str))
                covered_nodes += 1
            else:
                data_lookup[graph_str][node_id] = (motif_str, None)
        try:
            mol = Chem.MolFromSmiles(graph_str)

            # get feature dimensions
            n_nodes = mol.GetNumAtoms()
            assert total_nodes == n_nodes
            node_coverages.append(covered_nodes / total_nodes)
        except:
            pdb.set_trace()

    return sum(node_coverages) / len(node_coverages) , graph_to_motifs
            
def mol_with_atom_index(mol):
    '''
    Add Atom indices to a Rdkit molecule
    Input: Rdkit molecule object
    '''
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

def get_mol_with_index(smiles, set_atom_index = True):
    '''
    Coverts a Smiles String to a Rdkit Molecule
    Input: Smiles representation of molecule, flag to recreate atom indexs 
    CCCC
    C1C2C3C4
    '''
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    # Check if smiles has index using Rdkit function
    if set_atom_index:
        return mol_with_atom_index(mol)
    else:
        return mol

def remove_bad_mols(dataset):
    indices_to_remove = np.ones(len(dataset), dtype=bool)
    for i,data in enumerate(dataset):
        if data is None: 
            indices_to_remove[i] = False
        elif data.num_nodes == 0:
            print(f"Skipping molecule {data['smiles']} since it "
                      f"resulted in zero atoms")
            indices_to_remove[i] = False

    return dataset[indices_to_remove]

def atom_counts(smiles):
    # Parse the SMILES string to a molecule object
    mol = Chem.MolFromSmiles(smiles)
    
    assert(mol.GetNumAtoms() > 0)
    
    # Extract atoms from the molecule
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    
    # Count occurrences of each atom
    atom_count = Counter(atoms)
    
    # Sort atoms alphabetically and create the result string
    sorted_atoms = sorted(atom_count.items())
    result = ''.join(f"{atom}{count}" for atom, count in sorted_atoms)
    
    return result
    
def canonicalize_fragment(fragment):
    sorted_atoms = sorted(fragment.GetAtoms(), key=lambda atom: atom.GetSymbol())
    return Chem.MolToSmiles(sorted_atoms, isomericSmiles=True)

def process_molecule(smiles_string, original_mol=True):
    """Creates an RDKit molecule and preserves atom indices if not the original molecule."""
    if original_mol:
        molecule = get_mol_with_index(smiles_string)
    else:
        molecule = get_mol_with_index(smiles_string, set_atom_index=False)
    Chem.SanitizeMol(molecule)
    return molecule

def fragment_molecule(molecule, recursive=True):
    """Break the molecule into fragments using BRICS."""
    
    if recursive:
        pbonds = list(FindrBRICSBonds(molecule))
        ppieces3 = BreakrBRICSBonds(molecule, pbonds)
        brics_fragments = Chem.GetMolFrags(ppieces3, asMols=True)
        if brics_fragments is not None:
            fragments = reBRICS(brics_fragments)
        else:
            fragments = brics_fragments
    else:
        pbonds = list(FindrBRICSBonds(molecule))
        ppieces3 = BreakrBRICSBonds(molecule, pbonds)
        # pbonds = list(BRICS.FindBRICSBonds(molecule))
        # ppieces3 = BRICS.BreakBRICSBonds(molecule, pbonds)
        fragments = Chem.GetMolFrags(ppieces3, asMols=True)
    return fragments

def extract_clique_fragments(mol, cliques):

    fragments = []

    # Iterate over each clique

    for clique in cliques:

        # Create a new editable molecule for the fragment

        editable_mol = Chem.RWMol()
 
        # Mapping of original atom indices to new indices in the fragment

        atom_map = {}
 
        # Add atoms from the clique to the new molecule

        for atom_idx in clique:

            atom = mol.GetAtomWithIdx(atom_idx)

            new_idx = editable_mol.AddAtom(atom)

            atom_map[atom_idx] = new_idx  # Map original index to new fragment index
 
        # Add bonds between the atoms in the clique

        added_bonds = set()  # Track bonds that have been added to avoid duplicates

        for atom_idx in clique:

            for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():

                neighbor_idx = neighbor.GetIdx()

                if neighbor_idx in clique:

                    # Create a sorted tuple of atom indices to avoid duplicate bonds

                    bond_tuple = tuple(sorted([atom_idx, neighbor_idx]))

                    if bond_tuple not in added_bonds:

                        # Add the bond between the atoms in the clique

                        bond = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)

                        if bond:

                            editable_mol.AddBond(atom_map[atom_idx], atom_map[neighbor_idx], bond.GetBondType())

                            added_bonds.add(bond_tuple)  # Mark the bond as added
 
        # Sanitize and append the fragment molecule

        fragment = editable_mol.GetMol()

        # Disable Kekulization and sanitize the molecule with Kekulization turned off

        Chem.SanitizeMol(fragment, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
 
        fragments.append(fragment)

    return fragments

def brics_decomp(mol, molecule_smiles):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []
 
    cliques = []
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])
        
    # Step 2: Identify isolated nodes (atoms with no bonds)
    all_nodes = set(range(mol.GetNumAtoms()))
    connected_nodes = set(a for clique in cliques for a in clique)
    isolated_nodes = all_nodes - connected_nodes
    
    # Step 4: Add isolated nodes to standalone cliques
    for isolated_node in isolated_nodes:
        cliques.append([isolated_node])
        
 
 
    res = list(BRICS.FindBRICSBonds(mol))
    if len(res) == 0:
        return [list(range(n_atoms))], []
    else:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])
 
    # break bonds between rings and non-ring atoms
    for c in cliques:
        if len(c) > 1:
            if mol.GetAtomWithIdx(c[0]).IsInRing() and not mol.GetAtomWithIdx(c[1]).IsInRing():
                cliques.remove(c)
                cliques.append([c[1]])
                breaks.append(c)
            if mol.GetAtomWithIdx(c[1]).IsInRing() and not mol.GetAtomWithIdx(c[0]).IsInRing():
                cliques.remove(c)
                cliques.append([c[0]])
                breaks.append(c)
 
    # select atoms at intersections as motif
    for atom in mol.GetAtoms():
        if len(atom.GetNeighbors()) > 2 and not atom.IsInRing():
            cliques.append([atom.GetIdx()])
            for nei in atom.GetNeighbors():
                if [nei.GetIdx(), atom.GetIdx()] in cliques:
                    cliques.remove([nei.GetIdx(), atom.GetIdx()])
                    breaks.append([nei.GetIdx(), atom.GetIdx()])
                elif [atom.GetIdx(), nei.GetIdx()] in cliques:
                    cliques.remove([atom.GetIdx(), nei.GetIdx()])
                    breaks.append([atom.GetIdx(), nei.GetIdx()])
                cliques.append([nei.GetIdx()])
 
    # merge cliques
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0:
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if len(c) > 0]
 
    # edges
    edges = []
    for bond in res:
        for c in range(len(cliques)):
            if bond[0][0] in cliques[c]:
                c1 = c
            if bond[0][1] in cliques[c]:
                c2 = c
        edges.append((c1, c2))
    for bond in breaks:
        for c in range(len(cliques)):
            if bond[0] in cliques[c]:
                c1 = c
            if bond[1] in cliques[c]:
                c2 = c
        edges.append((c1, c2))
 
    return cliques, edges

 

# def handle_fragment(fragment, molecule_smiles, data, lookup, is_test=False, recursive = True):
#     """Handle a fragment by checking if it can be further broken down or adding it to the lookup."""
#     if recursive:
#         fbonds = list(BRICS.FindBRICSBonds(fragment)) # FindrBRICSBonds
#     else:
#         fbonds = list(FindrBRICSBonds(fragment)) # FindrBRICSBonds
    
#     if len(fbonds) == 0:
#         atom_nums = [atom.GetAtomMapNum() if atom.GetAtomicNum() != 0 else None for atom in fragment.GetAtoms()]
#         [a.SetAtomMapNum(0) for a in fragment.GetAtoms()]  # Remove atom map for unique motif
#         fragment_smiles = Chem.MolToSmiles(fragment, isomericSmiles=False, canonical=True)
#         if is_test:
#             lookup.add_entry_test(molecule_smiles, fragment_smiles, atom_nums, data.y.item())
#         else:
#             lookup.add_entry(molecule_smiles, fragment_smiles, atom_nums, data.y.item())
#     else:
#         fragment_smiles = Chem.MolToSmiles(fragment)
#         return fragment_smiles
    
def add_fragment(fragment, molecule_smiles, data, lookup, is_test=False):
    """Handle a fragment by checking if it can be further broken down or adding it to the lookup."""
    atom_nums = [atom.GetAtomMapNum() if atom.GetAtomicNum() != 0 else None for atom in fragment.GetAtoms()]
    for a in fragment.GetAtoms():  # Remove atom map for unique motif
        a.SetAtomMapNum(0)
    fragment_smiles = Chem.MolToSmiles(fragment, isomericSmiles=False, canonical=True)
    # if is_test:
    #     lookup.add_entry_test(molecule_smiles, fragment_smiles, atom_nums, data.y.item())
    # else:
    #     lookup.add_entry(molecule_smiles, fragment_smiles, atom_nums, data.y.item())
    data_label_count = data.y.squeeze().shape[0]
        
    if is_test:
        if data_label_count == 1:
            lookup.add_entry_test(molecule_smiles, fragment_smiles, atom_nums, data.y.item())
        else:
            lookup.add_entry_test(molecule_smiles, fragment_smiles, atom_nums, data.y.tolist())
    else:
        if data_label_count == 1:
            lookup.add_entry(molecule_smiles, fragment_smiles, atom_nums, data.y.item())
        else:
            lookup.add_entry(molecule_smiles, fragment_smiles, atom_nums, data.y.tolist())


def process_dataset(dataset, lookup, is_test=False, algorithm='BRICS'):
    """Processes a dataset (train/test) and fragments each molecule.
    ["RBRICS", "MGSSL", "Energy_100", "Energy_200"]"""
    if algorithm == 'RBRICS':
        for i, data in enumerate(dataset):
            molecule_smiles = data["smiles"]
            molecule = process_molecule(molecule_smiles, True)
            all_fragments = fragment_molecule(molecule, recursive=True)
            for fragment in all_fragments:
                #Todo add min length
                add_fragment(fragment, molecule_smiles, data, lookup, is_test)
#     if algorithm == 'RBRICS':
#         for i, data in enumerate(dataset):
#             molecule_smiles = data["smiles"]
#             to_process = [molecule_smiles]
#             original_mol = True

#             while to_process:
#                 # input(to_process)
#                 smiles_string = to_process.pop()
#                 molecule = process_molecule(smiles_string, original_mol)
#                 original_mol = False

#                 all_fragments = fragment_molecule(molecule)
#                 for fragment in all_fragments:
#                     #Todo add min length
#                     fragment_smiles = handle_fragment(fragment, molecule_smiles, data, lookup, is_test)
#                     if fragment_smiles:
#                         # print("here",fragment_smiles)
#                         to_process.append(fragment_smiles)
    elif algorithm == 'BRICS':
        for i, data in enumerate(dataset):
            molecule_smiles = data["smiles"]
            molecule = process_molecule(molecule_smiles, True)
            all_fragments = fragment_molecule(molecule, recursive=False)
            for fragment in all_fragments:
                #Todo add min length
                add_fragment(fragment, molecule_smiles, data, lookup, is_test)
    elif algorithm =='MGSSL':
        for i, data in enumerate(dataset):
            molecule_smiles = data["smiles"]
            
            molecule = process_molecule(molecule_smiles, original_mol=True)
            mol_mgssl = get_mol(molecule_smiles)
            cliques, edges = brics_decomp(mol_mgssl, molecule_smiles)
            for i,c in enumerate(cliques):
                cmol = get_clique_mol(mol_mgssl, c)
                fragment_smiles = get_smiles(cmol)
                data_label_count = data.y.squeeze().shape[0]
                if is_test:
                    if data_label_count == 1:
                        lookup.add_entry_test(molecule_smiles, fragment_smiles, c, data.y.item())
                    else:
                        lookup.add_entry_test(molecule_smiles, fragment_smiles, c, data.y.tolist())
                else:
                    if data_label_count == 1:
                        lookup.add_entry(molecule_smiles, fragment_smiles, c, data.y.item())
                    else:
                        lookup.add_entry(molecule_smiles, fragment_smiles, c, data.y.tolist())
            # pbonds = list(FindrBRICSBonds(molecule))
            # ppieces3 = BreakrBRICSBonds(molecule, pbonds)
            # clique = Chem.GetMolFrags(ppieces3, asMols=False)
            # all_fragments = extract_clique_fragments(molecule,clique)
            # for fragment in all_fragments:
            #     #Todo add min length
            #     add_fragment(fragment, molecule_smiles, data, lookup, is_test)
            #     # fragment_smiles = handle_fragment(fragment, molecule_smiles, data, lookup, is_test)
    else:
        parts = algorithm.split("_")
        if parts[0] == "Energy":
            for i, data in enumerate(dataset):
                molecule_smiles = data["smiles"]
                molecule = process_molecule(molecule_smiles, original_mol=True)
                all_fragments = energy_based_fragmentation(molecule, energy_threshold = int(parts[1]))[0]
                for fragment in all_fragments:
                    #Todo add min length
                    add_fragment(fragment, molecule_smiles, data, lookup, is_test)
                    # fragment_smiles = handle_fragment(fragment, molecule_smiles, data, lookup, is_test)
        else:
            raise Exception(f'Incorrect Algorithm {algorithm}')
            
def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol

def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol


def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol


def get_clique_mol(mol, atoms):
    # get the fragment of clique
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol)  # We assume this is not None
    return new_mol
                    
class MotifDictionary:
    def __init__(self):
        self.data = defaultdict(lambda: defaultdict())
        self.test_data = defaultdict(lambda: defaultdict())
        self.motifs_length = defaultdict()
        self.motifs_class = defaultdict(dict)
        self.test_motifs_length = defaultdict()

    def add_entry(self, graph_str, motif_str, nodes, class_id):
        '''
        Maps a Graph String and nodes within the graph to a Motif String
        '''
        # motif_str = atom_counts(motif_str)
        for element in nodes:
            if element is not None:
                self.data[graph_str][element]= motif_str 
        self.motifs_class[motif_str][graph_str] = class_id
        self.motifs_length[motif_str]= len(nodes) - nodes.count(None)
        
    def add_entry_test(self, graph_str, motif_str, nodes, class_id):
        '''
        Maps a Graph String and nodes within the graph to a Motif String
        '''
        # motif_str = atom_counts(motif_str)
        for element in nodes:
            if element is not None:
                
                self.test_data[graph_str][element]= motif_str 
                
        self.test_motifs_length[motif_str]= len(nodes) - nodes.count(None)

    def query_by_graph(self, graph_str):
        '''
        Returns Nodes to Motif_String map
        '''
        return self.data.get(graph_str, {})
    
    def query_by_test_graph(self, graph_str):
        '''
        Returns Nodes to Motif_String map
        '''
        return self.test_data.get(graph_str, {})
    
    def remove_motifs(self, list_of_motifs_to_remove):
        '''
        Removes less frequent motifs
        '''
        for key in list_of_motifs_to_remove:
            self.motifs_length.pop(key)
            self.motifs_class.pop(key)

    def get_all_unique_motif(self):
        '''
        All unique motifs
        '''
        return list(self.motifs_length.keys())
    
    def get_motif_lengths(self):
        return self.motifs_length
    
    def get_test_motif_lengths(self):
        return self.test_motifs_length