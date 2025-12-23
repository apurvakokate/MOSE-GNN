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
# pip install rdkit-pypi
from rdkit import Chem
from rdkit.Chem import rdchem

# ----------------------------
# (A) Motif finders
# ----------------------------

def find_carbonyl_motifs(mol):
    """Return list of carbonyl motifs as lists of atom indices [carbon_idx, oxygen_idx]."""
    motifs = []
    for b in mol.GetBonds():
        if b.GetBondType() == rdchem.BondType.DOUBLE:
            a, c = b.GetBeginAtom(), b.GetEndAtom()
            if {a.GetAtomicNum(), c.GetAtomicNum()} == {6, 8}:
                c_idx, o_idx = (a.GetIdx(), c.GetIdx()) if a.GetAtomicNum() == 6 else (c.GetIdx(), a.GetIdx())
                motifs.append([c_idx, o_idx])
    return motifs


def _eligible_carbon(a, *, exclude_rings=True, aliphatic_only=True):
    # Eligibility for being part of an alkyl chain graph (adj build).
    if a.GetAtomicNum() != 6: return False
    if aliphatic_only and a.GetIsAromatic(): return False
    if exclude_rings and a.IsInRing(): return False
    return True  # sp3 enforced only for INTERIORS


def _carbon_single_neighbors(mol, idx):
    out = []
    a = mol.GetAtomWithIdx(idx)
    for nb in a.GetNeighbors():
        b = mol.GetBondBetweenAtoms(idx, nb.GetIdx())
        if nb.GetAtomicNum() == 6 and b.GetBondType() == rdchem.BondType.SINGLE:
            out.append(nb.GetIdx())
    return out


def find_unbranched_alkane_chains(
    mol,
    min_len=3,
    exclude_rings=True,
    aliphatic_only=True,
    allow_sp2_endpoints=True,
):
    """
    Linear, unbranched C–C SINGLE-bond chain of length >= min_len.
    Interiors: carbon-only, sp3, non-aromatic, non-ring, heavy-degree==2, neighbors are exactly two chain carbons.
    Endpoints: carbon; may be sp2 if allow_sp2_endpoints=True (helps avoid false negatives).
    Returns list of chains as ordered lists of atom indices.
    """
    n = mol.GetNumAtoms()
    eligible = [_eligible_carbon(mol.GetAtomWithIdx(i), exclude_rings=exclude_rings, aliphatic_only=aliphatic_only)
                for i in range(n)]

    adj = [[] for _ in range(n)]
    for b in mol.GetBonds():
        if b.GetBondType() != rdchem.BondType.SINGLE:
            continue
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if eligible[u] and eligible[v]:
            adj[u].append(v); adj[v].append(u)

    def valid_segment(seg):
        L = len(seg)
        if L < min_len: return False
        for k, idx in enumerate(seg):
            a = mol.GetAtomWithIdx(idx)
            c_single = _carbon_single_neighbors(mol, idx)
            deg_heavy = a.GetDegree()
            if 0 < k < L-1:
                if a.GetHybridization() != rdchem.HybridizationType.SP3: return False
                if deg_heavy != 2: return False
                if len(c_single) != 2 or set(c_single) != {seg[k-1], seg[k+1]}: return False
            else:
                # endpoint: exactly one chain neighbor in seg
                if len([j for j in adj[idx] if j in seg]) != 1: return False
                if not allow_sp2_endpoints and a.GetHybridization() != rdchem.HybridizationType.SP3:
                    return False
        return True

    visited = set()
    chains = []

    # grow from ends (eligible degree != 2)
    for i in range(n):
        if not eligible[i]: continue
        if len(adj[i]) != 2:
            for nb in adj[i]:
                e = tuple(sorted((i, nb)))
                if e in visited: continue
                seg = [i, nb]; visited.add(e)
                prev, cur = i, nb
                while len(adj[cur]) == 2:
                    nxt = adj[cur][0] if adj[cur][1] == prev else adj[cur][1]
                    e2 = tuple(sorted((cur, nxt)))
                    if e2 in visited: break
                    seg.append(nxt); visited.add(e2)
                    prev, cur = cur, nxt
                if valid_segment(seg):
                    chains.append(seg)

    # components where every eligible node has degree==2 (non-ring corridor)
    seen = set()
    for i in range(n):
        if not eligible[i] or i in seen: continue
        stack = [i]; comp = []
        while stack:
            u = stack.pop()
            if u in seen: continue
            seen.add(u); comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    stack.append(v)
        if comp and all(len(adj[u]) == 2 for u in comp):
            # build a path order
            path = [comp[0]]
            while len(adj[path[-1]]) == 2 and (len(path) == 1 or adj[path[-1]][0] != path[-2]):
                nxt = adj[path[-1]][0] if len(path) == 1 or adj[path[-1]][0] != path[-2] else adj[path[-1]][1]
                if nxt in path: break
                path.append(nxt)
            path = path[::-1]
            while len(adj[path[-1]]) == 2 and (len(path) == 1 or adj[path[-1]][0] != path[-2]):
                nxt = adj[path[-1]][0] if len(path) == 1 or adj[path[-1]][0] != path[-2] else adj[path[-1]][1]
                if nxt in path: break
                path.append(nxt)
            if valid_segment(path):
                chains.append(path)

    return chains


# ----------------------------
# (B) Helper utilities
# ----------------------------

def _bond_idx(mol, a, b):
    bnd = mol.GetBondBetweenAtoms(a, b)
    return None if bnd is None else bnd.GetIdx()

def _protect_intra_motif_bonds(mol, atom_groups):
    """Set of bond indices that must NOT be cut (intra-motif)."""
    protect = set()
    owner = [-1] * mol.GetNumAtoms()
    for gid, atoms in enumerate(atom_groups):
        for i in atoms:
            owner[i] = gid
    for a in range(mol.GetNumAtoms()):
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            b = nb.GetIdx()
            if a < b and owner[a] != -1 and owner[a] == owner[b]:
                bi = _bond_idx(mol, a, b)
                if bi is not None: protect.add(bi)
    return protect

def _boundary_bonds(mol, atoms):
    """Bond indices from motif atoms to outside (to be cut to isolate motif)."""
    atoms_set = set(atoms)
    cuts = set()
    for a in atoms:
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            b = nb.GetIdx()
            if b not in atoms_set:
                bi = _bond_idx(mol, a, b)
                if bi is not None:
                    cuts.add(bi)
    return cuts
from rdkit import Chem
from rdkit.Chem import rdchem

def safe_sanitize_fragment(frag: Chem.Mol) -> Chem.Mol:
    """
    Sanitize a fragment defensively:
      1) try full sanitize;
      2) on Kekulize/Aromaticity issues, clear aromatic flags & re-sanitize without KEKULIZE,
         then set aromaticity explicitly.
    """
    frag = Chem.Mol(frag)  # copy
    frag.UpdatePropertyCache(strict=False)
    try:
        Chem.SanitizeMol(frag)
        return frag
    except Exception:
        # Clear existing aromatic flags (puts explicit bonds), then re-run a reduced sanitize
        try:
            Chem.Kekulize(frag, clearAromaticFlags=True)
        except Exception:
            pass  # if this fails, we'll still try reduced sanitize below

        # Run sanitize but SKIP KEKULIZE step to avoid the same failure
        ops = (Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        Chem.SanitizeMol(frag, sanitizeOps=ops)

        # Recompute aromaticity (without enforcing kekulized form)
        Chem.SetAromaticity(frag)
        return frag

def safe_split_fragments(mol_after: Chem.Mol) -> list[Chem.Mol]:
    """
    Split molecule into fragments WITHOUT sanitizing at split time,
    then sanitize each fragment safely.
    """
    raw_frags = Chem.GetMolFrags(mol_after, asMols=True, sanitizeFrags=False)
    out = []
    for f in raw_frags:
        out.append(safe_sanitize_fragment(f))
    return out

# ----------------------------
# (C) Main pipeline: Preserve motifs, then BRICS on the rest
# ----------------------------

def fragment_preserve_motifs_then_BRICS(
    mol_or_smiles,
    motifs=("alkane", "carbonyl"),
    alkane_min_len=3,
    prefer_order=("alkane", "carbonyl"),
    allow_sp2_endpoints=True,
):
    """
    1) Identify motifs (alkane chains, carbonyls).
    2) Cut ONLY motif boundary bonds so motifs are standalone fragments.
    3) On the remaining (non-motif) fragments, run BRICS fragmentation (rBRICS if available; else RDKit BRICS).
    Returns list of fragment RDKit Mol objects.
    """
    mol = Chem.MolFromSmiles(mol_or_smiles) if isinstance(mol_or_smiles, str) else Chem.Mol(mol_or_smiles)

    # --- 1) find motifs ---
    pending = []
    if "carbonyl" in motifs:
        for atoms in find_carbonyl_motifs(mol):
            pending.append(("carbonyl", atoms))
    if "alkane" in motifs:
        for chain in find_unbranched_alkane_chains(
            mol, min_len=alkane_min_len, allow_sp2_endpoints=allow_sp2_endpoints
        ):
            pending.append(("alkane", chain))

    # Resolve overlaps by priority
    prio = {name: i for i, name in enumerate(prefer_order)}
    owned = [None] * mol.GetNumAtoms()
    groups = []  # (name, atoms)
    for name, atoms in sorted(pending, key=lambda x: prio.get(x[0], 999)):
        kept = [a for a in atoms if owned[a] is None]
        if kept:
            for a in kept: owned[a] = name
            groups.append((name, kept))

    motif_atoms = [atoms for _, atoms in groups]

    # --- 2) compute protected & boundary bonds; cut boundaries to isolate motifs ---
    protected = _protect_intra_motif_bonds(mol, motif_atoms)
    cut = set()
    for _, atoms in groups:
        cut |= _boundary_bonds(mol, atoms)
    cut -= protected  # safety

    # First cut: isolate motifs
    if cut:
        mol_after = Chem.FragmentOnBonds(mol, sorted(cut), addDummies=True)
    else:
        mol_after = Chem.Mol(mol)

    # Split into fragments
    frags = safe_split_fragments(mol_after)

    # Helper to re-check motifs in a fragment (labeling)
    def _frag_has(m, kind):
        if kind == "carbonyl":
            return len(find_carbonyl_motifs(m)) > 0
        if kind == "alkane":
            return len(find_unbranched_alkane_chains(m, min_len=alkane_min_len,
                                                    allow_sp2_endpoints=allow_sp2_endpoints)) > 0
        return False

    # --- 3) BRICS on non-motif fragments only ---
    final_frags = []
    # Try rBRICS first (your custom module)
    rbrics_ok = False
    try:
        # expects functions with these names in your file
        # from rBRICS_public import FindreBRICSBonds, BreakrBRICSBonds, reBRICS
        rbrics_ok = True
    except Exception:
        rbrics_ok = False

    for f in frags:
        has_alk = _frag_has(f, "alkane") if "alkane" in motifs else False
        has_carb = _frag_has(f, "carbonyl") if "carbonyl" in motifs else False
        if has_alk or has_carb:
            # preserve motif fragments as is
            final_frags.append(f)
            continue

        if rbrics_ok:
            # Use your rBRICS flow on this fragment
            try:
                # Find & break BRICS bonds (respecting rBRICS’ logic)
                pbonds = list(FindreBRICSBonds(f))
                broken = BreakrBRICSBonds(f, pbonds)
                # rBRICS post-processing (if needed)
                sub_frags = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True)
                # Optional: apply reBRICS if your pipeline expects a second pass
                try:
                    sub_frags = reBRICS(sub_frags)
                except Exception:
                    pass
                final_frags.extend(sub_frags if isinstance(sub_frags, (list, tuple)) else [sub_frags])
                continue
            except Exception:
                # Fall back to RDKit BRICS if rBRICS failed on this fragment
                pass

        # RDKit BRICS fallback
        try:
            from rdkit.Chem import BRICS
            br_bonds = BRICS.FindBRICSBonds(f)  # [ ((a,b),(l1,l2)), ... ]
            if br_bonds:
                # Convert to bond indices for FragmentOnBonds OR use BreakBRICSBonds
                # Using BreakBRICSBonds keeps BRICS attachment labels.
                f2 = BRICS.BreakBRICSBonds(f, [(a, b) for (a, b), _ in br_bonds])
                sub_frags = Chem.GetMolFrags(f2, asMols=True, sanitizeFrags=True)
                final_frags.extend(sub_frags)
            else:
                final_frags.append(f)
        except Exception:
            # If BRICS unavailable, keep the fragment as-is
            final_frags.append(f)

    return final_frags




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

    # Save motifs to retrieve original list after filtering
    lookup.save_motifs()

    
    data_lookup = dict(lookup.data)
    test_data_lookup = dict(lookup.test_data)
    
    return data_lookup, test_data_lookup, lookup

def reindex_data(data_lookup, motif_list, value_counts, least_count, is_test=False):
    graph_to_motifs = defaultdict(set)

    for graph_str, graph_data in data_lookup.items():
        total_nodes = len(graph_data)
        for node_id, motif_str in graph_data.items():
            if is_test:
                try:
                    data_lookup[graph_str][node_id] = (motif_str, motif_list.index(motif_str))
                    graph_to_motifs[graph_str].add(motif_list.index(motif_str))
                except:
                    data_lookup[graph_str][node_id] = (motif_str, None)
            elif value_counts[motif_str] > least_count:
                data_lookup[graph_str][node_id] = (motif_str, motif_list.index(motif_str))
                graph_to_motifs[graph_str].add(motif_list.index(motif_str))
            else:
                data_lookup[graph_str][node_id] = (motif_str, None)
        
    return graph_to_motifs

def compute_node_coverage(data_lookup):
    """
    Compute the number of nodes across all graphs that have a valid motif index.
    Returns: total_covered_nodes (int)
    """
    node_coverages = []
    for node2motif in data_lookup.values():
        covered = 0
        total = 0
        for motif_str, idx in node2motif.values():
            total += 1
            if idx is not None:
                covered += 1
        node_coverages.append(covered / total)
    return sum(node_coverages) / len(node_coverages)
    

def reindex_data_lookup_by_class(data_lookup, motif_list, value_counts, class_id, cutoff, is_test=False):
    """
    Reindex data_lookup filtering motifs by class-specific counts.
    `value_counts` is a dict: motif_str -> {0: count0, 1: count1}
    Returns: graph_to_motifs (dict: graph_str -> set of motif indices)
    """
    motif_to_idx = {m: i for i, m in enumerate(motif_list)}
    graph_to_motifs = defaultdict(set)

    for graph_str, node2motif in data_lookup.items():
        for node_id, motif_str in node2motif.items():
            idx = motif_to_idx.get(motif_str)
            if data_lookup[graph_str][node_id][1] is not None:
                #Check if minority threshold applies
                if is_test:
                    data_lookup[graph_str][node_id] = (motif_str, idx)
                    if idx is not None:
                        graph_to_motifs[graph_str].add(idx)
                else:
                    counts_by_label = value_counts.get(motif_str, {})
                    cnt = counts_by_label.get(class_id, 0)
                    if cnt >= cutoff:
                        data_lookup[graph_str][node_id] = (motif_str, idx)
                        graph_to_motifs[graph_str].add(idx)
                    else:
                        data_lookup[graph_str][node_id] = (motif_str, None)
    return graph_to_motifs

def simplified_reindex_data(data_lookup, motif_list):
    '''
    Take an already-filtered motif_list, assign each motif an index,
    and rewrite data_lookup[node] = (motif, idx_or_None).
    Returns graph_to_motifs: graph_str -> set(idx).
    '''
    motif_to_idx = {m:i for i,m in enumerate(motif_list)}
    graph_to_motifs = defaultdict(set)

    for g, node2motif in data_lookup.items():
        for nid, val in node2motif.items():
            motif_str = val if isinstance(val, str) else val[0]
            idx = motif_to_idx.get(motif_str)  # None if not in motif_list
            data_lookup[g][nid] = (motif_str, idx)
            if idx is not None:
                graph_to_motifs[g].add(idx)

    return graph_to_motifs
            
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
    mol = sanitize(mol)
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
        pbonds = list(FindreBRICSBonds(molecule))
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

def fragment_molecule_with_bond_info(molecule, recursive=True):
    """Break the molecule into fragments using BRICS and return fragments + bond info."""
    
    if recursive:
        pbonds = list(FindreBRICSBonds(molecule))
        ppieces3 = BreakrBRICSBonds(molecule, pbonds)
        brics_fragments = Chem.GetMolFrags(ppieces3, asMols=True, sanitizeFrags=True, fragsMolAtomMapping=True)
        if brics_fragments is not None:
            fragments, atom_lists = zip(*brics_fragments)  # (Mol, atom_indices)
            bond_lists = [get_fragment_bonds(molecule, atom_idx) for atom_idx in atom_lists]
        else:
            fragments, atom_lists, bond_lists = [], [], []
    else:
        pbonds = list(FindreBRICSBonds(molecule))
        ppieces3 = BreakrBRICSBonds(molecule, pbonds)
        fragments, atom_lists = Chem.GetMolFrags(ppieces3, asMols=True, sanitizeFrags=True, fragsMolAtomMapping=True)
        bond_lists = [get_fragment_bonds(molecule, atom_idx) for atom_idx in atom_lists]
    
    return fragments, atom_lists, bond_lists

def get_fragment_bonds(original_mol, atom_indices):
    """Return bond indices from original molecule for a given fragment atom list."""
    bond_indices = []
    for bond in original_mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in atom_indices and a2 in atom_indices:
            bond_indices.append(bond.GetIdx())
    return bond_indices


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
    try:
        data_label_count = data.y.squeeze().shape[0]
    except IndexError:
        data_label_count = data.y.shape[0]
        
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

            
def process_dataset_with_bond_info(dataset, lookup, is_test=False, algorithm='BRICS'):
    """Processes a dataset (train/test) and fragments each molecule.
    ["RBRICS", "MGSSL", "Energy_100", "Energy_200"]"""
    if algorithm == 'RBRICS':
        for i, data in enumerate(dataset):
            molecule_smiles = data["smiles"]
            molecule = process_molecule(molecule_smiles, True)
            all_fragments, atom_list, bond_list = fragment_molecule_with_bond_info(molecule, recursive=True)
            for fragment in all_fragments:
                #Todo add min length
                add_fragment(fragment, molecule_smiles, data, lookup, is_test)
    else:
        raise Exception(f'Incorrect Algorithm {algorithm}')
    

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
    elif algorithm == 'BRICS':
        for i, data in enumerate(dataset):
            molecule_smiles = data["smiles"]
            molecule = process_molecule(molecule_smiles, True)
            all_fragments = fragment_molecule(molecule, recursive=False)
            for fragment in all_fragments:
                #Todo add min length
                add_fragment(fragment, molecule_smiles, data, lookup, is_test)
    elif algorithm == 'PRESERVE_ALKANE_CARBONYL':
        for i, data in enumerate(dataset):
            molecule_smiles = data["smiles"]
            molecule = process_molecule(molecule_smiles, True)
            all_fragments = fragment_preserve_motifs_then_BRICS(molecule)
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
                try:
                    data_label_count = data.y.squeeze().shape[0]
                except IndexError:
                    data_label_count = data.y.shape[0]
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
        self.test_motifs_class = defaultdict(dict)

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
        self.test_motifs_class[motif_str][graph_str] = class_id

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
    
    def save_motifs(self):
        # --- new: keep a pristine backup of the full vocab ---
        self._backup_motifs_length = self.motifs_length.copy()
        self._backup_motifs_class  = self.motifs_class.copy()
        self._test_backup_motifs_length = self.test_motifs_length.copy()
        self._test_backup_motifs_class = self.test_motifs_class.copy()
    
    def remove_motifs(self, list_of_motifs_to_remove):
        '''
        Removes less frequent motifs
        '''
        for key in list_of_motifs_to_remove:
            self.motifs_length.pop(key)
            self.motifs_class.pop(key)
            
    def readd_motifs(self, list_of_motifs_to_add):
        """
        Restores motifs (and their metadata) that were previously removed,
        pulling from the backups.
        """
        for motif in list_of_motifs_to_add:
            # only re-add if it was in the original backup
            if motif in self._backup_motifs_length:
                self.motifs_length[motif] = self._backup_motifs_length[motif]
                self.motifs_class[motif]  = self._backup_motifs_class[motif]

    def get_all_unique_motif(self):
        '''
        All unique motifs
        '''
        return list(self.motifs_length.keys())

    def get_all_possible_motif_without_filter(self):
        '''
        Unfiltered Motifs
        '''
        return self._backup_motifs_length, self._backup_motifs_class, self._test_backup_motifs_length, self._test_backup_motifs_class
    
    def get_motif_lengths(self):
        return self.motifs_length
    
    def get_test_motif_lengths(self):
        return self.test_motifs_length