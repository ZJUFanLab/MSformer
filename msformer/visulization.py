import seaborn as sns
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdFMCS
from collections import Counter, defaultdict
from scipy.stats import pearsonr

        
def visualize_attention(mol, fragments):
    """Visualize attention weights on molecular structure
    
    Args:
        mol: RDKit molecule object
        fragments: DataFrame containing fragment information with PCA components
    """
    cmap = plt.get_cmap('viridis')
    pc1_values = fragments['pc1']
    fragment_mols = fragments['mol']

    highlight_atoms = []
    highlight_bonds = []
    atom_colors = {}
    atom_radii = {}
    bond_colors = {}

    for idx, frag in enumerate(fragment_mols):
        if not frag:
            continue
            
        mcs_result = rdFMCS.FindMCS([mol, frag])
        mcs_pattern = Chem.MolFromSmarts(mcs_result.smartsString)
        matches = mol.GetSubstructMatches(mcs_pattern)
        matched_atoms = [atom for match in matches for atom in match]
        
        current_pc1 = pc1_values[idx]
        
        # Record atom information
        for atom in matched_atoms:
            highlight_atoms.append((atom, abs(current_pc1)))
            
        # Find relevant bonds
        bond_indices = set()
        for i in range(len(matched_atoms)):
            for j in range(i+1, len(matched_atoms)):
                bond = mol.GetBondBetweenAtoms(matched_atoms[i], matched_atoms[j])
                if bond:
                    bond_indices.add(bond.GetIdx())
        
        for bond_idx in bond_indices:
            highlight_bonds.append((bond_idx, abs(current_pc1)))

    # Process atom data
    atom_df = pd.DataFrame(highlight_atoms, columns=['atom_idx', 'pc1'])
    atom_features = atom_df.groupby('atom_idx').agg({'pc1': 'mean'})
    atom_features['normalized_pc1'] = (atom_features['pc1'] - atom_features['pc1'].min()) / \
                                      (atom_features['pc1'].max() - atom_features['pc1'].min())
    
    # Create atom visualization parameters
    for idx, row in atom_features.iterrows():
        alpha = row['normalized_pc1']
        atom_colors[idx] = cmap(alpha)
        atom_radii[idx] = alpha * 0.3  # Scale radius for visibility

    # Process bond data
    bond_df = pd.DataFrame(highlight_bonds, columns=['bond_idx', 'pc1']) 
    bond_features = bond_df.groupby('bond_idx').agg({'pc1': 'mean'})
    bond_features['normalized_pc1'] = (bond_features['pc1'] - bond_features['pc1'].min()) / \
                                      (bond_features['pc1'].max() - bond_features['pc1'].min())
    
    # Create bond visualization parameters
    for idx, row in bond_features.iterrows():
        alpha = row['normalized_pc1']
        bond_colors[idx] = (0.06, 0.73, 0.51, alpha)  # Teal color with alpha
        
    return list(atom_features.index), atom_colors, atom_radii, list(bond_features.index), bond_colors

def prepare_molecules(molecule_list, fragment_list):
    """Prepare molecules for visualization
    
    Args:
        molecule_list: List of RDKit molecule objects
        fragment_list: List of fragment molecules
        
    Returns:
        Tuple containing visualization parameters
    """
    vis_options = Draw.MolDrawOptions()
    vis_options.atomLabelFontSize = 14
    vis_options.bondLineWidth = 1.5
    vis_options.includeAtomNumbers = True
    
    processed_mols = []
    atom_highlights = []
    highlight_colors = []
    highlight_radii = []
    
    for mol, frag in zip(molecule_list, fragment_list):
        mcs_result = rdFMCS.FindMCS([mol, frag])
        mcs_pattern = Chem.MolFromSmarts(mcs_result.smartsString)
        matches = mol.GetSubstructMatches(mcs_pattern)
        matched_atoms = [atom for match in matches for atom in match]
        
        processed_mols.append(mol)
        atom_highlights.append(matched_atoms)
        highlight_colors.append({atom: (0.06, 0.73, 0.51, 1) for atom in matched_atoms})
        highlight_radii.append({atom: 0.25 for atom in matched_atoms})
        
    return processed_mols, atom_highlights, highlight_colors, highlight_radii

def process_attention_data(fragment_file, attention_weight_file, target_smiles, max_length=1000):
    """Process attention weight data for visualization
    
    Args:
        fragment_file: Path to fragment CSV file
        attention_weight_file: Path to attention weight numpy file
        target_smiles: SMILES string of target molecule
        max_length: Maximum number of fragments to consider
    """
    scaler = StandardScaler()
    pca = PCA(n_components=1)
    
    fragment_data = pd.read_csv(fragment_file)
    target_idx = fragment_data[fragment_data['smiles'] == target_smiles].index[0]
    
    attention_weights = np.load(attention_weight_file)[target_idx]
    scaled_weights = scaler.fit_transform(attention_weights)
    pca_results = pca.fit_transform(scaled_weights)
    
    target_mol = Chem.MolFromSmiles(target_smiles)
    fragment_smiles = fragment_data['frags'][target_idx].split(';')
    fragment_mols = [Chem.MolFromSmiles(s) for s in fragment_smiles]
    
    # Sort by molecular weight
    mol_weights = pd.Series([Descriptors.MolWt(m) if m else 0 for m in fragment_mols])
    sorted_indices = mol_weights.sort_values(ascending=False).index
    
    processed_data = {
        'pc1': pca_results.squeeze()[:len(fragment_mols)],
        'smiles': [fragment_smiles[i] for i in sorted_indices],
        'mol': [fragment_mols[i] for i in sorted_indices],
        'molecular_weight': mol_weights.values[sorted_indices],
        'compound': target_mol,
        'label': fragment_data['labels'][target_idx]
    }
    
    return pd.DataFrame(processed_data).head(max_length)

def generate_visualization(res):
    """Generate comprehensive visualization of attention patterns"""
    res.reset_index(drop=True,inplace=True)
    
    # Visualization data collection
    Atomweights = []
    fig,axes = plt.subplots(2,2,figsize=(18,10))
    axes = axes.flatten()
    for i in tqdm(range(res.shape[0])):
        pc1 = res.iloc[i]['pc1']
        fragment = res.iloc[i]['mol']
        mol = res.iloc[i]['compound']
        mcs_result = rdFMCS.FindMCS([mol, fragment])
        mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
        matches = mol.GetSubstructMatches(mcs_mol)
        matches = [b  for a in matches for b in a]
        symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in matches]
        Atomweights.extend([[a,b,pc1] for a,b in zip(matches,symbols)])
        
    # Create atom analysis dataframe
    Atomweights = pd.DataFrame(Atomweights)
    Atomweights.columns = ['id','symbol','pc1']
    Atomweights['count'] = 1
    counts = Atomweights.groupby('id').agg({'symbol':"max",'pc1':'mean','count':"sum"})
    
    # Generate molecule visualization
    highlightatomlists,highlightAtomColors,highlightAtomRadii,highlightbondlists,highlightBondColors = attention_vis(mol,res)
    d = rdMolDraw2D.MolDraw2DCairo(600, 400)
    do = rdMolDraw2D.MolDrawOptions()
    do.setAtomPalette = {i:(0, 0, 0) for i in range(20)}
    do.bondLineWidth = 4
    do.addAtomIndices = True
    do.minFontSize = 24
    do.maxFontSize = 40
    do.fixedBondLength = 40
    do.annotationFontScale = 1
    do.setSymbolColour((1.0, 1.0, 1.0, 1.0))
    d.SetDrawOptions(do)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=highlightatomlists,
                                       highlightBonds=highlightbondlists,
                                       highlightAtomColors=highlightAtomColors,
                                       highlightBondColors=highlightBondColors,
                                       highlightAtomRadii=highlightAtomRadii)
    
    d.FinishDrawing()
    d.WriteDrawingText('_.png')
    img = plt.imread('_.png')
    
    # Create plots
    counts.index = counts.index.astype(str)
    
    # Atom analysis plots
    sns.boxplot(Atomweights,x='id',y='pc1',hue='symbol',hue_order=['C','O','N','S','P','F','Cl','Br'],ax=axes[3],legend=False)
    x_ticks = axes[3].get_xticks()
    axes[3].set_xticks(x_ticks[::2])
    sns.scatterplot(counts,x='id',y='count',hue='symbol',hue_order=['C','O','N','S','P','F','Cl','Br'],s=50,ax=axes[1],ec='k',legend=False)
    x_ticks = axes[1].get_xticks()  
    axes[1].set_xticks(x_ticks[::2])
    axes[0].imshow(img)
    axes[0].axis('off')

    # Correlation analysis
    r_value,p_value =pearsonr(res['molecular_weight'],res['pc1'])
    print(f'pearson r {r_value}, p value {p_value}')
    sns.scatterplot(res,x='molecular_weight',y='pc1',s=50,ax=axes[2],c='#0f766e')
    axes[0].set_title(f"n meta structures: {res.shape[0]}",size=18)
    return fig