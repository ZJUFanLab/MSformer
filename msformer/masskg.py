from rdkit import Chem
from rdkit.Chem import AllChem,rdMolDescriptors
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs
from rdkit.Chem.AtomPairs import Pairs
import numpy as np
import pandas as pd
import re
import itertools


def draw_molb(mol,got_ms2,savepath):
    """
    Parameters
    ----------
    gms : TYPE
        DESCRIPTION.
    mollist : TYPE
        DESCRIPTION.
    savepath : TYPE
        不能有中文路径！！！.
    mode : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    got_ms2['bonds'] = got_ms2['bonds'].astype(str)
    for i,bd in enumerate(got_ms2['bonds']):
        Chem.Kekulize(mol,clearAromaticFlags=True)
        d = rdMolDraw2D.MolDraw2DCairo(500, 400)
        hit_bonds = eval(bd)
        bond_cols = {}
        for j, bd in enumerate(hit_bonds):
            bond_cols[bd] = tuple(np.random.rand(3))
        rdMolDraw2D.PrepareAndDrawMolecule(d,mol,
                                           highlightBonds=hit_bonds,
                                           highlightBondColors=bond_cols)
        d.DrawMolecule(mol)
        d.FinishDrawing()
        d.WriteDrawingText(os.path.join(savepath,'test.png'))       

def get_bridge_bonds(bonds_in_r):
    bb = []
    for i in range(len(bonds_in_r)):
        bs1 = bonds_in_r[i]
        bs2 = [b for j in range(len(bonds_in_r)) if i!=j for b in bonds_in_r[j]]
        for b in bs1:
            if b in bs2:
                bb.append(b)
    bb = set(bb)
    return bb

def bondscomb2(bonds1,bonds2):
    b2 = []
    if bonds1 == bonds2:
        for i in range(len(bonds1)):
            for j in range(i+1,len(bonds1)):
                if not type(bonds1[i]) == list:
                    bonds1[i] = [bonds1[i]]
                if not type(bonds1[j]) == list:
                    bonds1[j] = [bonds1[j]]
                b2.append(bonds1[i]+bonds1[j])
    else:
        for i in bonds1:
            if not type(i) == list:
                i = [i]
            elif len(i) == 0:
                continue
            for j in bonds2:
                if not type(j) == list:
                    j = [j]
                elif len(j) == 0:
                    continue
                if not i == j:
                    b2.append(i+j)
    return b2

def enu_com(l):
    l = [i+999 for i in l]
    k = len(l)
    choices = [list(np.binary_repr(i,k)) for i in range(1,2**k)]
    combinations = [np.array([eval(i) for i in a])*np.array(l) for a in choices]
    combinations = [[i for i in a if i!=0] for a in combinations]
    combinations = [[i-999 for i in a] for a in combinations]
    combinations = [[int(i) for i in c] for c in combinations]
    return combinations
def pop_list(l):
    out = []
    for x in l :
        
        if  len(out)<1:
            out.append(x)
        else:
            if not x in out:
                out.append(x)
    return out


def get_fragments(mol,bonds_comb,adduct,mode):
    H = 1.007825
    Na = 22.98977
    K = 38.963707
    Cl = 34.968853
    if 'Na' in adduct:
        adct = Na
    elif 'K' in adduct:
        adct = K
    elif 'Cl' in adduct:
        adct = -1*Cl
    else:
        adct = H
    fragments = []
    frag_weights = []
    all_bonds = []
    symb_val_rank = {'C':0,'O':2,'N':1,'P':0,'S':0,'na':0}
    for bd in bonds_comb:
        if not type(bd) == list:
            bd = [bd]
        elif not len(bd) > 0:
            continue
        bd = list(set(bd))
        bateat = [(mol.GetBondWithIdx(b).GetBeginAtom(),mol.GetBondWithIdx(b).GetEndAtom()) for b in bd]
        atoms_idx = [[a.GetIdx()for a in be] for be in bateat]
        atoms_symb = [[a.GetSymbol()for a in be] for be in bateat]
        atoms_symb = [[a if a in symb_val_rank.keys() else 'na' for a in be ] for be in atoms_symb]
        atoms_ms = [[a.GetMass()for a in be] for be in bateat]
        atoms_val = [[symb_val_rank[be[0]]-symb_val_rank[be[1]],symb_val_rank[be[1]]-symb_val_rank[be[0]]]for be in atoms_symb]
        # atoms_val = [[be[0]-be[1],be[1]-be[0]]for be in atoms_ms]
        val_dict = {a:b for aa,bb in zip(atoms_idx,atoms_val) for a,b in zip(aa,bb) }
        fmol = Chem.FragmentOnBonds(mol,bd) #断裂键
        try:
            frag_mols = Chem.GetMolFrags(fmol,asMols=True) #提取碎片分子(剔除键后) 
        except Exception as e:
            print(e)
            continue
        frag_smarts = [Chem.MolToSmarts(f) for f in frag_mols]
        
        if mode == '-':
            n_charges = [rdMolDescriptors.CalcNumHeteroatoms(mol) for f in frag_mols]
        else:
            n_charges = [1 for f in frag_mols]
        n_Hs = [sum([a.GetNumImplicitHs() for a in f.GetAtoms()]) for f in frag_mols] 
        n_ids = [re.findall("\d*#0",s) for s in frag_smarts] # 找到所有与断键连结的atom
        n_ids = [["0#0"if _=="#0" else _ for _ in n] for n in n_ids] # 似乎0号键不会被记录,这里补上
        n_ids = [[eval(s.replace('#0','')) for s in n] for n in n_ids]
       # n_ids = [n if n else [0] for n in n_ids] # 似乎0号键不会被记录,这里补上
        # 注意，这里是反的，O-连的是C所以O反而会是-1，所以乘以-1改回来
        n_vals = [[-1*val_dict[s] for s in n] for n in n_ids]
        n_breaks = [len(re.findall("-.\d*#0",s))+ 2*len(re.findall("=.\d*#0",s)) for s in frag_smarts]
        # n_breaks = [min(a,b) for a,b in zip(n_Hs,n_breaks)] # 取自由基与活泼H的最小值
        n_atoms = [i.GetNumAtoms() for i in frag_mols]
        # 加减H规则，不同环境可能的方法不同，但要求所有碎片总共加减H为0
        fw = []
        ff = []
        ab = []
        for i in range(len(frag_mols)):
            if n_charges[i] == 0: #判断是否带电
                continue
            
            a = n_breaks[i]
            vals = n_vals[i]
            vals = [int(v/abs(v)) if v!=0 else int(v) for v in vals]
            if n_atoms[i] > 2: 
                if n_Hs[i] > 0: 
                    b = [0] + [-1*(j+1) for j in range(a)] + [(j+1) for j in range(a)]
                else:
                    b = [0] + [(j+1) for j in range(a)] #对于缺少H的基团只能得到H而无法给出
         #       if min(n_atoms)>2: 
                for v in vals:
                    if v == -1:
                        b.remove(max(b))
                    elif v == 1:
                        b.remove(min(b))
            else:
                # 小分子特权,只拿不给，如CH3 HO NH2等
                b = [0] + [(j+1) for j in range(a)]
            ab.append(b)
        if len(ab) == 2:
            ab_ = itertools.product(ab[0],ab[1])
        elif len(ab) == 3:
            ab_ = itertools.product(ab[0],ab[1],ab[2])
        elif len(ab) == 4:
            ab_ = itertools.product(ab[0],ab[1],ab[2],ab[3])
        else:
            continue
        # ab_是不同碎片间所有可能的组合
        for a_b_ in ab_:
            if not sum(a_b_) == 0 :
                continue
            fw_ = [rdMolDescriptors.CalcExactMolWt(frag_mols[i])+a_b_[i]*H for i in range(len(a_b_))]
            # ff_ = [(frag_mols[i],'{}H'.format(i)) for i in range(len(a_b_))] # 暂时不返回碎片
            ff_ = [frag_mols[i]for i in range(len(a_b_))]
            ff_ = [Chem.MolToSmarts(f) for f in ff_]
            ff_ = [re.sub("..\d*#0.","",s) for s in ff_]  # remove cleaved bonds
            if not max(fw_)>=50:
                continue
            fw.extend(fw_)
            ff.extend(ff_)
        fragments.append(ff)
        frag_weights.append(fw)
        all_bonds.append(bd)
    frag_weights.append([rdMolDescriptors.CalcExactMolWt(mol)]) #母离子加进去)
    fragments.append([Chem.MolToSmarts(mol)])
    all_bonds.append([])
    if mode == '-':
        frag_weights = [[f - adct for f in fw] for fw in frag_weights]
    elif mode == '+':
        frag_weights = [[f + adct for f in fw] for fw in frag_weights]
    return fragments,frag_weights,all_bonds

def break_all2(mol,mode='-',adduct=''):
    fragments = []
    frag_weigths = []
    all_bonds = []
    ri = mol.GetRingInfo() # ring information
    bonds_in_r = ri.BondRings()
    bridge_bonds = get_bridge_bonds(bonds_in_r) #获取桥键，不断裂
    bonds_in_r = [[b_ for b_ in b if b_ not in bridge_bonds] for b in bonds_in_r]
    
    chain_bonds = [b.GetIdx() for b in mol.GetBonds() if not b.IsInRing() and b.GetBondTypeAsDouble()<=2]
    ring_bonds = [[[xza[i],xza[j]] for i in range(len(xza)) for j in range(i,len(xza)) if i!=j] for xza in bonds_in_r] # 换上的键需要成组断
    
    chain_comb = bondscomb2(chain_bonds,chain_bonds) # 单键+单键组合
    ring_comb = [bondscomb2(ring_bonds[i],ring_bonds[j]) for i in range(len(ring_bonds)) for j in range(i,len(ring_bonds)) if i!=j] # 两个环的组合
    ri_ch_comb = [bondscomb2(chain_bonds,i) for i in ring_bonds] # 生成单键+环2键组合 
    
    
    # 整理键组
    ring_bonds = [b for bs in ring_bonds for b in bs] 
    ring_comb = [b for bs in ring_comb for b in bs] 
    ri_ch_comb = [b for bs in ri_ch_comb for b in bs]
    bonds_comb = chain_bonds + ring_bonds + chain_comb + ring_comb + ri_ch_comb
    if not len(bonds_comb) > 0:
        return fragments,frag_weigths,all_bonds
    bonds_comb = pop_list(bonds_comb)
    fragments,frag_weigths,all_bonds = get_fragments(mol,bonds_comb,adduct=adduct,mode=mode)
    return fragments,frag_weigths,all_bonds




def insiloscore2(ms_table,refms2,mserror=20e-3):
    refms2 = refms2.copy()
    refms2['intensity'] = refms2['intensity']/refms2['intensity'].sum()
    if len(ms_table.shape) == 1:
        mslist = ms_table
    else:    
        mslist = ms_table['mz']
    if not 'bonds' in list(ms_table.keys()):
        bslist = [0]*len(mslist)
    else:
        bslist = ms_table['bonds']
    if not 'smarts' in list(ms_table.keys()):
        smarts = ['']*len(mslist)
    else:
        smarts = ms_table['smarts']       
            
    got_ms2 = []
    for m,b,smt in zip(mslist,bslist,smarts):
        delta = abs(refms2['mz'] - m)
        refms2['mserror'] = delta
        if (delta < mserror).any():
            temp = refms2[refms2['mserror'] < mserror].copy()
            temp['smarts'] = smt
            temp['bonds'] = str(b)
            got_ms2.append(temp)
            
    if len(got_ms2)>0:
        got_ms2 = pd.concat(got_ms2)
        got_ms2['mz'] = got_ms2['mz'].round(3)
        got_ms2 = got_ms2.sort_values(by='mz',ascending=True)
        got_ms2 = got_ms2.drop_duplicates(subset=['mz']).reset_index(drop=True)
        #got_ms2['intensity'] = got_ms2['intensity']/np.sqrt(got_ms2['BDE'])*10
        #count = got_ms2['intensity'].sum()
       # mes = sum(np.e**(-0.5*(got_ms2['mserror']/mserror)**2))/len(refms2)
   
    return got_ms2

def fragments_generation(smiles,mode = '+',clean=True, t=None):
    adduct = 'H'
    mol = AllChem.MolFromSmiles(smiles)
    try:
        Chem.Kekulize(mol,clearAromaticFlags=True)
    except Exception as e:
        print(e)
        temp_ = pd.DataFrame({'mz':[0],'smarts':[0],'bonds':[0],'mid':[0]})
        return None
    try:
        f,fw,bs = break_all2(mol,adduct=adduct,mode=mode)
        mids = [0]*len(fw)
    except Exception as e:
        return None
    
    # get valid fragments
    frag_smiles = []
    for _f in f:
        for _ in _f:
            if clean:
                _ = clear_smarts(_)
            m = Chem.MolFromSmarts(_)
            if m:
                if len(re.findall("\[#6\]",_))>2:
                    frag_smiles.append(Chem.MolToSmiles(m)) # at least 2 Carbon atoms
   #             try:
   #                 mv = Chem.rdMolDescriptors.CalcExactMolWt(m)
   #                 if mv > 50:
   #                     frag_smiles.append(Chem.MolToSmiles(m))
   #             except Exception as e:
   #                 print(_)
    return set(frag_smiles)    

def clear_smarts(smarts_string):  
    import re  
    # 正则表达式用于匹配与#0相关的整个原子组（包括前面的键）  
    pattern1 = r'-\[\d*#0\]'
    pattern2 = r'=\[\d*#0\]'
    pattern3 = r'\(\)'
    # 使用re.sub()函数替换匹配到的模式为空字符串（即删除它们）  
    smarts_string = re.sub(pattern1, '', smarts_string)  
    smarts_string = re.sub(pattern2, '', smarts_string)  
    smarts_string = re.sub(pattern3, '', smarts_string)  
    smarts_string.replace("()","")
    return smarts_string 
    

    
def fragment_filter(frag_table,mz_thresh=50):
    # 要不要把自由基过滤掉呢？原来的碎片集合是包含自由基的
    # --要
    frag_table = frag_table.copy()
    frag_table = frag_table[frag_table['mz']>=mz_thresh].reset_index()
    idx = []
    smiles = []
    for s in frag_table['smarts']:
        s = clear_smarts(s)
        m = Chem.MolFromSmarts(s)
        if m is None:
            idx.append(False)
        elif m.GetNumHeavyAtoms() < 3:  # 非H原子要求3个以上
            idx.append(False)
        else:
            idx.append(True)
            smiles.append(Chem.MolToSmiles(m))
    
    frag_table = frag_table[idx].reset_index(drop=True)
    frag_table['smiles'] = smiles 
    return frag_table