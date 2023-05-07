import os
import copy
import re
import tqdm
import torch
import torchani
import pandas as pd
import numpy as np
from functools import cache
from collections import OrderedDict
from typing import List
from datetime import date
from torch.utils.data import DataLoader
from biopandas.mol2 import PandasMol2
from biopandas.pdb import PandasPdb

DISTANCE_CUTOFF = 5.2
DISTANCE_MASK = 5
SPECIES_ANI2X = {'H', 'C', 'N', 'O', 'S', 'F', 'Cl'}
N_MODELS = 8
PERIODIC_TABLE = """
    H                                                                                                                           He
    Li  Be                                                                                                  B   C   N   O   F   Ne
    Na  Mg                                                                                                  Al  Si  P   S   Cl  Ar
    K   Ca  Sc                                                          Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
    Rb  Sr  Y                                                           Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe
    Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
    Fr  Ra  Ac  Th  Pa  U   Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr  Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
    """.strip().split()

consts_def = {'Rcr': 'radial cutoff',
            'Rca': 'angular cutoff',
            'EtaR': 'radial decay',
            'ShfR': 'radial shift',
            'EtaA': 'angular decay',
            'Zeta': 'angular multiplicity',
            'ShfA': 'angular radial shift',
            'ShfZ': 'angular shift',
            'num_species': 'number of species'}

def get_quality_pdbs_sub(df_gen, binding_symbols={'='},
                    cutoff_r_free=0.26, cutoff_resolution=2.25, cutoff_delta_r=0.06):
    query_binding = df_gen.Binding_Symbol.isin(binding_symbols)
    query_r_free = df_gen.R_free < cutoff_r_free
    query_delta_r = df_gen.delta_R < cutoff_delta_r
    query_resolution = df_gen.Resolution < cutoff_resolution
    queries = (query_binding, query_r_free, query_resolution, query_delta_r)
    queried_pdbs = set.intersection(*map(lambda query: set(df_gen[query].index), queries))
    return queried_pdbs

def get_natom_pdbs_sub(df_bind, natom_cutoff=2_400):
    natoms = df_bind.groupby('PDB_ID').element.count()
    queried_pdbs = set((natoms[natoms < natom_cutoff]).index)
    return queried_pdbs

def get_low_b_factors(df_bind, b_factor_cutoff=50):
    return df_bind.query(f'b_factor < {b_factor_cutoff} | b_factor.isna()')

def apply_masks_sub(df_bind, distance_mask=DISTANCE_MASK):
    within_cutoff = is_within_cutoff(df_bind, distance_cutoff=distance_mask)
    df_bind['Mask'] = True
    df_bind['Mask'] = df_bind['Mask'] & within_cutoff
    df_bind['Mask'] = df_bind['Mask'] & (df_bind.molecule_name != 'HOH')
    return df_bind

def get_train(df_bind, df_gen):
    pdbs_train = set(df_gen.index)
    for protein in ['HSP90', 'TRYP', 'BRD4', 'THRB']:
        pdbs_protein_train, _ = load_protein_benchmark(protein)
        pdbs_train = pdbs_train.intersection(pdbs_protein_train)
    pdbs_ligand_train, _ = load_ligand_benchmark()
    pdbs_train = pdbs_train.intersection(pdbs_ligand_train)
    train_overlap = df_bind.PDB_ID.isin(pdbs_train)
    return df_bind[train_overlap]

def get_test(df_bind):
    pdbs_test = set()
    for protein in ['HSP90', 'TRYP', 'BRD4', 'THRB']:
        _, pdbs_protein_test = load_protein_benchmark(protein)
        pdbs_test.update(pdbs_protein_test)
    _, pdbs_ligand_test = load_ligand_benchmark()
    pdbs_test.update(pdbs_ligand_test)
    test_overlap = df_bind.PDB_ID.isin(pdbs_test)
    return df_bind[test_overlap]

def get_n_overlaps():
    df_bind = load_pdb_bind_filtered_sub(train=False, test=False, convert=False)
    experiment_pdbs = set(df_bind.PDB_ID)
    proteins = ['BRD4', 'BSEC1', 'BSEC2', 'CAII', 'CDK2', 'HCRNAP',
                'HIVPR', 'HSP90', 'MKp38', 'THRB', 'TRYP']
    n_overlaps = {}
    for protein in proteins:
        _, pdbs_test = load_protein_benchmark(protein)
        n_overlaps[protein] = len(pdbs_test.intersection(experiment_pdbs))
    _, pdbs_test = load_ligand_benchmark()
    n_overlaps['ligand'] = len(pdbs_test.intersection(experiment_pdbs))
    n_overlaps = pd.Series(n_overlaps).sort_values(ascending=False)
    return n_overlaps

def load_pdb_bind_filtered_sub(train=True, test=False, convert=True):
    assert not (train and test)
    df_bind = load_pdb_bind()
    df_gen = load_df_gen()
    df_bind = get_low_b_factors(df_bind)
    quality_pdbs = get_quality_pdbs_sub(df_gen)
    natom_pdbs = get_natom_pdbs_sub(df_bind)
    queried_pdbs = set.intersection(natom_pdbs, quality_pdbs)
    df_bind = df_bind[df_bind.PDB_ID.isin(queried_pdbs)]
    df_bind = apply_masks_sub(df_bind)
    df_bind = get_within_cutoff(df_bind, distance_cutoff=DISTANCE_MASK+DISTANCE_CUTOFF)
    if train:
        df_bind = get_train(df_bind, df_gen)
    if test:
        df_bind = get_test(df_bind)
    if convert:
        return convert_to_data(df_bind, df_gen)
    return df_bind

def get_corr(x, y):
    return np.corrcoef(x, y)[0,1]

def get_labels(testloader):
    for _, labels, _, _ in testloader:
        labels = labels
    return labels.numpy()

def get_model_output(model, aev_computer, testloader):
    for _, labels, species_coordinates, mask in testloader:
        species, coordinates = species_coordinates
        aevs = aev_computer.forward((species, coordinates)).aevs
        species_ = species.clone()
        species_[~mask] = -1
        _, output = model((species_, aevs))
    return output.detach().numpy()

def load_best_model(id_, kind, name):
    assert kind in {'pre', 'rand'}
    model_load_path = f'./results/{name}_{kind}_{id_}_best.pth'
    model_load = load_pretrained()
    model_load_sd = torch.load(model_load_path, map_location=torch.device('cpu'))['state_dict']
    model_load.load_state_dict(model_load_sd)
    return model_load

def train_model(data, model, optimizer, id_, kind, batchsize, epochs, train_percentage=0.85, name=''):
    assert kind in {'pre', 'rand'}
    mse = torch.nn.MSELoss()
    consts_ani2x = get_consts_ani2x()
    aev_computer_ani2x = get_aev_computer(consts_ani2x)
    data_training, data_validation = split_data(data, id_, train_percentage)
    trainloader, validloader = get_data_loaders(data_training, data_validation, batchsize)
    train_losses, valid_losses = train(model, optimizer, mse, aev_computer_ani2x,
            trainloader, validloader, epochs=epochs, savepath=f'./results/{name}_{kind}_{id_}_')
    save_list(train_losses, f'train_losses_{name}_{kind}_{id_}')
    save_list(valid_losses, f'valid_losses_{name}_{kind}_{id_}')

def _get_grad_params(parameters):
    return (p for p in parameters if p.requires_grad)

def train_frozen_models(data, batchsize, epochs, lr_pre, lr_rand, betas, train_percentage, p_dropout=0.4, name=''):
    model_pres = [load_pretrained_frozen(id_=i, p_dropout=p_dropout) for i in range(N_MODELS)]
    optimizer_pres  = [torch.optim.Adam(_get_grad_params(model_pres[i].parameters()), lr=lr_pre, betas=betas) for i in range(N_MODELS)]
    model_rands = [load_random() for _ in range(N_MODELS)]
    optimizer_rands  = [torch.optim.Adam(model_rands[i].parameters(), lr=lr_rand, betas=betas) for i in range(N_MODELS)]
    for i in range(N_MODELS):
        train_model(data, model_pres[i], optimizer_pres[i], i, 'pre', batchsize, epochs, train_percentage, name=name)
        train_model(data, model_rands[i], optimizer_rands[i], i, 'rand', batchsize, epochs, train_percentage, name=name)

def train_models(data, batchsize, epochs, lr_pre, lr_rand, betas, train_percentage, name=''):
    model_pres = [load_pretrained(id_=i) for i in range(N_MODELS)]
    model_rands = [load_random() for _ in range(N_MODELS)]
    optimizer_pres  = [torch.optim.Adam(model_pres[i].parameters(), lr=lr_pre, betas=betas) for i in range(N_MODELS)]
    optimizer_rands  = [torch.optim.Adam(model_rands[i].parameters(), lr=lr_rand, betas=betas) for i in range(N_MODELS)]
    for i in range(N_MODELS):
        train_model(data, model_pres[i], optimizer_pres[i], i, 'pre',
                    batchsize, epochs, train_percentage, name=name)
        train_model(data, model_rands[i], optimizer_rands[i], i, 'rand',
                    batchsize, epochs, train_percentage, name=name)

def train_pre_models(data, batchsize, epochs, lr_pre, betas, train_percentage):
    model_pres = [load_pretrained(id_=i) for i in range(N_MODELS)]
    optimizer_pres  = [torch.optim.Adam(model_pres[i].parameters(), lr=lr_pre, betas=betas) for i in range(N_MODELS)]
    for i in range(N_MODELS):
        train_model(data, model_pres[i], optimizer_pres[i], i, 'pre',
                    batchsize, epochs, train_percentage)

def train_rand_models(data, batchsize, epochs, lr_rand, betas, train_percentage):
    model_rands = [load_random() for _ in range(N_MODELS)]
    optimizer_rands  = [torch.optim.Adam(model_rands[i].parameters(), lr=lr_rand, betas=betas) for i in range(N_MODELS)]
    for i in range(N_MODELS):
        train_model(data, model_rands[i], optimizer_rands[i], i, 'rand',
                    batchsize, epochs, train_percentage)

def split_data(data, id_, train_percentage=0.85, fixed_seed=True):
    train_size = int(train_percentage * len(data))
    test_size = len(data) - train_size
    if fixed_seed:
        torch.manual_seed(id_)
    training, validation = torch.utils.data.random_split(data, [train_size, test_size])
    if not fixed_seed:
        training_ids = [entry['ID'] for entry in training]
        validation_ids = [entry['ID'] for entry in validation]
        save_list(training_ids, f'training_ids_{id_}', folder='splits')
        save_list(validation_ids, f'validation_ids_{id_}', folder='splits')
    return training, validation

######
# DATA HANDLING

def get_gd_pdbs(df_gd):
    return set(df_gd['key'].str.upper())

def load_protein_benchmark(protein):
    path_train = f'./data/GD-protein_benchmark/pdbbind_2020_general_cluster_{protein}_train.csv'
    path_test = f'./data/GD-protein_benchmark/pdbbind_2020_general_cluster_{protein}_test.csv'
    pdbs_train = get_gd_pdbs(pd.read_csv(path_train))
    pdbs_test = get_gd_pdbs(pd.read_csv(path_test))
    return pdbs_train, pdbs_test

def load_ligand_benchmark():
    path_train = './data/GD-ligand_benchmark/0_ligand_benchmark_subsampled_train.csv'
    path_test = './data/GD-ligand_benchmark/0_ligand_benchmark_subsampled_test.csv'
    pdbs_train = get_gd_pdbs(pd.read_csv(path_train))
    pdbs_test = get_gd_pdbs(pd.read_csv(path_test))
    return pdbs_train, pdbs_test

def create_atom_name_guide(save=True):
    dfc = pd.read_csv('./data/atom_name_elements.csv', index_col=0)
    assert dfc.groupby(['atom_name', 'molecule_name']).element.nunique().max() == 1
    fails_an = set(dfc.groupby(['atom_name', 'molecule_name']).element.nunique()[dfc.groupby(['atom_name', 'molecule_name']).element.nunique() == 0].reset_index().atom_name)
    fails_mn = set(dfc.groupby(['atom_name', 'molecule_name']).element.nunique()[dfc.groupby(['atom_name', 'molecule_name']).element.nunique() == 0].reset_index().molecule_name)
    dfcf = dfc.groupby(['atom_name', 'molecule_name']).element.nunique()[dfc.groupby(['atom_name', 'molecule_name']).element.nunique() == 0].reset_index()
    assert set(dfcf.molecule_name) == {'STA'}
    dfcf['element'] = dfcf.atom_name.map(lambda name: name[0])
    dfg = dfc.dropna().groupby(['atom_name', 'molecule_name']).element.unique().explode().reset_index()
    dfg = pd.concat([dfg, dfcf]).reset_index(drop=True)
    if save:
        dfg.to_csv('./data/pdb_atom_name_guide-backup.csv', index=False)
    return dfg

def add_elements_pdbbind_protein(save=True):
    df_bind = pd.read_csv('../Data/pdbbind_protein-backup_2023_04_28.csv',
                      dtype={'PDB_ID': str, 'element': str, 'chain_id': str})
    dfg = pd.read_csv('./data/pdb_atom_name_guide.csv')
    dfg_dict = dfg.set_index(['atom_name', 'molecule_name']).element.to_dict()
    element_fill = df_bind.set_index(['atom_name', 'molecule_name']).index.map(lambda pair: dfg_dict.get(pair, np.nan))
    df_bind['element'] = df_bind['element'].fillna(pd.Series(element_fill))
    if save:
        df_bind.to_csv('../Data/pdbbind_protein_elements-backup.csv', index=False)
    return df_bind

def load_casf():
    with open('./data/casf.txt', 'r') as f:
        casf = f.readlines()
    casf = set([entry.rstrip('\n').upper() for entry in casf])
    return casf

def get_pdb_structural(pdb_structure_file):
    struct_pattern = re.compile(
        r'(\w{4}) \s+ (\w+\.?\w*) \s+ (\d{4}) \s+ (-?\d+\.?\d*) \s+ (-?\d+\.?\d*) \s+ (-?\d+\.?\d*)',
        re.VERBOSE)
    with open(pdb_structure_file, 'r') as f:
        lines = f.readlines()
    entries = list()
    for line in lines[7:]:
        struct_match = struct_pattern.match(line)
        entry = {'PDB_ID': struct_match.group(1).upper(),
                'R_factor': float(struct_match.group(4)),
                'R_free': float(struct_match.group(5)),
                'delta_R': float(struct_match.group(6))}
        entries.append(entry)
    return pd.DataFrame(entries).set_index('PDB_ID')

def get_pdb_entries(pdb_info_file):
    affinity_pattern = re.compile(r'(Ki|Kd|IC50)(=|~|<|<=|>|>=)\d')
    ligand_pattern = re.compile(r'\((.*)\)+$')
    with open(pdb_info_file, 'r') as f:
        lines = f.readlines()
    entries = list()
    for line in lines[6:]:
        entry = line.split('  ')
        a, b, c, d, e = entry[:5]
        f = entry[-1]
        binding_match = affinity_pattern.match(e)
        binding_type, binding_symbol = binding_match.group(1), binding_match.group(2)
        ligand = ligand_pattern.search(f).group(1)
        entry = {'PDB_ID': a.upper(),
                'Resolution': float(b) if not b == ' NMR' else np.NAN,
                'Release_Year': int(c),
                'pK': float(d),
                'Binding_Type': binding_type,
                'Binding_Symbol': binding_symbol,
                'Ligand': ligand}
        entries.append(entry)
    return pd.DataFrame(entries).set_index('PDB_ID')

def get_df_gen():
    general_file = '../Data/v2020-other-PL/index/INDEX_general_PL_data.2020'
    refined_file = '../Data/v2020-other-PL/index/INDEX_refined_data.2020'
    df_gen = get_pdb_entries(general_file)
    df_ref = get_pdb_entries(refined_file)
    refined_entry_ids = set(df_ref.index)
    general_entry_ids = set(df_gen.index)
    assert (df_gen[df_gen.index.isin(refined_entry_ids)] == df_ref).all(axis=None)
    df_gen['Refined'] = df_gen.index.isin(refined_entry_ids)
    pdb_structure_file = '../Data/v2020-other-PL/index/INDEX_structure.2020'
    df_gen_struct = get_pdb_structural(pdb_structure_file)
    df_gen = df_gen.join(df_gen_struct, how='inner')
    df_gen['ID'] = pd.Series(np.arange(df_gen.shape[0]) + 1, index=df_gen.index)
    casf_2016 = load_casf()
    df_gen['CASF_2016'] = df_gen.index.isin(casf_2016)
    return df_gen

def save_df_gen():
    df_gen = get_df_gen()
    df_gen.to_csv('./data/pdb_bind_gen_backup.csv')

def load_df_gen():
    return pd.read_csv('./data/pdb_bind_gen.csv', index_col=0)

def save_pdb_bind():
    n_files = 6
    # pdb_bind_path = '../Data/pdbbind_pocket.csv'
    pdb_bind_path = '../Data/pdbbind_protein_elements_12A_ani2xspecies_refined.csv'
    df_bind = pd.read_csv(pdb_bind_path)
    n_interval = df_bind.shape[0] // n_files
    s = np.arange(0, n_interval*n_files, n_interval)
    df_bind_1 = df_bind[s[0]: s[1]]
    df_bind_2 = df_bind[s[1]: s[2]]
    df_bind_3 = df_bind[s[2]: s[3]]
    df_bind_4 = df_bind[s[3]: s[4]]
    df_bind_5 = df_bind[s[4]: s[5]]
    df_bind_6 = df_bind[s[5]: ]
    df_bind_1.to_csv('./data/pdb_bind_sub_1.csv', index=False)
    df_bind_2.to_csv('./data/pdb_bind_sub_2.csv', index=False)
    df_bind_4.to_csv('./data/pdb_bind_sub_4.csv', index=False)
    df_bind_3.to_csv('./data/pdb_bind_sub_3.csv', index=False)
    df_bind_5.to_csv('./data/pdb_bind_sub_5.csv', index=False)
    df_bind_6.to_csv('./data/pdb_bind_sub_6.csv', index=False)

def load_pdb_bind():
    n_files = 6
    dataframes = []
    for i in range(n_files):
        file_number = i + 1
        # file_path = f'./data/pdb_bind_pocket_{file_number}.csv'
        file_path = f'./data/pdb_bind_sub_{file_number}.csv'
        dataframes.append(pd.read_csv(file_path))
    df_bind = pd.concat(dataframes)
    df_bind = df_bind.astype({'x': 'float32', 'y': 'float32', 'z': 'float32'})
    return df_bind

def make_subset(df_bind, df_gen, save=True):
    df_bind = get_within_cutoff(df_bind, distance_cutoff=12)
    if save:
        df_bind.to_csv('../Data/pdbbind_protein_elements_12A-backup.csv', index=False)
    df_bind = restrict_to_species(df_bind, species=SPECIES_ANI2X)
    df_bind = restrict_to_refined(df_bind, df_gen)
    df_bind = df_bind.drop(columns=['atom_name', 'chain_id', 'residue_number',
                        'insertion', 'occupancy', 'blank_4', 'line_idx', 'element2'])
    if save:
        df_bind.to_csv('../Data/pdbbind_protein_elements_12A_ani2xspecies_refined-backup.csv', index=False)
    return df_bind

def filter_casf(df_bind, df_gen, filter_out=True):
    casf_2016 = set(df_gen.query('CASF_2016').index)
    within_casf = df_bind.PDB_ID.isin(casf_2016)
    if filter_out:
        return df_bind[~within_casf]
    return df_bind[within_casf]

def mask_ligands(df_bind):
    df_bind['Mask'] = df_bind.atom_kind == 'L'
    return df_bind

def get_gd_pdbs(df_gd):
    return set(df_gd['key'].str.upper())

def load_ligand_benchmark():
    path_train = './data/GD-ligand_benchmark/0_ligand_benchmark_subsampled_train.csv'
    path_test = './data/GD-ligand_benchmark/0_ligand_benchmark_subsampled_test.csv'
    pdbs_train = get_gd_pdbs(pd.read_csv(path_train))
    pdbs_test = get_gd_pdbs(pd.read_csv(path_test))
    return pdbs_train, pdbs_test

def load_pdb_bind_filtered(filter_out_casf=True, convert=True, ligand_only=False,
                           mask_function=None):
    df_bind_all = load_pdb_bind()
    df_gen = load_df_gen()
    df_bind = filter_casf(df_bind_all, df_gen, filter_out=filter_out_casf)
    df_bind = restrict_to_species(df_bind, species=SPECIES_ANI2X)
    quality_pdbs = get_quality_pdbs(df_gen)
    natom_pdbs = get_natom_pdbs(df_bind_all)
    queried_pdbs = set.intersection(natom_pdbs, quality_pdbs)
    df_bind = df_bind[df_bind.PDB_ID.isin(queried_pdbs)]
    df_bind = get_within_cutoff(df_bind, distance_cutoff=DISTANCE_CUTOFF)
    if mask_function is None:
        df_bind['Mask'] = True
    else:
        df_bind = mask_function(df_bind)
    if ligand_only:
        df_bind = df_bind.query("atom_kind == 'L'")
    if convert:
        data = convert_to_data(df_bind, df_gen)
        return data
    return df_bind

def _get_entry(df_bind_pdb, df_gen, consts_ani2x):
    pdb = df_bind_pdb.PDB_ID.iloc[0]
    species = consts_ani2x.species_to_tensor(df_bind_pdb.element.values) #.unsqueeze(0)
    coordinates = torch.tensor(df_bind_pdb[['x','y','z']].values)
    mask = torch.tensor(df_bind_pdb.Mask.values)
    affinity = df_gen.loc[pdb].pK
    id_ = df_gen.loc[pdb].ID
    entry = {'species': species, 'coordinates': coordinates,
                'affinity': affinity, 'ID': id_, 'mask': mask}
    return entry

def convert_to_data(df_bind, df_gen):
    consts_ani2x = get_consts_ani2x()
    data = df_bind.groupby('PDB_ID').apply(lambda df_bind_pdb: _get_entry(df_bind_pdb, df_gen, consts_ani2x)).tolist()
    return data

def filter_casf(df_bind, df_gen, filter_out=True):
    casf_2016 = set(df_gen.query('CASF_2016').index)
    within_casf = df_bind.PDB_ID.isin(casf_2016)
    if filter_out:
        return df_bind[~within_casf]
    return df_bind[within_casf]

def get_natom_pdbs(df_bind, cutoff_quantile=0.95):
    ligand_natoms = df_bind[df_bind.atom_kind == 'L'].groupby('PDB_ID').element.count()
    ligand_query = ligand_natoms < ligand_natoms.quantile(cutoff_quantile)
    ligand_pdbs = set(ligand_query[ligand_query].index)
    structure_natoms = df_bind.groupby('PDB_ID').element.count()
    structure_query = structure_natoms < structure_natoms.quantile(cutoff_quantile)
    structure_pdbs = set(structure_query[structure_query].index)
    queried_pdbs = set.intersection(ligand_pdbs, structure_pdbs)
    return queried_pdbs

def get_quality_pdbs(df_gen, binding_symbols={'='}, cutoff_quantile=0.95):
    query_binding = df_gen.Binding_Symbol.isin(binding_symbols)
    query_r_free = df_gen.R_free < df_gen.R_free.quantile(cutoff_quantile)
    query_resolution = df_gen.Resolution < df_gen.Resolution.quantile(cutoff_quantile)
    queries = (query_binding, query_r_free, query_resolution)
    queried_pdbs = set.intersection(*map(lambda query: set(df_gen[query].index), queries))
    return queried_pdbs

def restrict_to_refined(df_bind, df_gen):
    refined_pdbs = set(df_gen.query('Refined').index)
    return df_bind[df_bind.PDB_ID.isin(refined_pdbs)]

def restrict_to_species(df_bind, species):
    pdb_in_species = df_bind.groupby('PDB_ID').element.apply(lambda element_list: set(element_list).issubset(species))
    valid_pdbs = set(pdb_in_species[pdb_in_species].index)
    return df_bind[df_bind.PDB_ID.isin(valid_pdbs)]

def get_ligand_cutoffs(df_bind, distance_cutoff):
    ligand_cutoffs = df_bind[df_bind.atom_kind == 'L'].groupby('PDB_ID')[['x', 'y', 'z']].agg(['min', 'max'])
    ligand_cutoffs.columns.names = ['coord', 'measure']
    ligand_cutoffs = ligand_cutoffs.reorder_levels(order=['measure', 'coord'], axis=1)
    ligand_cutoffs.rename(columns={'min': 'Min', 'max': 'Max'}, inplace=True)
    ligand_cutoffs['Min'] -= distance_cutoff
    ligand_cutoffs['Max'] += distance_cutoff
    return ligand_cutoffs

def _check_if_within(df_bind_pdb, ligand_cutoffs):
    pdb = df_bind_pdb.PDB_ID.iloc[0]
    greater_than_min = (df_bind_pdb[['x', 'y', 'z']] > ligand_cutoffs.loc[pdb].Min).all(axis=1)
    less_than_max = (df_bind_pdb[['x', 'y', 'z']] < ligand_cutoffs.loc[pdb].Max).all(axis=1)
    return greater_than_min  & less_than_max

def is_within_cutoff(df_bind, distance_cutoff):
    ligand_cutoffs = get_ligand_cutoffs(df_bind, distance_cutoff)
    return df_bind.groupby('PDB_ID').apply(lambda df_bind_pdb: _check_if_within(df_bind_pdb, ligand_cutoffs)).values

def get_within_cutoff(df_bind, distance_cutoff):
    within_cutoff = is_within_cutoff(df_bind, distance_cutoff)
    return df_bind[within_cutoff]

def get_file(pdb, kind, df_gen, file_type='pocket'):
    assert file_type in {'pocket', 'protein'}
    folder_lookup = {'refined': 'refined-set', 'general': 'v2020-other-PL'}
    kind_lookup = {'ligand': 'ligand.mol2', 'protein': f'{file_type}.pdb'}
    kind_text = kind_lookup[kind]
    subset = 'refined' if df_gen.loc[pdb].Refined else 'general'
    folder = folder_lookup[subset]
    file = f'../Data/{folder}/{pdb.lower()}/{pdb.lower()}_{kind_text}'
    return file

def get_pdbbind(file_type='pocket', save=True):
    assert file_type in {'pocket', 'protein'}
    df_gen = get_df_gen()
    df_hetatm = get_pdb_df(atom_kind='hetatm', df_gen=df_gen, file_type=file_type)
    df_protein = get_pdb_df(atom_kind='protein', df_gen=df_gen, file_type=file_type)
    df_ligand = get_ligand_df(df_gen)
    df_bind = pd.concat([df_ligand, df_protein, df_hetatm])
    df_bind = df_bind[df_bind.PDB_ID != '2W73']
    df_bind = df_bind.sort_values(by=['PDB_ID', 'atom_kind', 'atom_number'])
    if save:
        todays_date = date.today().strftime("%Y_%m_%d")
        filename = f'../Data/pdbbind_{file_type}-backup_{todays_date}.csv'
        df_bind.to_csv(filename, index=False)
    return df_bind

def _drop_empty_columns(df):
    dropped_cols = set(df.nunique()[df.nunique() <= 1].index) # - {'atom_kind'}
    df_updated = df.drop(columns=dropped_cols)
    return df_updated

def get_pdb_df(atom_kind, df_gen, file_type='pocket'):
    assert atom_kind in {'protein', 'hetatm'}
    assert file_type in {'pocket', 'protein'}
    ppdb_key = {'protein': 'ATOM', 'hetatm': 'HETATM'}
    atom_kind_key = {'protein': 'P', 'hetatm': 'H'}
    dfs = []
    for pdb_id in df_gen.index:
        pdb_file = get_file(pdb=pdb_id, kind='protein', df_gen=df_gen, file_type=file_type)
        df = PandasPdb().read_pdb(pdb_file).df[ppdb_key[atom_kind]]
        df['PDB_ID'] = pdb_id
        dfs.append(df)
    df = pd.concat(dfs)
    df = _drop_empty_columns(df)
    df['atom_kind'] = atom_kind_key[atom_kind]
    # drop_columns = ['atom_name', 'chain_id', 'residue_number', 'insertion', 'line_idx']
    # df.drop(columns=drop_columns, inplace=True)
    df.element_symbol = df.element_symbol.str.capitalize()
    df.rename(columns={'element_symbol': 'element', 'residue_name': 'molecule_name',
                        'x_coord': 'x', 'y_coord': 'y', 'z_coord': 'z'}, inplace=True)
    df = df.astype({'x': 'float32', 'y': 'float32', 'z': 'float32'})
    return df

def get_ligand_df(df_gen):
    dfs = []
    for pdb_id in df_gen.index:
        if pdb_id == '2W73':  # Failed PDB
            continue
        mol_file = get_file(pdb=pdb_id, kind='ligand', df_gen=df_gen)
        df = PandasMol2().read_mol2(mol_file).df
        df['PDB_ID'] = pdb_id
        dfs.append(df)
    df = pd.concat(dfs)
    df['atom_kind'] = 'L'
    df = df.astype({'x': 'float32', 'y': 'float32', 'z': 'float32'})
    df['element'] = df.atom_type.map(lambda string: re.match(r'\w+', string).group(0))
    df['b_factor'] = np.nan
    df.drop(columns=['atom_name', 'atom_type', 'subst_id', 'charge'], inplace=True)
    df.rename(columns={'subst_name': 'molecule_name', 'atom_id': 'atom_number'}, inplace=True)
    return df

# TORCH STUFF
#######

def get_layer_sizes():
    layer_sizes = list()
    layer_size = 256
    for i in range(7, -1, -1):
        layer_sizes.append(int(layer_size))
        layer_size *= i / (i+1)
    return layer_sizes

def load_pretrained_frozen(id_=0, p_dropout=0.4):
    model_orig = load_pretrained(id_=id_)
    model_new = OrderedDict()
    consts_ani2x = get_consts_ani2x()
    layer_sizes = get_layer_sizes()
    n_layers_add = 2
    for i in consts_ani2x.species:
        # Get original neural network
        nn_orig = model_orig[i]
        # Freeze original
        for parameter in nn_orig[:-1].parameters():
            parameter.requires_grad_(False)
        # Get the size of the last layer of the original
        last_size = nn_orig[-1].in_features
        # Get sizes of the layers to add
        li = layer_sizes.index(last_size)
        s = layer_sizes[li:li+n_layers_add+1]
        # Make the layers to add (this is tied to 2 layers, n_layers_add = 2)
        nn_add = torch.nn.Sequential(
                    torch.nn.Linear(s[0], s[1]), torch.nn.CELU(alpha=0.1), torch.nn.Dropout(p_dropout),
                    torch.nn.Linear(s[1], s[2]), torch.nn.CELU(alpha=0.1), torch.nn.Dropout(p_dropout),
                    torch.nn.Linear(s[2], 1))
        # Initialize parameters according to Meli's initialization (optional)
        nn_add.apply(init_params)
        # Get the new neural network
        nn_new = torch.nn.Sequential(*nn_orig[:-1], *nn_add)
        # Assert frozen layers
        assert not np.array([parameter.requires_grad for parameter in nn_new[:6].parameters()]).any()
        # Keep the last singleton layer information from the original
        with torch.no_grad():
            nn_new[6].weight[0] = nn_orig[-1].weight[0]
            nn_new[6].bias[0] = nn_orig[-1].bias[0]
        model_new[i] = nn_new
    model_new = torchani.ANIModel(model_new)
    return model_new

def get_aev_computer(consts):
    return torchani.AEVComputer(**consts)

def load_pretrained(id_=0):
    assert id_ in set(range(8))
    torchani_path = get_torchani_path()
    consts_ani2x = get_consts_ani2x()
    networks_path = f'resources/ani-2x_8x/train{id_}/networks'
    network_ani2x_dir = os.path.join(torchani_path, networks_path)  # noqa: E501
    model_pre = torchani.neurochem.load_model(consts_ani2x.species, network_ani2x_dir)
    return model_pre

def load_random():
    model_pre = load_pretrained()
    consts_ani2x = get_consts_ani2x()
    models = OrderedDict()
    for i in consts_ani2x.species:
        # Models S, F, and Cl each have the same architecture
        models[i] = model_pre[i]
    models = copy.deepcopy(models)
    model_rand = torchani.ANIModel(models)
    model_rand.apply(init_params)
    return model_rand

def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        torch.nn.init.zeros_(m.bias)

@cache
def get_torchani_path():
    path = torchani.__file__
    path = path.rstrip('__init__.py')
    return path

@cache
def get_consts_ani2x():
    torchani_path = get_torchani_path()
    parameters_path = 'resources/ani-2x_8x/rHCNOSFCl-5.1R_16-3.5A_a8-4.params'
    consts_file_path = os.path.join(torchani_path, parameters_path)
    consts_ani2x = torchani.neurochem.Constants(consts_file_path)
    return consts_ani2x

def get_consts_ani1x():
    torchani_path = get_torchani_path()
    parameters_path = 'resources/ani-1x_8x/rHCNO-5.2R_16-3.5A_a4-8.params'
    consts_file_path = os.path.join(torchani_path, parameters_path)
    consts_ani1x = torchani.neurochem.Constants(consts_file_path)
    return consts_ani1x

def get_consts_aescore():
    consts_ani1x = get_consts_ani1x()
    consts_aescore = {**consts_ani1x}
    consts_aescore['Rca'] = 5.2
    consts_aescore['EtaA'] = torch.tensor([3.5])
    consts_aescore['ShfA'] = torch.tensor([0])
    consts_aescore['ShfZ'] = torch.tensor([0, np.pi])
    consts_aescore['num_species'] = 10
    return consts_aescore

def pad_collate(batch, species_pad_value=-1, coords_pad_value=0,
                mask_pad_value=False, device= None):
    ids, labels, species_and_coordinates, mask = zip(*batch)
    species, coordinates = zip(*species_and_coordinates)
    pad_species = torch.nn.utils.rnn.pad_sequence(
        species, batch_first=True, padding_value=species_pad_value)
    pad_coordinates = torch.nn.utils.rnn.pad_sequence(
        coordinates, batch_first=True, padding_value=coords_pad_value)
    pad_mask = torch.nn.utils.rnn.pad_sequence(
        mask, batch_first=True, padding_value=mask_pad_value)
    labels = torch.tensor(np.array(labels)).reshape(1, -1).squeeze(0)
    return np.array(ids), labels, (pad_species, pad_coordinates), pad_mask


class Data(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.n: int = -1
        self.ids: List[str] = []
        self.labels: List[float] = []  # energies or affinity
        self.species: List[torch.Tensor] = []
        self.coordinates: List[torch.Tensor] = []
        self.mask: List[torch.Tensor] = []

    def __len__(self) -> int:
        return self.n

    def __getitem__(
        self, idx: int
    ):  # -> Tuple[str, float, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Tuple[str, float, Tuple[torch.Tensor, torch.Tensor]]
            Item from the dataset (PDB IDs, labels, species, coordinates)
        """
        return (
            self.ids[idx],
            self.labels[idx],
            (self.species[idx], self.coordinates[idx]),
            self.mask[idx])

    def load(self, data):
        self.n = len(data)
        for entry in data:
            self.ids.append(entry['ID'])
            self.labels.append(entry['affinity'])
            self.species.append(entry['species'])
            self.coordinates.append(entry['coordinates'])
            self.mask.append(entry['mask'])

def get_data_loader(dataset, batchsize=40, shuffle=True):
    out = Data()
    out.load(dataset)
    return DataLoader(out, batch_size=batchsize, shuffle=shuffle, collate_fn=pad_collate)

def get_data_loaders(training, validation, batchsize=40):
    trainloader = get_data_loader(training, batchsize=batchsize)
    validloader = get_data_loader(validation, batchsize=batchsize)
    return trainloader, validloader

def savemodel(model: torch.nn.ModuleDict, path) -> None:
    """Save AffinityModel."""
    torch.save(
        {"state_dict": model.state_dict(),}, path,)
    # mlflow.log_artifact(path)


def train(model, optimizer, loss_function, aev_computer, trainloader, testloader,
    epochs, savepath=None, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model and AEVComputer to device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.to(device)
    aev_computer.to(device)

    train_losses: List[float] = []
    valid_losses: List[float] = []

    best_valid_loss = np.inf
    best_epoch = 0

    for epoch in tqdm.trange(epochs, desc="Training"):

        # Model in training mode
        model.train()

        epoch_loss: float = 0.0

        # Training
        for _, labels, species_coordinates, mask in trainloader:

            # Move data to device
            labels = labels.to(device).float()
            species = species_coordinates[0].to(device)
            coordinates = species_coordinates[1].to(device).float()  # converts to float32
            mask = mask.to(device)

            aevs = aev_computer.forward((species, coordinates)).aevs

            species_ = species.clone()
            species_ = species_.to(device)
            species_[~mask] = -1

            optimizer.zero_grad()

            _, output = model((species_, aevs))

            loss = loss_function(output, labels)

            loss.backward()

            # Update weights
            optimizer.step()

            epoch_loss += loss.item()

        else:
            valid_loss: float = 0.0

            # Model in evaluation mode
            model.eval()

            # Validation
            with torch.no_grad():
                for _, labels, species_coordinates, mask in testloader:

                    # Move data to device
                    labels = labels.to(device)
                    species = species_coordinates[0].to(device)
                    coordinates = species_coordinates[1].to(device)
                    mask = mask.to(device)

                    aevs = aev_computer.forward((species, coordinates)).aevs

                    species_ = species.clone()
                    species_ = species_.to(device)
                    species_[~mask] = -1

                    _, output = model((species_, aevs))

                    valid_loss += loss_function(output, labels).item()

            # Normalise losses
            epoch_loss /= len(trainloader)
            valid_loss /= len(testloader)

            # Save best model
            if valid_loss < best_valid_loss and savepath is not None:
                # TODO: Save Optimiser
                modelname = savepath + "best.pth"

                savemodel(model, modelname)

                best_epoch = epoch
                best_valid_loss = valid_loss

            train_losses.append(epoch_loss)
            valid_losses.append(valid_loss)

            # Log losses
            # mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            # mlflow.log_metric("valid_loss", valid_loss, step=epoch)

    # Track best model
    # if best_epoch != 0:
    #     mlflow.log_artifact(modelname)

    #     mlflow.log_param("best _epoch", best_epoch)

    return train_losses, valid_losses

def get_list(name):
    with open(f'./losses/{name}.txt', 'r') as f:
        lines = f.readlines()
        lines = [float(line.rstrip('\n')) for line in lines]
    return lines

def get_losses(name, kind, train_valid, n_models=8):
    losses = []
    for i in range(n_models):
        filepath = f'{train_valid}_losses{name}_{kind}_{i}'
        losses.append(get_list(filepath))
    return np.array(losses)

def save_list(lines, name, folder='losses'):
    with open(f'./{folder}/{name}.txt', 'w+') as f:
        for line in lines:
            f.write(f"{line}\n")

# df_bind.groupby(['atom_name', 'molecule_name', 'atom_kind']).element.value_counts(dropna=False).to_frame().rename(columns={'element': 'count'}).reset_index().to_csv('../Data/atom_name_elements.csv')