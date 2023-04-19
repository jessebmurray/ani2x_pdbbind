import os
import copy
import json
import re
import tqdm
import torch
import torchani
import pandas as pd
import numpy as np
from functools import cache
from collections import OrderedDict
from typing import Optional, Tuple, Union, List
from rdkit import Chem
from biopandas.mol2 import PandasMol2
from biopandas.pdb import PandasPdb
from torch.utils.data import DataLoader
from torch import nn, optim
from datetime import date

DISTANCE_CUTOFF = 5.2
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

SPECIES_ANI2X = {'H', 'C', 'N', 'O', 'S', 'F', 'Cl'}

consts_def = {'Rcr': 'radial cutoff',
            'Rca': 'angular cutoff',
            'EtaR': 'radial decay',
            'ShfR': 'radial shift',
            'EtaA': 'angular decay',
            'Zeta': 'angular multiplicity',
            'ShfA': 'angular radial shift',
            'ShfZ': 'angular shift',
            'num_species': 'number of species'}

def get_corr(x, y):
    return np.corrcoef(x, y)[0][1]

def get_labels(testloader):
    for _, labels, _ in testloader:
        labels = labels
    return labels.numpy()

def get_model_output(model, aev_computer, testloader):
    for _, labels, species_coordinates in testloader:
        species, coordinates = species_coordinates
        aevs = aev_computer.forward((species, coordinates)).aevs
        _, output = model((species, aevs))
    return output.detach().numpy()

def load_best_model(id_, kind):
    assert kind in {'pre', 'rand'}
    model_load_path = f'./results/{kind}_{id_}_best.pth'
    model_load = load_pretrained()
    model_load_sd = torch.load(model_load_path, map_location=torch.device('cpu'))['state_dict']
    model_load.load_state_dict(model_load_sd)
    return model_load

def train_model(data, model, optimizer, id_, kind, batchsize, epochs, train_percentage=0.85):
    assert kind in {'pre', 'rand'}
    mse = nn.MSELoss()
    consts_ani2x = get_consts_ani2x()
    aev_computer_ani2x = get_aev_computer(consts_ani2x)
    data_training, data_validation = split_data(data, id_, train_percentage)
    trainloader, validloader = get_data_loaders(data_training, data_validation, batchsize)
    train_losses, valid_losses = train(model, optimizer, mse, aev_computer_ani2x,
            trainloader, validloader, epochs=epochs, savepath=f'./results/{kind}_{id_}_')
    save_list(train_losses, f'train_losses_{kind}_{id_}')
    save_list(valid_losses, f'valid_losses_{kind}_{id_}')

def train_models(data, batchsize, epochs, lr_pre, lr_rand, betas, train_percentage):
    model_pres = [load_pretrained(id_=i) for i in range(N_MODELS)]
    model_rands = [load_random() for _ in range(N_MODELS)]
    optimizer_pres  = [optim.Adam(model_pres[i].parameters(), lr=lr_pre, betas=betas) for i in range(N_MODELS)]
    optimizer_rands  = [optim.Adam(model_rands[i].parameters(), lr=lr_rand, betas=betas) for i in range(N_MODELS)]
    for i in range(N_MODELS):
        train_model(data, model_pres[i], optimizer_pres[i], i, 'pre',
                    batchsize, epochs, train_percentage)
        train_model(data, model_rands[i], optimizer_rands[i], i, 'rand',
                    batchsize, epochs, train_percentage)

def split_data(data, id_, train_percentage=0.85):
    train_size = int(train_percentage * len(data))
    test_size = len(data) - train_size
    training, validation = torch.utils.data.random_split(data, [train_size, test_size])
    training_ids = [entry['ID'] for entry in training]
    validation_ids = [entry['ID'] for entry in validation]
    save_list(training_ids, f'training_ids_{id_}', folder='splits')
    save_list(validation_ids, f'validation_ids_{id_}', folder='splits')
    return training, validation


######
# DATA HANDLING

def load_casf():
    with open('casf.txt', 'r') as f:
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
    with open(pdb_info_file, 'r') as f:
        lines = f.readlines()
    entries = list()
    for line in lines[6:]:
        entry = line.split('  ')
        entry = entry[:5]
        a, b, c, d, e = entry
        match_obj = affinity_pattern.match(e)
        binding_type, binding_symbol = match_obj.group(1), match_obj.group(2)
        entry = {'PDB_ID': a.upper(),
                'Resolution': float(b) if not b == ' NMR' else np.NAN,
                'Release_Year': int(c),
                'pK': float(d),
                'Binding_Type': binding_type,
                'Binding_Symbol': binding_symbol}
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

def split_data():
    n_files = 6
    pdb_bind_path = '../Data/pdbbind_pocket.csv'
    df_bind = pd.read_csv(pdb_bind_path)
    n_interval = df_bind.shape[0] // n_files
    s = np.arange(0, n_interval*n_files, n_interval)
    df_bind_1 = df_bind[s[0]: s[1]]
    df_bind_2 = df_bind[s[1]: s[2]]
    df_bind_3 = df_bind[s[2]: s[3]]
    df_bind_4 = df_bind[s[3]: s[4]]
    df_bind_5 = df_bind[s[4]: s[5]]
    df_bind_6 = df_bind[s[5]: ]
    df_bind_1.to_csv('./data/pdb_bind_pocket_1.csv', index=False)
    df_bind_2.to_csv('./data/pdb_bind_pocket_2.csv', index=False)
    df_bind_4.to_csv('./data/pdb_bind_pocket_4.csv', index=False)
    df_bind_3.to_csv('./data/pdb_bind_pocket_3.csv', index=False)
    df_bind_5.to_csv('./data/pdb_bind_pocket_5.csv', index=False)
    df_bind_6.to_csv('./data/pdb_bind_pocket_6.csv', index=False)

def load_pdb_bind():
    n_files = 6
    dataframes = []
    for i in range(n_files):
        file_number = i + 1
        file_path = f'./data/pdb_bind_pocket_{file_number}.csv'
        dataframes.append(pd.read_csv(file_path))
    df_bind = pd.concat(dataframes)
    df_bind = df_bind.astype({'x': 'float32', 'y': 'float32', 'z': 'float32'})
    return df_bind

def filter_casf(df_bind, df_gen, filter_out=True):
    casf_2016 = set(df_gen[df_gen.CASF_2016].index)
    within_casf = df_bind.PDB_ID.isin(casf_2016)
    if filter_out:
        return df_bind[~within_casf]
    return df_bind[within_casf]

def load_pdb_bind_filtered(filter_out_casf=True, convert=True):
    df_bind_all = load_pdb_bind()
    df_gen = load_df_gen()
    df_bind = filter_casf(df_bind_all, df_gen, filter_out=filter_out_casf)
    df_bind = restrict_to_species(df_bind, species=SPECIES_ANI2X)
    quality_pdbs = get_quality_pdbs(df_gen)
    natom_pdbs = get_natom_pdbs(df_bind_all)
    queried_pdbs = set.intersection(natom_pdbs, quality_pdbs)
    df_bind = df_bind[df_bind.PDB_ID.isin(queried_pdbs)]
    df_bind = get_within_cutoff(df_bind, distance_cutoff=DISTANCE_CUTOFF)
    if convert:
        data = convert_to_data(df_bind, df_gen)
        return data
    return df_bind

def _get_entry(df_bind_pdb, df_gen, consts_ani2x):
    pdb = df_bind_pdb.PDB_ID.iloc[0]
    species = consts_ani2x.species_to_tensor(df_bind_pdb.element.values) #.unsqueeze(0)
    coordinates = torch.tensor(df_bind_pdb[['x','y','z']].values)
    affinity = df_gen.loc[pdb].ID
    id_ = df_gen.loc[pdb].ID
    entry = {'species': species, 'coordinates': coordinates,
                'affinity': affinity, 'ID': id_}
    return entry

def convert_to_data(df_bind, df_gen):
    consts_ani2x = get_consts_ani2x()
    data = df_bind.groupby('PDB_ID').apply(lambda df_bind_pdb: _get_entry(df_bind_pdb, df_gen, consts_ani2x)).tolist()
    return data

def filter_casf(df_bind, df_gen, filter_out=True):
    casf_2016 = set(df_gen[df_gen.CASF_2016].index)
    within_casf = df_bind.PDB_ID.isin(casf_2016)
    if filter_out:
        return df_bind[~within_casf]
    return df_bind[within_casf]

def get_natom_pdbs(df_bind, cutoff_quantile=0.975):
    ligand_natoms = df_bind[df_bind.atom_kind == 'L'].groupby('PDB_ID').element.count()
    ligand_query = ligand_natoms < ligand_natoms.quantile(cutoff_quantile)
    ligand_pdbs = set(ligand_query[ligand_query].index)
    structure_natoms = df_bind.groupby('PDB_ID').element.count()
    structure_query = structure_natoms < structure_natoms.quantile(cutoff_quantile)
    structure_pdbs = set(structure_query[structure_query].index)
    queried_pdbs = set.intersection(ligand_pdbs, structure_pdbs)
    return queried_pdbs

def get_quality_pdbs(df_gen, binding_symbols={'='}, cutoff_quantile=0.975):
    query_binding = df_gen.Binding_Symbol.isin(binding_symbols)
    query_r_free = df_gen.R_free < df_gen.R_free.quantile(cutoff_quantile)
    query_resolution = df_gen.Resolution < df_gen.Resolution.quantile(cutoff_quantile)
    queries = (query_binding, query_r_free, query_resolution)
    queried_pdbs = set.intersection(*map(lambda query: set(df_gen[query].index), queries))
    return queried_pdbs

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

def get_within_cutoff(df_bind, distance_cutoff):
    ligand_cutoffs = get_ligand_cutoffs(df_bind, distance_cutoff)
    within_cutoff = df_bind.groupby('PDB_ID').apply(lambda df_bind_pdb: _check_if_within(df_bind_pdb, ligand_cutoffs))
    return df_bind[within_cutoff.values]

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
    df_gen = get_df_gen()
    df_hetatm = get_pdb_df(atom_kind='hetatm', df_gen=df_gen, file_type='pocket')
    df_protein = get_pdb_df(atom_kind='protein', df_gen=df_gen, file_type='pocket')
    df_ligand = get_ligand_df(df_gen)
    df_bind = pd.concat([df_ligand, df_protein, df_hetatm])
    df_bind = df_bind[df_bind.PDB_ID != '2W73']
    df_bind = df_bind.sort_values(by=['PDB_ID', 'atom_kind', 'atom_number'])
    if save:
        todays_date = date.today().strftime("%Y_%m_%d")
        filename = f'../Data/pdbbind_pocket-backup_{todays_date}.csv'
        df_bind.to_csv(filename, index=False)
    return df_bind

def _drop_empty_columns(df):
    dropped_cols = set(df.nunique()[df.nunique() <= 1].index) - {'atom_kind'}
    df_updated = df.drop(columns=dropped_cols)
    return df_updated

def get_pdb_df(atom_kind, df_gen, file_type='pocket'):
    assert atom_kind in {'protein', 'hetatm'}
    ppdb_key = {'protein': 'ATOM', 'hetatm': 'HETATM'}
    atom_kind_key = {'protein': 'P', 'hetatm': 'H'}
    dfs = []
    for pdb_id in df_gen.index:
        pdb_file = get_file(pdb=pdb_id, kind='protein', df_gen=df_gen, file_type=file_type)
        df = PandasPdb().read_pdb(pdb_file).df[ppdb_key[atom_kind]]
        df['PDB_ID'] = pdb_id
        dfs.append(df)
    df = pd.concat(dfs)
    df['atom_kind'] = atom_kind_key[atom_kind]
    df = _drop_empty_columns(df)
    drop_columns = ['atom_name', 'chain_id', 'residue_number', 'insertion', 'line_idx']
    df.drop(columns=drop_columns, inplace=True)
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

def pad_collate(
    batch,
    species_pad_value=-1,
    coords_pad_value=0,
    device: Optional[Union[str, torch.device]] = None,
    ) -> Union[
    Tuple[np.ndarray, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    Tuple[np.ndarray, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
    ids, labels, species_and_coordinates = zip(*batch)
    species, coordinates = zip(*species_and_coordinates)
    pad_species = torch.nn.utils.rnn.pad_sequence(
        species, batch_first=True, padding_value=species_pad_value)
    pad_coordinates = torch.nn.utils.rnn.pad_sequence(
        coordinates, batch_first=True, padding_value=coords_pad_value)
    labels = torch.tensor(np.array(labels)).reshape(1, -1).squeeze(0)
    return np.array(ids), labels, (pad_species, pad_coordinates)


class Data(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.n: int = -1
        self.ids: List[str] = []
        self.labels: List[float] = []  # energies or affinity
        self.species: List[torch.Tensor] = []
        self.coordinates: List[torch.Tensor] = []

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
            (self.species[idx], self.coordinates[idx]))

    def load(self, data):
        self.n = len(data)
        for entry in data:
            self.ids.append(entry['ID'])
            self.species.append(entry['species'])
            self.coordinates.append(entry['coordinates'])
            self.labels.append(entry['affinity'])

def get_data_loader(dataset, batchsize=40, shuffle=True):
    out = Data()
    out.load(dataset)
    return DataLoader(out, batch_size=batchsize, shuffle=shuffle, collate_fn=pad_collate)

def get_data_loaders(training, validation, batchsize=40):
    trainloader = get_data_loader(training, batchsize=batchsize)
    validloader = get_data_loader(validation, batchsize=batchsize)
    return trainloader, validloader

def savemodel(model: nn.ModuleDict, path) -> None:
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
        for _, labels, species_coordinates in trainloader:

            # Move data to device
            labels = labels.to(device).float()
            species = species_coordinates[0].to(device)
            coordinates = species_coordinates[1].to(device).float()

            aevs = aev_computer.forward((species, coordinates)).aevs

            optimizer.zero_grad()

            _, output = model((species, aevs))

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
                for _, labels, species_coordinates in testloader:

                    # Move data to device
                    labels = labels.to(device)
                    species = species_coordinates[0].to(device)
                    coordinates = species_coordinates[1].to(device)
                    aevs = aev_computer.forward((species, coordinates)).aevs

                    # Forward pass
                    _, output = model((species, aevs))

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

def save_list(lines, name, folder='losses'):
    with open(f'./{folder}/{name}.txt', 'w+') as f:
        for line in lines:
            f.write(f"{line}\n")

