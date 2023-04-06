import torch
import torchani
import os
import tqdm
import copy
# import mlflow
from collections import OrderedDict
import pandas as pd
import numpy as np
from rdkit import Chem
from biopandas.mol2 import PandasMol2
from typing import Optional, Tuple, Union, List
from torch.utils.data import DataLoader
from torch import nn, optim
from functools import cache

DISTANCE_CUTOFF = 6
N_MODELS = 8
SPECIES_ANI2X = {'H', 'C', 'N', 'O', 'S', 'F', 'Cl'}

def train_pre_models(batchsize, epochs, lr_pre):
    trainloader, validloader = load_casf_split(DISTANCE_CUTOFF, batchsize)
    model_pres = [load_pretrained(id_=i) for i in range(N_MODELS)]
    optimizer_pres  = [optim.Adam(model_pres[i].parameters(), lr=lr_pre) for i in range(N_MODELS)]
    for i in range(N_MODELS):
        train_model(trainloader, validloader, model_pres[i], optimizer_pres[i], i, 'pre', epochs)

def train_rand_models(batchsize, epochs, lr_rand):
    trainloader, validloader = load_casf_split(DISTANCE_CUTOFF, batchsize)
    model_rands = [load_random() for _ in range(N_MODELS)]
    optimizer_rands  = [optim.Adam(model_rands[i].parameters(), lr=lr_rand) for i in range(N_MODELS)]
    for i in range(N_MODELS):
        train_model(trainloader, validloader, model_rands[i], optimizer_rands[i], i, 'rand', epochs)

def train_models(batchsize, epochs, lr_pre, lr_rand):
    trainloader, validloader = load_casf_split(DISTANCE_CUTOFF, batchsize)
    model_pres = [load_pretrained(id_=i) for i in range(N_MODELS)]
    model_rands = [load_random() for _ in range(N_MODELS)]
    optimizer_pres  = [optim.Adam(model_pres[i].parameters(), lr=lr_pre) for i in range(N_MODELS)]
    optimizer_rands  = [optim.Adam(model_rands[i].parameters(), lr=lr_rand) for i in range(N_MODELS)]
    for i in range(N_MODELS):
        train_model(trainloader, validloader, model_pres[i], optimizer_pres[i], i, 'pre', epochs)
        train_model(trainloader, validloader, model_rands[i], optimizer_rands[i], i, 'rand', epochs)

def train_model(trainloader, validloader, model, optimizer, id_, kind, epochs):
    assert kind in {'pre', 'rand'}

    # Define losses
    mse = nn.MSELoss()
    consts_ani2x = get_consts_ani2x()
    aev_computer_ani2x = torchani.AEVComputer(**consts_ani2x)
    train_losses, valid_losses = train(model, optimizer, mse, aev_computer_ani2x,
            trainloader, validloader, epochs=epochs, savepath=f'./results/{kind}_{id_}_')
    save_list(train_losses, f'train_losses_{kind}_{id_}')
    save_list(valid_losses, f'valid_losses_{kind}_{id_}')

def load_casf_split(distance_cutoff, batchsize):
    df_gen = get_df_gen()
    df_training, df_validation = split_by_casf(df_gen)
    data_training, failed_entries_training = load_data(distance_cutoff, df_training)
    data_validation, failed_entries_validation = load_data(distance_cutoff, df_validation)
    save_list(failed_entries_training, 'failed_entries_training')
    save_list(failed_entries_validation, 'failed_entries_validation')
    trainloader, validloader = get_data_loaders(data_training, data_validation, batchsize)
    return trainloader, validloader

def split_by_casf(df_gen):
    casf = load_casf()
    df_validation = df_gen.loc[df_gen.index.isin(casf)]
    df_training = df_gen.loc[~df_gen.index.isin(casf)]
    return df_training, df_validation

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

def load_casf():
    with open('casf.txt', 'r') as f:
        casf = f.readlines()
    casf = set([entry.rstrip('\n').upper() for entry in casf])
    return casf

def get_df_gen():
    general_file = '../Data/v2020-other-PL/index/INDEX_general_PL_data.2020'
    refined_file = '../Data/v2020-other-PL/index/INDEX_refined_data.2020'
    df_gen = get_pdb_entries(general_file)
    df_ref = get_pdb_entries(refined_file)
    refined_entry_ids = set(df_ref.index)
    general_entry_ids = set(df_gen.index)
    assert set(refined_entry_ids).issubset(set(general_entry_ids))
    df_gen['Refined'] = df_gen.index.isin(refined_entry_ids)
    df_gen['ID'] = pd.Series(np.arange(df_gen.shape[0]) + 1, index=df_gen.index)
    return df_gen

def get_pdb_entries(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    entries = list()
    for line in lines[6:]:
        entry = line.split('  ')
        # Can add parser for Ki<10mM stuff by going [:5]
        entry = entry[:4]
        a, b, c, d = entry
        entry = {'Entry ID': a.upper(),
                'Resolution': float(b) if not b == ' NMR' else np.NAN,
                'Release Year': int(c),
                'pK': float(d)}
        entries.append(entry)
    return pd.DataFrame(entries).set_index('Entry ID')

def _get_id(df_gen, pdb):
    return df_gen.loc[pdb]['ID']

def get_entry(pdb, distance_cutoff, df_gen):
    species, coordinates = get_species_coordinates(pdb, distance_cutoff, df_gen)
    affinity = _get_binding_affinity(df_gen, pdb)
    id_ = _get_id(df_gen, pdb)
    entry = {'species': species, 'coordinates': coordinates,
            'affinity': affinity, 'id': id_}
    return entry

def load_data(distance_cutoff, df_gen):
    data = []
    failed_entries = []
    for pdb in df_gen.index:
        try:
            entry = get_entry(pdb, distance_cutoff, df_gen)
            data.append(entry)
        except WrongElements:
            pass
        except:
            failed_entries.append(pdb)
    return data, failed_entries

def _get_binding_affinity(df_gen, pdb):
    return  df_gen.loc[pdb].pK

def _get_subset(df_gen, pdb):
    subset = 'refined' if df_gen.loc[pdb].Refined else 'general'
    return subset

def get_file(pdb, kind, df_gen):
    folder_lookup = {'refined': 'refined-set', 'general': 'v2020-other-PL'}
    kind_lookup = {'ligand': 'ligand.mol2', 'protein': 'protein.pdb'}
    kind_text = kind_lookup[kind]
    subset = _get_subset(df_gen, pdb)
    folder = folder_lookup[subset]
    file = f'../Data/{folder}/{pdb.lower()}/{pdb.lower()}_{kind_text}'
    return file

# def get_aevs_from_file(file_name):
#     consts = torchani.neurochem.Constants(file_name)
#     aev_computer = torchani.AEVComputer(**consts)
#     return aev_computer, consts

def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        torch.nn.init.zeros_(m.bias)

class WrongElements(Exception):
    pass

def get_protein_df(pdb, df_gen):
    colspecs = [(0, 6), (6, 11), (12, 16), (16, 17), (17, 20), (21, 22), (22, 26),
            (26, 27), (30, 38), (38, 46), (46, 54), (54, 60), (60, 66), (76, 78),
            (78, 80)]
    names = ['ATOM', 'serial', 'name', 'altloc', 'resname', 'chainid', 'resseq',
         'icode', 'x', 'y', 'z', 'occupancy', 'tempfactor', 'element', 'charge']
    pdb_file = get_file(pdb=pdb, kind='protein', df_gen=df_gen)
    df = pd.read_fwf(pdb_file, names=names, colspecs=colspecs)
    df = df.loc[df.ATOM == 'ATOM']
    df = df.drop(columns='ATOM')
    df = df.rename(columns = {'element': 'species'})
    df = df.astype({'x': 'float32', 'y': 'float32', 'z': 'float32'})
    if not set(df.species.unique()).issubset(SPECIES_ANI2X):
        raise WrongElements
    return df

def get_ligand_df(pdb, df_gen):
    mol_file = get_file(pdb=pdb, kind='ligand', df_gen=df_gen)
    df = PandasMol2().read_mol2(mol_file).df
    df['species'] = df.atom_type.str.split('.').str[0]
    df = df.astype({'x': 'float32', 'y': 'float32', 'z': 'float32'})
    if not set(df.species.unique()).issubset(SPECIES_ANI2X):
        raise WrongElements
    return df

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

def get_species_coordinates(pdb, distance_cutoff, df_gen):
    protein = get_protein_df(pdb, df_gen)
    ligand = get_ligand_df(pdb, df_gen)
    for i in ["x","y","z"]:
        protein = protein[protein[i] < float(ligand[i].max())+distance_cutoff]
        protein = protein[protein[i] > float(ligand[i].min())-distance_cutoff]
    # Reduce to just the neural network elements
    protein_ligand = pd.concat([protein, ligand], join='inner')
    protein_ligand = protein_ligand.loc[protein_ligand.species.isin(SPECIES_ANI2X)]
    coordinates = torch.tensor(protein_ligand[['x','y','z']].values) #.unsqueeze(0)
    consts_ani2x = get_consts_ani2x()
    species = consts_ani2x.species_to_tensor(protein_ligand.species.values) #.unsqueeze(0)
    return species, coordinates

def pad_collate(
    batch,
    species_pad_value=-1,
    coords_pad_value=0,
    device: Optional[Union[str, torch.device]] = None,
    ) -> Union[
    Tuple[np.ndarray, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    Tuple[np.ndarray, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
    """
    Collate function to pad batches.
    Parameters
    ----------
    batch:
        Batch
    species_pad_value:
        Padding value for species vector
    coords_pad_value:
        Padding value for coordinates
    device: Optional[Union[str, torch.device]]
        Computation device
    Returns
    -------
    Tuple[np.ndarray, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    Notes
    -----
    :code:`torch.utils.data.Dataset`'s :code:`__getitem__` returns a batch that need
    to be padded. This function can be passed to :code:`torch.utils.data.DataLoader`
    as :code:`collate_fn` argument in order to pad species and coordinates
    """
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

        # TODO: Better way to avoid mypy complaints?
        self.n: int = -1
        self.ids: List[str] = []
        self.labels: List[float] = []  # energies or affinity
        self.species: List[torch.Tensor] = []
        self.coordinates: List[torch.Tensor] = []

    def __len__(self) -> int:
        """
        Number of protein-ligand complexes in the dataset.
        Returns
        -------
        int
            Dataset length
        """
        return self.n

    def __getitem__(
        self, idx: int
    ):  # -> Tuple[str, float, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get item from dataset.
        Parameters
        ----------
        idx: int
            Item index within the dataset
        Returns
        -------
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
            self.ids.append(entry['id'])
            self.species.append(entry['species'])
            self.coordinates.append(entry['coordinates'])
            self.labels.append(entry['affinity'])

def get_data_loader(dataset, batch_size=40, shuffle=True):
    out = Data()
    out.load(dataset)
    return DataLoader(out, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)


def get_data_loaders(training, validation, batch_size=40):
    trainloader = get_data_loader(training, batch_size=batch_size)
    validloader = get_data_loader(validation, batch_size=batch_size)
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


def save_list(lines, name):
    with open(f'./losses/{name}.txt', 'w+') as f:
        for line in lines:
            f.write(f"{line}\n")
