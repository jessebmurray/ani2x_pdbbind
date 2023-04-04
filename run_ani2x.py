from ani2x import *

SPECIES_ANI2X = {'H', 'C', 'N', 'O', 'S', 'F', 'Cl'}

distance_cutoff = 6
lr = 0.00007
epochs = 150  # e
batchsize = 32

path = torchani.__file__
path = path.rstrip('__init__.py')
const_file_ani2x = os.path.join(path, 'resources/ani-2x_8x/rHCNOSFCl-5.1R_16-3.5A_a8-4.params')

df_gen = get_df_gen()
aev_computer_ani2x, consts_ani2x = get_aevs_from_file(const_file_ani2x)

network_ani2x_dir = os.path.join(path, 'resources/ani-2x_8x/train0/networks')  # noqa: E501
network_ani2x = torchani.neurochem.load_model(consts_ani2x.species, network_ani2x_dir)
model_ani2x = torchani.nn.Sequential(aev_computer_ani2x, network_ani2x)

species_ani2x = consts_ani2x.species
assert set(species_ani2x) == SPECIES_ANI2X
num_species = len(species_ani2x)
models = OrderedDict()
for i in species_ani2x:
    # Models S, F, and Cl each have the same architecture
    models[i] = network_ani2x[i]
model_pre = torchani.ANIModel(models)
model_rand = torchani.ANIModel(copy.deepcopy(models))
model_rand.apply(init_params);

# model_pre2 = torchani.nn.Sequential(aev_computer_ani2x, model_pre)
# model_rand2 = torchani.nn.Sequential(aev_computer_ani2x, model_rand)


data, failed_entries = load_data(distance_cutoff, consts_ani2x, df_gen)
save_list(failed_entries, 'failed_entries')
train_size = int(0.8 * len(data))
test_size = len(data) - train_size
training, validation = torch.utils.data.random_split(data, [train_size, test_size])

trainloader, validloader = get_data_loaders(training, validation, batch_size=batchsize)

# Define optimizers
optimizer_pre = optim.Adam(model_pre.parameters(), lr=lr)
optimizer_rand = optim.Adam(model_rand.parameters(), lr=lr)

# Define losses
mse = nn.MSELoss()
# Train model
train_losses_pre, valid_losses_pre = train(model_pre, optimizer_pre, mse, aev_computer_ani2x,
                    trainloader, validloader, epochs=epochs, savepath='./results_pre/')

train_losses_rand, valid_losses_rand = train(model_rand, optimizer_rand, mse, aev_computer_ani2x,
                    trainloader, validloader, epochs=epochs, savepath='./results_rand/')

save_list(train_losses_pre, 'train_losses_pre')
save_list(valid_losses_pre, 'valid_losses_pre')
save_list(train_losses_rand, 'train_losses_rand')
save_list(valid_losses_rand, 'valid_losses_rand')
