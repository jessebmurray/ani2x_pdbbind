from ani2x import *
df_gen = load_df_gen()
consts_ani2x = get_consts_ani2x()
aev_computer_ani2x = get_aev_computer(consts_ani2x)
aev_computer = aev_computer_ani2x
model_pres = [load_pretrained(id_=i) for i in range(N_MODELS)]
for model in model_pres:
    model.eval()
df_bind = load_pdb_bind()
df_bind = apply_masks_compare_ligand(df_bind)
data_pl_loader = get_data_loader_protein_ligand(df_bind, df_gen)
data_pl_loader_nw = get_data_loader_protein_ligand_no_water(df_bind, df_gen)

data_l_loader = get_data_loader_ligand(df_bind, df_gen)
data_l_loader_nw = get_data_loader_ligand_no_water(df_bind, df_gen)

consts_ani2x = get_consts_ani2x()
aev_computer_ani2x = get_aev_computer(consts_ani2x)
output_pl = np.array([get_model_output(model_pres[i], aev_computer_ani2x, data_pl_loader) for i in range(N_MODELS)])
output_pl_nw = np.array([get_model_output(model_pres[i], aev_computer_ani2x, data_pl_loader_nw) for i in range(N_MODELS)])
output_l = np.array([get_model_output(model_pres[i], aev_computer_ani2x, data_l_loader) for i in range(N_MODELS)])
output_l_nw = np.array([get_model_output(model_pres[i], aev_computer_ani2x, data_l_loader_nw) for i in range(N_MODELS)])
np.save('./ani2x_refined_lig/output_pl.npy', output_pl)
np.save('./ani2x_refined_lig/output_pl_nw.npy', output_pl_nw)
np.save('./ani2x_refined_lig/output_l.npy', output_l)
np.save('./ani2x_refined_lig/output_l_nw.npy', output_l_nw)
