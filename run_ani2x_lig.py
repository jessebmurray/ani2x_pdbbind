from ani2x import load_pdb_bind_filtered, train_models

def main():
    lr_both = 0.5 * 1e-5
    betas = (0.9, 0.96)
    data = load_pdb_bind_filtered(ligand_only=True)
    train_models(data, batchsize=45, epochs=500,
                 lr_pre=lr_both, lr_rand=lr_both, betas=betas, train_percentage=0.85, name='lig_only')

if __name__ == "__main__":
    main()
