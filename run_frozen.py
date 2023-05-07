from ani2x import load_pdb_bind_filtered_sub, train_frozen_models

def main():
    lr_pre = 0.5 * 1e-5
    lr_rand = 1.5 * lr_pre
    betas = (0.9, 0.96)
    data = load_pdb_bind_filtered_sub()
    train_frozen_models(data, batchsize=40, epochs=500, lr_pre=lr_pre, lr_rand=lr_rand,
                 betas=betas, train_percentage=0.85, p_dropout=0.4, name='gen_frozen')

if __name__ == "__main__":
    main()
