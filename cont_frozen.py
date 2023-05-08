from ani2x import load_pdb_bind_filtered_sub, train_frozen_models_cont, get_lr

def main():
    batchsize = 15
    lr_pre = get_lr(batchsize)
    lr_rand = 1.5 * lr_pre
    betas = (0.9, 0.96)
    data = load_pdb_bind_filtered_sub()
    train_frozen_models_cont(data, batchsize=batchsize, epochs=500, lr_pre=lr_pre, lr_rand=lr_rand,
                 betas=betas, train_percentage=0.85, p_dropout=0.4, name='gen_frozen')

if __name__ == "__main__":
    main()
