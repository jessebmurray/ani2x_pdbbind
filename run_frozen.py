from ani2x import load_pdb_bind_filtered_sub, train_frozen_models_cont, get_lr

def main():
    batchsize = 15
    lr_pre = get_lr(batchsize)
    betas = (0.9, 0.96)
    data = load_pdb_bind_filtered_sub()
    train_frozen_models_cont(data, batchsize=batchsize, epochs=500, lr_pre=lr_pre,
                 betas=betas, train_percentage=0.85, name='gen_frozen2')

if __name__ == "__main__":
    main()
