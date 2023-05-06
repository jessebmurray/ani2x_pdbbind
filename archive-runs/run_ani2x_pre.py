from ani2x import load_pdb_bind_filtered, train_pre_models

def main():
    betas = (0.9, 0.96)
    data = load_pdb_bind_filtered()
    train_pre_models(data, batchsize=40, epochs=200,
                      lr_pre=0.5 * 1e-5, betas=betas, train_percentage=0.85)

if __name__ == "__main__":
    main()
