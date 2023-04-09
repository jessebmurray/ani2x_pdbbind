from ani2x import *

def main():
    lr_both = 0.5 * 1e-5
    betas = (0.9, 0.96)
    data = load_data_6_angstrom_refined()
    train_models(data, batchsize=30, epochs=150,
                 lr_pre=lr_both, lr_rand=lr_both, betas=betas, train_percentage=0.85)

if __name__ == "__main__":
    main()
