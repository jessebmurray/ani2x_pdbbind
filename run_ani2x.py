from ani2x import *

def main():
    train_models(batchsize=26, epochs=150, lr_pre=1e-5, lr_rand=1e-4)

if __name__ == "__main__":
    main()
