from indicators import prepare_technical_data
from train import train_models
from backtesting import backtest_all

def main():
    prepare_technical_data()
    train_models()
    backtest_all()


if __name__ == "__main__":
    print("Main module started!")
    main()
