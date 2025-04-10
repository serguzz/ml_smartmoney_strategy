from indicators import prepare_technical_data
from train import train_all_timeframes_models
from backtesting import backtest_all_timeframes

def main():
    prepare_technical_data()
    train_all_timeframes_models()
    backtest_all_timeframes()


if __name__ == "__main__":
    print("Main module started!")
    main()
