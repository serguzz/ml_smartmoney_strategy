# Smart Money ML Strategy

This project focuses on developing a machine learning-based strategy for trading. The goal is to analyze market data, identify patterns, and make informed trading decisions using predictive models. Among various technical indicators, BOS - break of structure and CHOCH - change of character are used. Two boosting machine learning models are used for comparison: XGBoost and CatBoost.
Models are build and tested for spot and futures markets. Both Long and Short trading is covered.

## Features

- Data collection and preprocessing for financial markets.
- Implementation of machine learning algorithms for trading strategies.
- Backtesting and performance evaluation of strategies.

## Installation
- Install dependencies from requirements.txt

## Usage

1. Use config.py to set up needed trading pairs, intervals (timeframe) and date range.
2. Run main.py to fetch data, add indicators, train models and backtest. Among other, this will create folders structure of the project.
3. Run separate scripts for the specific tasks:
    - fetch_data.py - fetches history prices and saves to data/ohlcv folder
    - indicators.py - adds indicators, saves to data/indicators folder
    - train.py - adds targets, trains and saves models
    - backtesting.py - backtests models predictions

## Project Structure

```
smartmoney_ml_strategy/
├── data/               # Raw and processed data
    ├── ohlcv/          # Raw price/volume data
    ├── indicators/     # Data with indicators
    ├── targets/        # Data with indicators and targets
    ├── predictions/    # Saved predictions of the models
    ...
├── models/             # Saved machine learning models
├── src/                # Source code - Python scripts for various tasks
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
