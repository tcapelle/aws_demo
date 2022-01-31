# Training a Language model for financial market seetiment analysis

- The data comes from https://www.kaggle.com/yash612/stockmarket-sentiment-dataset?select=stock_data.csv
- Apply tokenizer
- Train model and log data to wandb

## Notebooks

- The first notebook prepares the data and uploads the dataset as a `wandb` artifact
- The second notebook downloads the processed data from the previously logged artifact and then trains a Bert model on it. At the end, it performns a simple sweep for hyper param optimizations.