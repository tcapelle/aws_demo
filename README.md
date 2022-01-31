# Training a Language model for financial market seetiment analysis

- The data comes from https://www.kaggle.com/yash612/stockmarket-sentiment-dataset?select=stock_data.csv
- Apply tokenizer
- Train model and log data to wandb

## Notebooks

- The first notebook prepares the data and uploads the dataset as a `wandb` artifact
- The second notebook downloads the processed data from the previously logged artifact and then trains a Bert model on it. At the end, it performns a simple sweep for hyper param optimizations.

## Script

You can also run the training script from the command line doing:

```bash

$ python train.py

```
Call `--help` to get info!
```bash
$ python train.py --help
usage: train.py [-h] [--epochs EPOCHS] [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
                [--model_name MODEL_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       The number of training epochs
  --learning_rate LEARNING_RATE
                        The initial learning rate, uses linear scheduler
  --batch_size BATCH_SIZE
                        batch size, 32 fits on a 16GB GPU
  --model_name MODEL_NAME
                        The old mightty bert!
```
