import argparse, os, wandb
import numpy as np
from wandb.sdk.integration_utils.data_logging import ValidationDataLogger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset, Features, ClassLabel, Value, load_metric

recall_metric = load_metric("recall")
f1_metric = load_metric("f1")
accuracy_metric = load_metric("accuracy")
precision_metric = load_metric("precision")

dataset_path = '.'

labels = ["negative", "positive"]

def load_data():
    stock_features = Features({"Text": Value("string"), "labels": ClassLabel(names=labels)})
    dataset = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(dataset_path, "train.csv"),
            "test": os.path.join(dataset_path, "test.csv"),
        },
        delimiter=",",
        features=stock_features,
    )
    return dataset

def tokenize_data(dataset, model_name="bert-base-cased"):

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length")

    def tokenize_function(examples):
        return tokenizer(examples["Text"], padding="max_length", truncation=True)

    return dataset.map(tokenize_function, batched=True), data_collator


def compute_metrics(eval_pred, log_preds=True):
    "Get a bunch of metrics and log predictions to wandb"
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    recall = recall_metric.compute(predictions=predictions, references=labels, average="macro")["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    precision = precision_metric.compute(predictions=predictions, references=labels, average="macro")["precision"]
    
    return {"recall": recall, "f1": f1, "accuracy": accuracy, "precision": precision}


default_training_args = {
    "per_device_train_batch_size": 64,
    "per_device_eval_batch_size": 64,
    "num_train_epochs": 2,
    "learning_rate": 2e-5,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "logging_strategy": "steps",
    "logging_first_step": True,
    "logging_steps": 5,
    "report_to": "wandb",
    "fp16": True,
    "dataloader_num_workers": 6,
}


def get_trainer(
    model, output_dir, data_collator, training_args, train, test
):
    "Prepare the hf Trainer"
    training_args = TrainingArguments(output_dir=output_dir, **training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    return trainer


def train(train_args=default_training_args, model_name="bert-base-cased"):

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    dataset = load_data()
    tokenized_datasets, data_collator = tokenize_data(dataset, model_name)
    
    trainer = get_trainer(
        output_dir=f"training_dir",
        model=model,
        data_collator=data_collator,
        training_args=train_args,
        train=tokenized_datasets["train"],
        test=tokenized_datasets["test"],
    )

    trainer.train()


def update_args(args):
    default_training_args["learning_rate"] = args.learning_rate
    default_training_args["per_device_train_batch_size"] = args.batch_size
    default_training_args["per_device_eval_batch_size"] = args.batch_size
    default_training_args["num_train_epochs"] = args.epochs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=5, help="The number of training epochs, default=5")
    parser.add_argument("--learning_rate", default=1e-5, help="The initial learning rate, uses linear scheduler, default=1e-5")
    parser.add_argument("--batch_size", default=16, help="batch size, 32 fits on a 16GB GPU, default=16")
    parser.add_argument( "--model_name", default="bert-base-cased", help="A Huggingface model name compatible with text classification!, default=bert-base-cased")

    args = parser.parse_args()
    
    print('Training')
    with wandb.init(config=args, project="aws_demo"):
        args = wandb.config
        update_args(args)
        train(default_training_args, model_name=args.model_name)

        
if __name__=='__main__':
    main()