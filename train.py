import pickle as pickle
import os
import random

import numpy as np
import pandas as pd
import torch
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from madgrad import MADGRAD
from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification,
    RobertaConfig,
    Trainer,
    TrainingArguments,
)

from load_data import *
from adamp import AdamP


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# 평가를 위한 metrics function.
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
    }


def train():
    # load model and tokenizer
    MODEL_NAME = "xlm-roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    dataset = load_data("train-with-marker.tsv")

    # 인물:사망_국가: 40 은 데이터가 1개라 stratify에서 오류가 발생한다.
    # 데이터를 보충하기 전까지는 제거해준다.
    dataset = dataset[dataset.label != 40]

    train_dataset, valid_dataset = train_test_split(
        dataset, test_size=0.2, stratify=dataset[["label"]]
    )

    train_label = train_dataset["label"].values
    valid_label = valid_dataset["label"].values

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_valid = tokenized_dataset(valid_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_valid_dataset = RE_Dataset(tokenized_valid, valid_label)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # setting model hyperparameter
    roberta_config = RobertaConfig.from_pretrained(MODEL_NAME)
    roberta_config.num_labels = 42
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, config=roberta_config
    )
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir="./results",  # output directory
        save_total_limit=2,  # number of total save model.
        # save_steps=500,  # model saving step.
        load_best_model_at_end=True,
        num_train_epochs=10,  # total number of training epochs
        learning_rate=1e-5,  # learning_rate
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,  # batch size for evaluation
        warmup_steps=300,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=100,  # log saving step.
        dataloader_num_workers=4,
        label_smoothing_factor=0.5,
        evaluation_strategy="epoch",  # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        # eval_steps=200,  # evaluation step.
    )
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_valid_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
    )

    # train model
    trainer.train()

    return model


def main():
    random_seed = 1231248
    seed_everything(random_seed)
    train()


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "P2_KLUE"
    os.environ["WANDB_LOG_MODEL"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    main()

