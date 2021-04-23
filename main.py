import os

from transformers import AutoTokenizer
import pandas as pd
import torch

from train import train, seed_everything
from inference import inference, load_test_dataset, RE_Dataset


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "P2_KLUE"
    os.environ["WANDB_LOG_MODEL"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed = 17
    seed_everything(seed)
    model = train()

    TOK_NAME = "xlm-roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)

    test_dataset_dir = "test-with-marker.tsv"
    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    test_dataset = RE_Dataset(test_dataset, test_label)

    # predict answer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred_answer = inference(model, test_dataset, device)

    output = pd.DataFrame(pred_answer, columns=["pred"])
    output.to_csv("./prediction/submission.csv", index=False)

