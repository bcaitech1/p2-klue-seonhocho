import pickle as pickle
import os
import pandas as pd
import torch

# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach()
            for key, val in self.tokenized_dataset.items()
            # key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def add_entity(s: str, s1: int, d1: int, s2: int, d2: int) -> str:
    return (
        f"{s[:s1]}[e1]{s[s1:d1+1]}[/e1]{s[d1+1:s2]}[e2]{s[s2:d2+1]}[/e2]{s[d2+1:]}"
        if s1 < s2
        else f"{s[:s2]}[e2]{s[s2:d2+1]}[/e2]{s[d2+1:s1]}[e1]{s[s1:d1+1]}[/e1]{s[d1+1:]}"
    )


# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type):
    label = []
    idx = 0
    for i in dataset[8]:
        if i == "blind":
            label.append(100)
        else:
            label.append(label_type[i])
        idx += 1

    sentences = dataset.apply(
        lambda x: add_entity(x[1], x[3], x[4], x[6], x[7]), axis=1
    )
    out_dataset = pd.DataFrame(
        {
            "sentence": sentences,
            "entity_01": dataset[2],
            "entity_02": dataset[5],
            "label": label,
        }
    )
    return out_dataset


# tsv 파일을 불러옵니다.
def load_data(dataset_dir):
    # load label_type, classes
    with open("/opt/ml/input/data/label_type.pkl", "rb") as f:
        label_type = pickle.load(f)
    # load dataset
    dataset = pd.read_csv(dataset_dir, delimiter="\t", header=None)
    # preprecessing dataset
    dataset = preprocessing_dataset(dataset, label_type)

    return dataset


# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.
def tokenized_dataset(dataset, tokenizer):
    concat_entity = []
    for e01, e02 in zip(dataset["entity_01"], dataset["entity_02"]):
        temp = ""
        temp = e01 + "[SEP]" + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset["sentence"]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=100,
        add_special_tokens=True,
    )
    return tokenized_sentences
