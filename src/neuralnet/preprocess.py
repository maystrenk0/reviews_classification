import pandas as pd
import pickle

from transformers import BertTokenizer

import torch
from torch.utils.data import TensorDataset, DataLoader


def tokenize_reviews(df, tokenizer, max_length):
    df = df.copy()
    tokenizer_output = ["input_ids", "attention_mask"]

    # Function to tokenize the text and return input_ids and attention_mask
    def tokenize_text(text):
        encoding = tokenizer(
            text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        return (
            encoding["input_ids"].squeeze(0).tolist(),
            encoding["attention_mask"].squeeze(0).tolist(),
        )

    # Apply the function to each row of the DataFrame and create new columns
    df[tokenizer_output] = df["review"].apply(lambda x: pd.Series(tokenize_text(x)))

    df = df.drop(["review"], axis=1)
    for col in tokenizer_output:
        new_columns = pd.DataFrame(df[col].tolist(), index=df.index)
        new_columns.columns = [f"{col}_{i}" for i in range(max_length)]

        # Concatenate the new columns with the original DataFrame
        df = pd.concat([df, new_columns], axis=1)

    df = df.drop(tokenizer_output, axis=1)

    return df


def prepare_data(transformer_model, max_length, device, seed):
    # Set seeds for random modules
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    df_train = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")
    with open("data/categorical_features.pkl", "rb") as f:
        categorical_features = pickle.load(f)
    with open("data/numeric_features.pkl", "rb") as f:
        numeric_features = pickle.load(f)

    X_train, y_train = (
        df_train.drop(["is_positive"], axis=1).copy(),
        df_train["is_positive"].copy(),
    )
    X_val, y_val = df_test.drop(["is_positive"], axis=1).copy(), df_test["is_positive"].copy()

    tokenizer = BertTokenizer.from_pretrained(transformer_model)

    X_train = tokenize_reviews(X_train, tokenizer, max_length)
    X_val = tokenize_reviews(X_val, tokenizer, max_length)

    X_train = torch.tensor(X_train.astype(float).values, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.astype(float).values, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val.values, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val.values, dtype=torch.float32).to(device)

    # Create the dataset and dataloader
    BATCH_SIZE = 1024
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

    # Determine the size of embeddings for categorical features
    embedding_sizes = [
        (df_train[c].nunique() + 1, int(min(50, (df_train[c].nunique() + 1) ** 0.5)))
        for c in categorical_features
    ]

    n_continuous = len(numeric_features)

    columns_idx = dict()
    columns_idx["INPUT_IDS_START"] = len(categorical_features) + len(numeric_features)
    columns_idx["INPUT_IDS_END"] = len(categorical_features) + len(numeric_features) + max_length
    columns_idx["ATTENTION_MASK_START"] = (
        len(categorical_features) + len(numeric_features) + max_length
    )
    columns_idx["ATTENTION_MASK_END"] = (
        len(categorical_features) + len(numeric_features) + 2 * max_length
    )
    columns_idx["CATEGORICAL_START"] = 0
    columns_idx["CATEGORICAL_END"] = len(categorical_features)
    columns_idx["NUMERIC_START"] = len(categorical_features)
    columns_idx["NUMERIC_END"] = len(categorical_features) + len(numeric_features)

    return train_loader, val_loader, columns_idx, embedding_sizes, n_continuous
