import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


SEED = 42


def prepare_train_test():
    df = pd.read_csv("data/reviews.csv")
    df.drop(["Unnamed: 0", "Clothing ID"], axis=1, inplace=True)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df.info()

    # # For recommended_ind == 1
    # plt.figure(figsize=(8, 6))
    # plt.hist(df[df['recommended_ind'] == 1]['rating'], bins=5, color='RoyalBlue', edgecolor='gray')
    # plt.title('Histogram of Ratings (Recommended = 1)')
    # plt.xlabel('Rating')
    # plt.ylabel('Frequency')
    # plt.show()

    # # For recommended_ind == 0
    # plt.figure(figsize=(8, 6))
    # plt.hist(df[df['recommended_ind'] == 0]['rating'], bins=5, color='FireBrick', edgecolor='gray')
    # plt.title('Histogram of Ratings (Recommended = 0)')
    # plt.xlabel('Rating')
    # plt.ylabel('Frequency')
    # plt.show()

    df.drop(["recommended_ind"], axis=1, inplace=True)

    # plt.figure(figsize=(8, 6))
    # plt.hist(df['rating'], bins=5, color='RoyalBlue', edgecolor='gray')
    # plt.title('Histogram of Ratings')
    # plt.xlabel('Rating')
    # plt.ylabel('Frequency')
    # plt.show()

    df["is_positive"] = (df["rating"] > 3).astype(int)
    df.drop(["rating"], axis=1, inplace=True)

    df["title"] = df["title"].fillna("")
    df["review_text"] = df["review_text"].fillna("")
    df["review"] = df[["title", "review_text"]].agg("\n".join, axis=1)
    df["review"] = df["review"].replace("", np.nan)
    df.drop(["title", "review_text"], axis=1, inplace=True)

    print("review len max:", df["review"].str.len().max())
    df = df[df["review"] != "\n"].copy()

    cols_sorted = [
        "division_name",
        "department_name",
        "class_name",
        # "clothing_id",
        "age",
        "positive_feedback_count",
        "review",
        "is_positive",
    ]
    df = df[cols_sorted].copy()

    df_train, df_test = train_test_split(df, test_size=0.25, random_state=SEED)
    del df

    def fill_missing(df, modes):
        df = df.fillna(modes)
        return df

    modes_train = df_train.mode().iloc[0]
    df_train = fill_missing(df_train, modes_train)
    df_test = fill_missing(df_test, modes_train)

    scaler = StandardScaler()
    numeric_features = ["age", "positive_feedback_count"]
    scaler.fit(df_train[numeric_features])

    def scale_nums(df, cols, scaler):
        # Apply scaling to numeric columns
        df[cols] = scaler.transform(df[cols])

        return df

    df_train = scale_nums(df_train, numeric_features, scaler)
    df_test = scale_nums(df_test, numeric_features, scaler)

    categorical_features = ["division_name", "department_name", "class_name"]  # , 'clothing_id']
    encoders = {
        col: OrdinalEncoder(
            encoded_missing_value=-1, handle_unknown="use_encoded_value", unknown_value=-1
        )
        for col in categorical_features
    }
    for col in categorical_features:
        encoders[col].fit(df_train[[col]])

    def prepare_cats(df, encoders):
        for col in encoders.keys():
            df[col] = encoders[col].transform(df[[col]])

        return df

    df_train = prepare_cats(df_train, encoders)
    df_test = prepare_cats(df_test, encoders)

    df_train[categorical_features] = df_train[categorical_features] + 1
    df_test[categorical_features] = df_test[categorical_features] + 1

    df_train[categorical_features + numeric_features] = df_train[
        categorical_features + numeric_features
    ].astype(int)
    df_test[categorical_features + numeric_features] = df_test[
        categorical_features + numeric_features
    ].astype(int)

    print("df_train.shape:", df_train.shape)
    print("df_test.shape:", df_test.shape)

    df_train.to_csv("data/train.csv", index=False)
    df_test.to_csv("data/test.csv", index=False)

    with open("data/categorical_features.pkl", "wb") as f:
        pickle.dump(categorical_features, f)
    with open("data/numeric_features.pkl", "wb") as f:
        pickle.dump(numeric_features, f)


if __name__ == "__main__":
    prepare_train_test()
