import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


def prepare_data():
    df_train = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")
    with open("data/categorical_features.pkl", "rb") as f:
        categorical_features = pickle.load(f)

    X_train, y_train = (
        df_train.drop(["is_positive"], axis=1).copy(),
        df_train["is_positive"].copy(),
    )
    X_val, y_val = df_test.drop(["is_positive"], axis=1).copy(), df_test["is_positive"].copy()

    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(X_train[categorical_features])

    def encode_cats(df, cols, enc):
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    enc.transform(df[cols]).toarray(), columns=enc.get_feature_names_out()
                ),
            ],
            axis=1,
        )
        df = df.drop(cols, axis=1)
        return df

    X_train = encode_cats(X_train, categorical_features, enc)
    X_val = encode_cats(X_val, categorical_features, enc)

    vectorizer = TfidfVectorizer()
    text_column = "review"
    vectorizer.fit(X_train[text_column])

    def vectorize_text(df, col, vectorizer):
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    vectorizer.transform(df[col]).toarray(),
                    columns=vectorizer.get_feature_names_out(),
                ),
            ],
            axis=1,
        )
        df = df.drop(col, axis=1)
        return df

    X_train = vectorize_text(X_train, text_column, vectorizer)
    X_val = vectorize_text(X_val, text_column, vectorizer)

    return X_train, X_val, y_train, y_val
