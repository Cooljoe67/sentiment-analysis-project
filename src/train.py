import os

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def load_and_validate_data(data_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV and ensures it has the required columns.
    """
    df = pd.read_csv(data_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns")
    return df

def split_data(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split the DataFrame into training and testing sets."""
    try:
        # Stratified splitting preserves the label distribution
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"],
            df["label"],
            test_size=0.2,
            random_state=42,
            stratify=df["label"],
        )
    except ValueError:
        # Use a regular split if stratification fails on a small dataset
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"],
            df["label"],
            test_size=0.2,
            random_state=42,
        )

    return X_train, X_test, y_train, y_test


def create_model() -> Pipeline:
    """Create a TF-IDF and Logistic Regression pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 1),
                    max_features=10_000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight="balanced"
                ),
            ),
        ]
    )


def save_model(model: Pipeline, model_path: str) -> None:
    """Save the trained model pipeline to disk."""
    model_directory = os.path.dirname(model_path)

    if model_directory:
        os.makedirs(model_directory, exist_ok=True)

    dump(model, model_path)


if __name__ == "__main__":
    data_path = "data/sentiments.csv"
    model_path = "models/sentiment_model.joblib"

    dataframe = load_and_validate_data(data_path)
    X_train, X_test, y_train, y_test = split_data(dataframe)

    model = create_model()

    # Train the model using only the training set
    model.fit(X_train, y_train)

    # Evaluate the model using unseen test data
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    save_model(model, model_path)
    print(f"Model saved to: {model_path}")