import sys
from typing import Any

import numpy as np
from joblib import load
from numpy.typing import NDArray


def load_model(model_path: str) -> Any:
    """Load and return a trained classifier."""
    return load(model_path)


def predict_texts(
    classifier: Any, input_texts: list[str]
) -> tuple[list[int], list[float | None]]:
    """Return labels and probability-of-positive for each text."""
    preds: NDArray[Any] = classifier.predict(input_texts)
    if hasattr(classifier, "predict_proba"):
        probs_arr: NDArray[np.float64] = classifier.predict_proba(input_texts)[:, 1]
        probs = [float(p) for p in probs_arr.tolist()]
    else:
        probs = [None] * len(input_texts)
    return preds.astype(int).tolist(), probs


def get_sentiment(label: int) -> str:
    """Convert numeric label to sentiment."""
    return "positive" if label == 1 else "negative"


def predict_single_text(classifier: Any, text: str) -> dict[str, Any]:
    """Predict sentiment for a single text."""
    predictions, probabilities = predict_texts(classifier, [text])

    return {
        "text": text,
        "sentiment": get_sentiment(predictions[0]),
        "probability": probabilities[0],
    }


def print_prediction(result: dict[str, Any]) -> None:
    """Print prediction result."""
    print("Text:", result["text"])
    print("Sentiment:", result["sentiment"])
    print("Probability:", result["probability"])


def main() -> None:
    """Parse CLI arguments and run prediction."""
    text = sys.argv[1] if len(sys.argv) > 1 else "This project is awesome!"
    model_path = "models/sentiment.joblib"

    model = load_model(model_path)
    result = predict_single_text(model, text)
    print_prediction(result)


if __name__ == "__main__":
    main()