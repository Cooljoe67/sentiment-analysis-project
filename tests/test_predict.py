import pytest

from src.predict import load_model, predict_texts


@pytest.fixture
def model():
    return load_model("models/sentiment.joblib")


@pytest.mark.parametrize(
    "text, expected_label",
    [
        ("Great movie!", 1),
        ("The service was terrible and the food was awful.", 0),
    ],
)
def test_sentiment_predictions(model, text, expected_label):
    preds, _ = predict_texts(model, [text])
    assert preds[0] == expected_label
