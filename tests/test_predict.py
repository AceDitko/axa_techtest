import numpy as np
import pandas as pd

from src.predict import predict

class MockModel:
    def predict_proba(self, data):
        # Return a fixed probability for testing
        return np.array([[0.3, 0.7]])


def test_predict_returns_expected_probability():
    model = MockModel()

    row = pd.Series({
        "Age": 31,
    })

    result = predict(model, row)

    assert result["probability"] == 0.7
    assert result["prediction"] == 1