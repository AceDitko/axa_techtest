import joblib
import pandas as pd

MODEL_PATH = 'models/xgb_model.joblib'

def predict(row, threshold=0.5):
    """
    Predict the target value for a given row of input data.

    Parameters:
    row (pd.Series): A pandas Series representing a single row of input data.
    threshold (float): The threshold for classifying the predicted probability into binary classes.

    Returns:
    Dictionary: A dictionary containing the prediction, probability, label, and threshold.
    """
    # Load the trained model
    model = joblib.load(MODEL_PATH)

    # Convert the row to a DataFrame
    input_data = pd.DataFrame([row])

    # Make prediction
    probability = model.predict_proba(input_data)[0,1]
    prediction = int(probability > threshold)

    return {
        "prediction": prediction,
        "probability": probability,
        "label": ("History of Mental Illness" 
        if prediction == 1 
        else "No History of Mental Illness"
        ),
        "threshold": threshold
    }