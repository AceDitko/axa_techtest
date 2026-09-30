import argparse
import joblib
import pandas as pd

MODEL_PATH = 'models/xgb_model.joblib'

def predict(model, row, threshold=0.3):
    """
    Predict the target value for a given row of input data.

    Parameters:
    model (Pipeline): A trained scikit-learn pipeline model.
    row (pd.Series): A pandas Series representing a single row of input data.
    threshold (float): The threshold for classifying the predicted probability into binary classes.

    Returns:
    Dictionary: A dictionary containing the prediction, probability, label, and threshold.
    """
    # Make prediction
    probability = model.predict_proba(pd.DataFrame([row]))[0,1]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("--output_path")
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    data = pd.read_csv(args.input_path)

    model = joblib.load(MODEL_PATH)

    results = []
    for _, row in data.iterrows():
        results.append(predict(model, 
                               row, 
                               threshold=args.threshold))

    if args.output_path:
        pd.DataFrame(results).to_csv(args.output_path, index=False)
    else:
        print(results)


if __name__ == "__main__":
    main()
