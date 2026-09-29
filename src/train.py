import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .model import create_pipeline

DATA_PATH = 'data/depression_data.csv'
MODEL_PATH = 'models/xgb_model.joblib'

TARGET = "History of Mental Illness"

def main():
    # Load the processed data
    data = pd.read_csv(DATA_PATH)

    # Split the data into features and target
    X = data.drop(columns=[TARGET])
    y = data[TARGET].map({"Yes": 1, "No": 0})  # Convert target to binary

    categorical_cols = X.select_dtypes(
        include=['object']
    ).columns.tolist()

    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    model = create_pipeline(categorical_cols, categorical_transformer)

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    print(f"Model trained and saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()

    