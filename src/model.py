from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

def create_pipeline(categorical_cols, categorical_transformer):
    """
    Create a machine learning pipeline with preprocessing and model training.

    Parameters:
    categorical_cols (list): List of categorical column names.
    categorical_transformer (Transformer): Preprocessing transformer for categorical features.

    Returns:
    Pipeline: A scikit-learn pipeline object.
    """    
    # Define the column transformer for preprocessing
    preprocessor = ColumnTransformer(
        [
            ('categorical', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'  # Keep other columns unchanged
    )

    model = XGBClassifier(eval_metric='logloss', random_state=42)

    # Create the pipeline with preprocessing and model training
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])