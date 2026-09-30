import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.model import create_pipeline


def test_create_pipeline_returns_pipeline():
    categorical_cols = ['Marital Status']
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    pipeline = create_pipeline(categorical_cols, categorical_transformer)

    assert isinstance(pipeline, Pipeline)


def test_pipeline_has_expected_steps():
    categorical_cols = ['Marital Status']
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    pipeline = create_pipeline(categorical_cols, categorical_transformer)

    expected_steps = ['preprocessor', 'classifier']
    actual_steps = [step[0] for step in pipeline.steps]

    assert actual_steps == expected_steps