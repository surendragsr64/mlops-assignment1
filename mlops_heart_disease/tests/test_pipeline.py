import os
import joblib
import numpy as np

def test_model_file_exists():
    """Check if the packaged model file exists"""
    assert os.path.exists("../models/heart_disease_pipeline.pkl")

def test_pipeline_prediction_shape():
    """Ensure pipeline returns a prediction with correct shape"""
    pipeline = joblib.load("../models/heart_disease_pipeline.pkl")
    sample = np.array([[63, 1, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1, 1]])
    pred = pipeline.predict(sample)
    assert pred.shape == (1,)

def test_pipeline_prediction_values():
    """Check prediction output is binary (0 or 1)"""
    pipeline = joblib.load("../models/heart_disease_pipeline.pkl")
    sample = np.array([[63, 1, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1, 1]])
    pred = pipeline.predict(sample)
    assert pred[0] in [0, 1]
