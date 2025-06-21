import os
import joblib
import numpy as np
from pydantic import BaseModel
from datetime import date

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

class TradeSignal(BaseModel):
    ticker: str
    option_symbol: str
    option_price: float
    strike: float
    expiry: date
    pred_return: float

def load_models():
    if not os.path.exists(os.path.join(MODEL_DIR, "mean_model.pkl")):
        raise FileNotFoundError("mean_model.pkl not found in models directory")
    if not os.path.exists(os.path.join(MODEL_DIR, "quantile_models.pkl")):
        raise FileNotFoundError("quantile_models.pkl not found in models directory")
    if not os.path.exists(os.path.join(MODEL_DIR, "ranker_model.pkl")):
        raise FileNotFoundError("ranker_model.pkl not found in models directory")
    if not os.path.exists(os.path.join(MODEL_DIR, "meta_model.pkl")):
        raise FileNotFoundError("meta_model.pkl not found in models directory")
    
    return {
        "mean_model": joblib.load(os.path.join(MODEL_DIR, "mean_model.pkl")),
        "quantile_models": joblib.load(os.path.join(MODEL_DIR, "quantile_models.pkl")),
        "ranker_model": joblib.load(os.path.join(MODEL_DIR, "ranker_model.pkl")),
        "meta_model": joblib.load(os.path.join(MODEL_DIR, "meta_model.pkl")),
    }

def load_beta_vector():
    if not os.path.exists(os.path.join(MODEL_DIR, "beta_vector.npy")):
        raise FileNotFoundError("beta_vector.npy not found in models directory")
    
    return np.load(os.path.join(MODEL_DIR, "beta_vector.npy"))