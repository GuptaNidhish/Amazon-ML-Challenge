import os
import numpy as np
import lightgbm as lgb
from typing import Dict, Optional, Tuple
from config import LGBM_PARAMS, MODEL_DIR
from features import FEATURE_NAMES

def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    custom_params: Optional[Dict] = None,
    model_save_path: Optional[str] = None
) -> Tuple[lgb.Booster, Dict[str, float]]:
    """
    Trains a LightGBM GBDT classifier on pairwise features.
    """
    params = dict(LGBM_PARAMS)
    if custom_params:
        params.update(custom_params)

    # Balanced / tuned pos weight based on ratio
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    if num_pos > 0 and 'scale_pos_weight' not in params:
        # Precision heavy: keep scale_pos_weight conservative (e.g. 1.0 to 1.5)
        params['scale_pos_weight'] = 1.0

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    valid_sets = [dtrain]
    valid_names = ['train']
    
    if X_val is not None and y_val is not None:
        dval = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=dtrain)
        valid_sets.append(dval)
        valid_names.append('val')

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100)
    ]

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=params.get('n_estimators', 1000),
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks
    )

    if model_save_path is None:
        model_save_path = os.path.join(MODEL_DIR, 'lgbm_model.txt')
    booster.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Feature importance
    importance = booster.feature_importance(importance_type='gain')
    feat_imp = {feat: float(gain) for feat, gain in zip(FEATURE_NAMES, importance)}
    sorted_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

    return booster, sorted_imp


def predict_proba(booster: lgb.Booster, X: np.ndarray) -> np.ndarray:
    """
    Returns array of P(match) probabilities.
    """
    if len(X) == 0:
        return np.array([], dtype=np.float32)
    return booster.predict(X)
