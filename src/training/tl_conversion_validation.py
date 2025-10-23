import numpy as np
import pandas as pd
import xgboost as xgb
import tl2cgen
import pickle

XGB_MODEL_PATH = "data/mod_latest.json"
TL2_MODEL_PATH = "data/mod_latest.so"
FEATURE_ORDER_PATH = "data/feature_order.pkl"

with open(FEATURE_ORDER_PATH, "rb") as f:
    feature_order = pickle.load(f)

# If your saved list accidentally contains the label, drop it
if "iit" in feature_order:
    feature_order.remove("iit")

n_samples = 100
X_df = pd.DataFrame(
    np.random.rand(n_samples, len(feature_order)).astype(np.float32),
    columns=feature_order
)

# XGBoost probabilities (binary:logistic applies sigmoid internally)
bst = xgb.Booster()
bst.load_model(XGB_MODEL_PATH)
xgb_probs = bst.predict(xgb.DMatrix(X_df, feature_names=feature_order))

pred = tl2cgen.Predictor(TL2_MODEL_PATH)

def report(name, a, b):
    d = np.abs(a - b)
    print(f"{name}\n  max diff: {d.max():.6e} | mean diff: {d.mean():.6e}\n")

# --- Case A: TL2cgen probabilities (pred_margin=False) ---
probs_tl = pred.predict(tl2cgen.DMatrix(X_df.values.astype(np.float32)),
                        pred_margin=False)  # <-- explicit
probs_tl = np.asarray(probs_tl).reshape(-1)  # ensure 1D
report("XGB prob  vs TL2 prob", xgb_probs, probs_tl)

# --- Case B: TL2cgen logits (pred_margin=True) + manual sigmoid ---
logits_tl = pred.predict(tl2cgen.DMatrix(X_df.values.astype(np.float32)),
                         pred_margin=True)  # <-- raw margins
logits_tl = np.asarray(logits_tl).reshape(-1)
probs_from_logits = 1.0 / (1.0 + np.exp(-logits_tl))
report("XGB prob  vs TL2 sigmoid(logits)", xgb_probs, probs_from_logits)

# --- Optional: include some NaNs to confirm equivalence with missing routing ---
X_nan = X_df.copy()
X_nan.iloc[::10, ::4] = np.nan
xgb_probs_nan = bst.predict(xgb.DMatrix(X_nan, feature_names=feature_order))
probs_tl_nan = pred.predict(tl2cgen.DMatrix(X_nan.values.astype(np.float32)),
                            pred_margin=False)
probs_tl_nan = np.asarray(probs_tl_nan).reshape(-1)
report("XGB prob  vs TL2 prob (with NaNs)", xgb_probs_nan, probs_tl_nan)
