from datetime import datetime
import random
import io
import os
import pandas as pd
import pickle
import tl2cgen
import numpy as np
from src.common.feature_dtypes import expected_dtypes

def rtc_gen_inference(df, sitecode):

    if df is None or df.empty:
        return {"pred_out": None, "pred_cat": "unavailable"}

    # make sure nad is a datetime
    df["nad"] = pd.to_datetime(df["nad"], format="%Y-%m-%d")
    # make sure data is sorted by nad in descending order
    df = df.sort_values(by="nad", ascending=False)

    df = df.drop(
        columns=[
            "key",
            "emr",
            "visitdate",
            "nad_imputation_flag",
            "sitecode",
            "pregnant_missing",
            "nad",
            "breastfeeding_missing",
            "startartdate",
            "month",
            "dayofweek",
            "timeatfacility"
            # "txcurr",
            # "rolling_weighted_noshow",
            # "rolling_weighted_dayslate"
        ]
    )

    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # ensure columns are right dtypes
    for col, dtype in expected_dtypes.items():
        if col in df.columns:
            if dtype in [float, "float", "float64", int, "int", "int64"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = df[col].astype(dtype)

    # load encoder which is called ohe_latest.pkl
    # from the models directory
    encoder = "data/rtc_ohe_latest.pkl"
    # Check if the encoder file exists
    if not os.path.exists(encoder):
        raise FileNotFoundError(
            f"Encoder file {encoder} not found. Please train the model first."
        )
    with open(encoder, "rb") as f:
        ohe = pickle.load(f)

    # encode categorical columns
    # Get the categorical columns from the DataFrame
    # Note: This assumes that the categorical columns are the same as those used during training
    # If the columns are different, you may need to adjust this part
    # to match the training columns
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    # One-hot encode the categorical columns
    try:
        encoded_features = ohe.transform(df[categorical_columns]).toarray()
    except Exception as e:
        print(f"OneHotEncoding failed: {e}")
        return {"pred_out": None, "pred_cat": "unavailable"}
    encoded_feature_names = ohe.get_feature_names_out(categorical_columns)

    # Create a DataFrame with the encoded features
    encoded_df = pd.DataFrame(
        encoded_features, columns=encoded_feature_names, index=df.index
    )

    # Concatenate the encoded features with the original DataFrame
    final_df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)

    # make sure the columns are in the right order
    with open("data/rtc_feature_order.pkl", "rb") as f:
        feature_order = pickle.load(f)
    try:
        final_df = final_df[feature_order]
    except KeyError as e:
        print(f"❌ Feature mismatch: some expected columns are missing: {e}")
        return {"pred_out": None, "pred_cat": "unavailable"}

    # --- new (TL2cgen inference) ---
    features = final_df.astype(np.float32)
    tl_model_path = "data/rtc_mod_latest.so"

    # Load the compiled predictor once (optional: cache globally)
    predictor = tl2cgen.Predictor(tl_model_path)

    # Predict probabilities (pred_margin=False ensures sigmoid applied)
    preds = predictor.predict(
        tl2cgen.DMatrix(features.values),
        pred_margin=False
    )

    pred_out = float(preds[0])  # extract scalar


    # load thresholds from models/thresholds.pkl
    thresholds_file = "data/rtc_thresholds_latest.pkl"
    if not os.path.exists(thresholds_file):
        raise FileNotFoundError(
            f"Thresholds file {thresholds_file} not found. Please train the model first."
        )
    with open(thresholds_file, "rb") as f:
        thresholds = pickle.load(f)

    # apply site-specific thresholds to pred_cat
    if pred_out > thresholds["high"]:
        pred_cat = "high"
    elif pred_out > thresholds["medium"]:
        pred_cat = "medium"
    else:
        pred_cat = "low"

    # return pred_out and pred_cat
    pred_out = {
        "pred_out": pred_out,
        "pred_cat": pred_cat,
        "evaluation_date": datetime.now().strftime("%Y-%m-%d"),
    }
    return pred_out
