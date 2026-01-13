import pandas as pd
import numpy as np
import xgboost as xgb
import random
import os
import boto3
import io
import pickle
import shutil
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from src.common.feature_dtypes import expected_dtypes
from pathlib import Path
import treelite
import tl2cgen


def rtc_refresh_model(pipeline=False, targets_df=None, targets_aws=None, refresh_date=str):

    # first, read in the processed dataset
    # if pipeline, then dataset is in the pipeline
    # else, it is in the AWS S3 bucket
    if pipeline:
        df = targets_df
    else:
        # Define S3 info
        bucket = "kehmisjan2025"
        # Initialize boto3 client
        s3 = boto3.client("s3")
        buffer = io.BytesIO()
        s3.download_fileobj(bucket, targets_aws, buffer)
        buffer.seek(0)
        df = pd.read_parquet(buffer)

    # set column names to lowercase and strip whitespace
    df.columns = df.columns.str.lower().str.strip()

    # filter to emr in kenyamer and ecare
    df = df[df["emr"].isin(["kenyaemr"])]
    df = df.drop(columns=["emr"])

    # make sure nad is a datetime
    df["nad"] = pd.to_datetime(df["nad"], errors="coerce")
    # keep rows where dayslate is not None or nad is before December 31, 2024 or dayslate is more than  0
    df = df[df["dayslate"].notna() & (df["dayslate"] > 0) & (df["nad"] < "2024-12-31")]

    # create target, if dayslate is less than 120, then 1, else 0
    df['target'] = np.where(df['dayslate'] <= 90, 0, 1)

    # filter to refresh period
    refresh_date = pd.Timestamp(refresh_date)

    # Filter to records from the refresh date and six months before
    # Define the date range to exclude
    after = refresh_date - pd.DateOffset(months=6)
    before = refresh_date
    df = df[(df["nad"] >= after) & (df["nad"] <= before)]
    
    print(df.shape)
    print(df.columns)

    # get each patientpkhash and sitecode and save to file
    df = df.drop(
        columns=[
            "key",
            "visitdate",
            "nad_imputation_flag",
            "sitecode",
            "pregnant_missing",
            "breastfeeding_missing",
            "startartdate",
            "month",
            "dayofweek",
            "timeatfacility",
            "iit",
            'code',
            'county',
            'men_knowledge', 
            'women_knowledge',
            'men_heardaids', 
            'men_highrisksex', 
            'men_highrisksex_multi',
            'men_sexnotwithpartner', 
            'men_sexpartners', 
            'men_nevertested',
            'men_testedrecent', 
            'men_sti', 
            'women_heardaids', 
            'women_highrisksex',
            'women_highrisksex_multi', 
            'women_sexnotwithpartner',
            'women_sexpartners', 
            'women_nevertested', 
            'women_testedrecent',
            'women_sti',  
            'returndate', 
            'dayslate', 
            'rtc',
            'tracingtype', 
            'tracingoutcome', 
            'reasonformissedappt', 
            'attemptnumber',
            'isfinaltrace'
        ]
    )

    # ensure columns are right dtypes
    for col, dtype in expected_dtypes.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    # categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_columns = [
        c for c in df.select_dtypes(include=["object"]).columns
    ]
    print(categorical_columns)
    ohe = OneHotEncoder(drop="first", handle_unknown="ignore")
    ohe.fit(df[categorical_columns])

        # Save the fitted encoder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"data/rtc_ohe_{timestamp}.pkl", "wb") as f:
        pickle.dump(ohe, f)
    # Save the refreshed encoder as latest to be used in inference
    shutil.copyfile(f"data/rtc_ohe_{timestamp}.pkl", "data/rtc_ohe_latest.pkl")

    def encode_xgboost(df, start_date, end_date, save_feature_order):

        # Filter the DataFrame to include only the rows within the specified date range
        # slice by date
        mask = (df["nad"] >= start_date) & (df["nad"] <= end_date)
        df_slice = df.loc[mask].copy()

        # drop non-feature cols before encoding
        df_slice = df_slice.drop(columns=["nad"])

        # one-hot encode categorical cols (may be empty)
        if categorical_columns:
            encoded = ohe.transform(df_slice[categorical_columns]).toarray()
            encoded_cols = ohe.get_feature_names_out(categorical_columns)
            enc_df = pd.DataFrame(encoded, columns=encoded_cols, index=df_slice.index)
            final_df = pd.concat([df_slice.drop(columns=categorical_columns), enc_df], axis=1)
        else:
            final_df = df_slice

        # Extract feature order (excluding target)
        feature_order = [col for col in final_df.columns if col != "target"]
        if save_feature_order:
            with open("data/rtc_feature_order.pkl", "wb") as f:
                pickle.dump(feature_order, f)

        # convert to xgb.Dmatrix
        xgb_df = xgb.DMatrix(data=final_df.drop(columns=["target"]), label=final_df["target"])

        return xgb_df

    # encoded dataset
    dtrain = encode_xgboost(
        df, start_date=after, end_date=refresh_date - pd.DateOffset(months=1), save_feature_order=True
    )
    dval = encode_xgboost(
        df, start_date=refresh_date - pd.DateOffset(months=1), end_date=refresh_date, save_feature_order=False
    )

    params = {
        "eta": 0.01,
        "max_depth": 6,
        "subsample": 0.5,
        "colsample_bytree": 0.6,
        "lambda": 1,
        "scale_pos_weight": 10,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
    }

    random.seed(42)
    gb_model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=3000,
        evals=[(dtrain, "train"), (dval, "eval")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )

    # After training with xgb.train(...)
    gb_model.save_model(f"data/rtc_mod_{timestamp}.json")
    shutil.copyfile(f"data/rtc_mod_{timestamp}.json", "data/rtc_mod_latest.json")

    MODEL_JSON = Path("data/rtc_mod_latest.json")   # your existing XGBoost JSON
    OUT_LIB    = Path("data/rtc_mod_latest.so")     # .dll on Windows, .dylib on macOS

    # 1) Load XGBoost model into Treelite
    #    (Treelite 4.x is the model exchange/serialization layer)
    model = treelite.frontend.load_xgboost_model(str(MODEL_JSON))

    # 2) Export a compiled shared library with TL2cgen
    #    Use 'gcc' on Linux, 'clang' on macOS, 'msvc' on Windows
    tl2cgen.export_lib(
        model,
        toolchain="gcc",
        libpath=str(OUT_LIB),
        params={"parallel_comp": 1},  # compile in parallel; tweak for your cores
        verbose=True,
    )

    # Generate predictions on the validation set
    preds = gb_model.predict(dval)
    
    # get the 25th percentile of the predictions
    threshold_high = pd.Series(preds).quantile(0.75)
    threshold_medium = pd.Series(preds).quantile(0.5)
    print(f"Thresholds: high={threshold_high}, medium={threshold_medium}")
    # combine thresholds into a dictionary
    thresholds = {
        "high": threshold_high,
        "medium": threshold_medium,
    }   
    # save thresholds to a file with timestamp and as latest
    with open(f"data/rtc_thresholds_{timestamp}.pkl", "wb") as f:
        pickle.dump(thresholds, f)
    shutil.copyfile(f"data/rtc_thresholds_{timestamp}.pkl", "data/rtc_thresholds_latest.pkl")


if __name__ == "__main__":
    rtc_refresh_model(
        pipeline=False,
        targets_aws="rtc1001.parquet",
        refresh_date="2024-09-30"
    )