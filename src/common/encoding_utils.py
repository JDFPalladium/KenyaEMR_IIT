import pandas as pd
import json

def fit_encoder(df, categorical_columns, drop_first=True):
    category_map = {}
    for col in categorical_columns:
        cats = df[col].dropna().unique().tolist()
        # preserve order of appearance, not sorted
        if drop_first and len(cats) > 1:
            cats = cats[1:]
        category_map[col] = cats
    return category_map

def save_encoder(category_map, path):
    with open(path, "w") as f:
        json.dump(category_map, f, indent=2)

def load_encoder(path):
    with open(path) as f:
        return json.load(f)

def apply_encoder(df, category_map):
    encoded_parts = []
    for col, cats in category_map.items():
        for cat in cats:
            encoded_parts.append((f"{col}_{cat}", (df[col] == cat).astype(int)))
    enc_df = pd.concat(
        [series.rename(name) for name, series in encoded_parts],
        axis=1
    )
    final_df = pd.concat([df.drop(columns=category_map.keys()), enc_df], axis=1)
    return final_df
