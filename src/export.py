import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as rt
import json

FEATURE_COLS = [
    "length", "entropy", "digit_ratio", "vowel_ratio",
    "unique_char_ratio", "longest_consonant_run", "dot_count",
    "has_hyphen", "ngram_score", "tld_freq"
]

def _fix_base_values(onnx_model):
    for node in onnx_model.graph.node:
        if node.op_type != "TreeEnsembleClassifier":
            continue
        n_classes = next(
            (len(a.ints) for a in node.attribute if a.name == "classlabels_int64s"),
            None,
        )
        for attr in node.attribute:
            if attr.name != "base_values":
                continue
            if n_classes is None or len(attr.floats) == n_classes:
                return  # already correct, nothing to do
            original = list(attr.floats)
            padded = original + [0.0] * (n_classes - len(original))
            attr.floats[:] = padded
            print(f"Patched base_values: {original} -> {padded}")
            return


def export(
    features_path="../data/processed/features.csv",
    model_path="../models/classifier.json",
    output_path="../models/classifier.onnx"
):
    print("Loading model and data...")
    model = XGBClassifier()
    model.load_model(model_path)

    df = pd.read_csv(features_path)
    X = df[FEATURE_COLS].astype(np.float32)

    print("Converting to ONNX...")
    booster = model.get_booster()
    original_names = booster.feature_names
    booster.feature_names = [f"f{i}" for i in range(len(FEATURE_COLS))]

    initial_type = [("float_input", FloatTensorType([None, len(FEATURE_COLS)]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type)

    booster.feature_names = original_names  # restore

    _fix_base_values(onnx_model)

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Saved to {output_path}")

    with open("../models/feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f)
    print("Saved feature column order to models/feature_cols.json")

    print("\nVerifying ONNX output matches original model...")
    X_sample = X.sample(500, random_state=42).values.astype(np.float32)

    original_proba = model.predict_proba(
        pd.DataFrame(X_sample, columns=FEATURE_COLS)
    )[:, 1]

    sess = rt.InferenceSession(output_path)
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: X_sample})
    onnx_proba = np.array([p[0] for p in onnx_out[1]])

    max_diff = np.abs(original_proba - onnx_proba).max()
    mean_diff = np.abs(original_proba - onnx_proba).mean()
    print(f"Max probability difference:  {max_diff:.6f}")
    print(f"Mean probability difference: {mean_diff:.6f}")

    if max_diff < 0.01:
        print("ONNX model matches original within acceptable tolerance")
    else:
        print("WARNING: significant difference detected")

if __name__ == "__main__":
    export()