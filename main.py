"""
Main script to run ADHD GNN-STAN pipeline
- Supports staged execution (preprocessing → feature extraction → training)
"""

import argparse
import torch
from pathlib import Path
import subprocess

from utils import ADHDDataLoader
from models import GNNSTANHybrid
from evaluation import Evaluator


def run_preprocessing(metadata_out: Path, preproc_out: Path):
    print(">>> Running Preprocessing...")
    subprocess.run([
        "python", "-m", "preprocessing.adhd_preprocessing_pipeline",
        "--metadata", str(metadata_out),
        "--out-dir", str(preproc_out)
    ], check=True)


def run_feature_extraction(preproc_out: Path, features_out: Path):
    print(">>> Running Feature Extraction...")
    subprocess.run([
        "python", "-m", "feature_extraction.parcellation_and_feature_extraction",
        "--preproc-dir", str(preproc_out),
        "--atlas", "schaefer200",
        "--out-dir", str(features_out)
    ], check=True)


def run_training(feature_manifest: Path, demographics: Path, model_config: dict, pretrained_weights: Path = None):
    print(">>> Loading data for training...")
    data_loader = ADHDDataLoader(feature_manifest, demographics)
    data = data_loader.load_data_for_training()

    fc_matrices = data['fc_matrices']
    roi_timeseries = data['roi_timeseries']
    labels = data['labels']
    sites = data['sites']

    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GNNSTANHybrid(model_config).to(device)

    if pretrained_weights:
        model.load_state_dict(torch.load(pretrained_weights, map_location=device))

    # Evaluate
    evaluator = Evaluator(model, device=device)
    metrics = evaluator.evaluate(fc_matrices, roi_timeseries, labels, sites)

    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        if isinstance(v, (float, int)):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ADHD GNN-STAN pipeline")
    parser.add_argument("--stage", type=str,
                        choices=["preprocessing", "features", "full"],
                        default="full",
                        help="Which stage of the pipeline to run")
    args = parser.parse_args()

    # --- Configuration ---
    DATA_DIR = Path("./data")
    METADATA_OUT = DATA_DIR / "subjects_metadata.csv"
    PREPROC_OUT = DATA_DIR / "preprocessed"
    FEATURES_OUT = DATA_DIR / "features"
    FEATURE_MANIFEST = FEATURES_OUT / "feature_manifest.csv"
    DEMOGRAPHICS = DATA_DIR / "demographics.csv"

    MODEL_CONFIG = {
        'n_rois': 200,
        'n_classes': 2,
        'gnn': {'hidden_dims': [128, 64, 32], 'dropout': 0.3, 'pool_ratios': [0.8, 0.6]},
        'stan': {'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.3},
        'fusion': {'fusion_dim': 128, 'dropout': 0.3},
        'classifier_dropout': 0.5
    }
    PRETRAINED_WEIGHTS = None

    # --- Run selected stage ---
    if args.stage in ["preprocessing", "features", "full"]:
        run_preprocessing(METADATA_OUT, PREPROC_OUT)

    if args.stage in ["features", "full"]:
        run_feature_extraction(PREPROC_OUT, FEATURES_OUT)

    if args.stage == "full":
        run_training(FEATURE_MANIFEST, DEMOGRAPHICS, MODEL_CONFIG, PRETRAINED_WEIGHTS)
