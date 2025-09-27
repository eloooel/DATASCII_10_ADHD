"""
Main script to run ADHD GNN-STAN pipeline
- Supports staged execution (preprocessing → feature extraction → training → evaluation)
- Handles automatic metadata discovery if not already present
"""

import sys
import argparse
from pathlib import Path
import subprocess
import torch
from tqdm import tqdm

import pandas as pd
import numpy as np
from preprocessing import PreprocessingPipeline

from feature_extraction import SchaeferParcellation, run_feature_extraction_stage
from utils import run_parallel
from utils import DataDiscovery
from models import GNNSTANHybrid
from training import TrainingOptimizationModule
from evaluation import ADHDModelEvaluator

# --- Configuration ---
DATA_DIR = Path("./data/raw")
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

TRAINING_CONFIG = {
    'batch_size': 16,
    'lr': 1e-3,
    'epochs': 50,
    'patience': 10,
    'output_dir': DATA_DIR / "checkpoints"
}


# --- Pipeline stages ---
def ensure_metadata(data_dir: Path, metadata_out: Path):
    """Ensure the subjects_metadata.csv exists, otherwise generate it"""
    if not metadata_out.exists():
        print("Generating subject metadata CSV...")
        discovery = DataDiscovery(data_dir)
        subjects = discovery.discover_subjects()
        discovery.save_metadata(subjects, metadata_out)
    else:
        print(f"Found existing metadata CSV at {metadata_out}")


def run_preprocessing(metadata_out: Path, preproc_out: Path, parallel: bool = True):
    """Run preprocessing for all subjects"""
    print("\nRunning Preprocessing...")
    
    # Load metadata
    try:
        metadata = pd.read_csv(metadata_out)
        print(f"Loaded metadata for {len(metadata)} subjects")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return
    
    # Create output directory
    preproc_out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {preproc_out}")

    # Initialize pipeline
    pipeline = PreprocessingPipeline()

    def worker(row):
        try:
            subject_id = row["subject_id"]
            func_path = row["input_path"]
            
            result = pipeline.process(func_path, subject_id)
            subj_out = preproc_out / subject_id
            subj_out.mkdir(parents=True, exist_ok=True)

            if result["status"] == "success":
                np.save(subj_out / "func_preproc.npy", result["processed_data"].get_fdata())
                np.save(subj_out / "mask.npy", result["brain_mask"])
                np.save(subj_out / "confounds.npy", result["confound_regressors"])
                return {"status": "success", "subject_id": subject_id}
            else:
                return {"status": "failed", "subject_id": subject_id, "error": result.get("error")}
        except Exception as e:
            return {"status": "failed", "subject_id": row.get("subject_id", "unknown"), "error": str(e)}

    print(f"\nStarting preprocessing for {len(metadata)} subjects...\n")

    # Fixed tqdm usage - use postfix for dynamic updates
    results = []
    
    with tqdm(total=len(metadata), desc="Preprocessing", unit="subject", 
              dynamic_ncols=True, leave=True) as pbar:
        
        for _, row in metadata.iterrows():
            subject_id = row["subject_id"]
            
            # Update the postfix to show current subject
            pbar.set_postfix_str(f"Current: {subject_id}")
            
            result = worker(row)
            results.append(result)
            
            # Update progress bar
            pbar.update(1)

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - success
    print(f"\nPreprocessing complete. Success: {success}, Failed: {failed}")



def run_feature_extraction(preproc_out: Path, features_out: Path):
    print("\nRunning Feature Extraction...")
    subprocess.run([
        sys.executable,
        "-m", "feature_extraction.parcellation_and_feature_extraction",
        "--preproc-dir", str(preproc_out),
        "--atlas", "schaefer200",
        "--out-dir", str(features_out)
    ], check=True)


def run_training(feature_manifest: Path, demographics: Path, model_config: dict, training_config: dict):
    print("\nLoading data for training...")
    # Load feature data
    from utils import load_metadata
    feature_data = load_metadata(feature_manifest)

    fc_matrices = feature_data['fc_matrices']
    roi_timeseries = feature_data['roi_timeseries']
    labels = feature_data['labels']
    sites = feature_data['sites']

    # Initialize trainer
    trainer = TrainingOptimizationModule(model_config, training_config)
    results = trainer.run_training(fc_matrices, roi_timeseries, labels, sites)
    print("\nTraining complete.")

    # Evaluate trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GNNSTANHybrid(model_config).to(device)
    if training_config.get('pretrained_weights'):
        model.load_state_dict(torch.load(training_config['pretrained_weights'], map_location=device))

    evaluator = ADHDModelEvaluator(model, device=device)
    metrics = evaluator.evaluate(fc_matrices, roi_timeseries, labels, sites)

    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        if isinstance(v, (float, int)):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


# --- Main execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ADHD GNN-STAN pipeline")
    parser.add_argument("--stage", type=str,
                        choices=["preprocessing", "features", "training", "full"],
                        default="full",
                        help="Which stage of the pipeline to run")
    parser.add_argument("--parallel", action="store_true", help="Run preprocessing in parallel")
    args = parser.parse_args()

    # Ensure metadata
    ensure_metadata(DATA_DIR, METADATA_OUT)

    # Run pipeline stages based on selection
    if args.stage in ["preprocessing", "features", "training", "full"]:
        run_preprocessing(METADATA_OUT, PREPROC_OUT, parallel=args.parallel)

    if args.stage in ["features", "training", "full"]:
        run_feature_extraction(PREPROC_OUT, FEATURES_OUT)

    if args.stage in ["training", "full"]:
        run_training(FEATURE_MANIFEST, DEMOGRAPHICS, MODEL_CONFIG, TRAINING_CONFIG)
    parser = argparse.ArgumentParser(description="Run ADHD GNN-STAN pipeline")

    if args.stage in ["preprocessing", "features", "training", "full"]:
        run_preprocessing(METADATA_OUT, PREPROC_OUT, parallel=args.parallel)

    parser.add_argument("--stage", type=str,
                        choices=["preprocessing", "features", "training", "full"],
                        default="full",
                        help="Which stage of the pipeline to run")
    parser.add_argument("--parallel", action="store_true", default=True,
                    help="Run preprocessing and feature extraction in parallel by default")
    args = parser.parse_args()

    # Ensure metadata is available
    ensure_metadata(DATA_DIR, METADATA_OUT)

    # Run pipeline stages based on user selection
    if args.stage in ["preprocessing", "features", "training", "full"]:
        run_preprocessing(METADATA_OUT, PREPROC_OUT)

    if args.stage in ["features", "training", "full"]:
        parcellation = SchaeferParcellation()
        parcellation.load_parcellation()

        run_feature_extraction_stage(
            metadata_csv=METADATA_OUT,
            preproc_dir=PREPROC_OUT,
            feature_out_dir=FEATURES_OUT,
            atlas_labels=parcellation.roi_labels,
            parallel=args.parallel
        )

    if args.stage in ["training", "full"]:
        run_training(FEATURE_MANIFEST, DEMOGRAPHICS, MODEL_CONFIG, TRAINING_CONFIG)
