"""
Main script to run ADHD GNN-STAN pipeline
- Supports staged execution (preprocessing → feature extraction → training → evaluation)
- Handles automatic metadata discovery if not already present
"""
import torch
import torch.cuda
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
from preprocessing.preprocess import _process_subject
from utils import run_parallel
from utils import DataDiscovery
from models import GNNSTANHybrid
from training import TrainingOptimizationModule
from evaluation import ADHDModelEvaluator

# --- Configuration ---
RAW_DIR = Path("./data/raw")
PREPROC_OUT = Path("./data/preprocessed")
FEATURES_OUT = Path("./data/features")
TRAINED_OUT = Path("./data/trained")
METADATA_OUT = RAW_DIR / "subjects_metadata.csv"
DEMOGRAPHICS = RAW_DIR / "demographics.csv"
FEATURE_MANIFEST = FEATURES_OUT / "feature_manifest.csv"

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
    'output_dir': TRAINED_OUT / "checkpoints"
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

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


def run_preprocessing(metadata_out: Path, preproc_out: Path, parallel: bool = True, device: torch.device = None):
    """Run preprocessing for all subjects"""
    print("\nRunning Preprocessing...")
    
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA available: Using GPU {torch.cuda.get_device_name()}")
    
    try:
        metadata = pd.read_csv(metadata_out)
        print(f"Loaded metadata for {len(metadata)} subjects")

        # Add output directory to metadata
        metadata['out_dir'] = str(preproc_out)

        # Create output directory
        preproc_out.mkdir(parents=True, exist_ok=True)

        if parallel:
            results = run_parallel(
                tasks=metadata.to_dict('records'),
                worker_fn=lambda row: _process_subject(row),
                max_workers=None  # Will use all available CPU cores
            )
        else:
            # Sequential processing with progress bar
            from tqdm import tqdm
            results = []
            with tqdm(total=len(metadata), desc="Preprocessing", unit="subj") as pbar:
                for _, row in metadata.iterrows():
                    # Show Site next to Subject in the postfix
                    site = row.get("site", "UnknownSite")
                    subject_id = row.get("subject_id", "unknown")
                    pbar.set_postfix_str(f"Site: {site} | Subject: {subject_id}")

                    # Call _process_subject and pass the progress bar if needed
                    result = _process_subject(row, pbar=pbar)
                    results.append(result)
                    pbar.update(1)


        # Print summary
        success = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - success
        print(f"\nPreprocessing complete. Success: {success}, Failed: {failed}")
        
        return results

    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise



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
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="Run preprocessing in parallel")
    parser.add_argument(
        "--no-parallel",
        action="store_false",
        dest="parallel",
        help="Disable parallel processing"
    )
    parser.add_argument("--no-cuda", action="store_true",
                       help="Disable CUDA even if available")
    args = parser.parse_args()

    # Device configuration based on args
    DEVICE = torch.device('cpu') if args.no_cuda else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # Preprocessing call to pass device
    if args.stage in ["preprocessing", "full"]:
        run_preprocessing(METADATA_OUT, PREPROC_OUT, parallel=args.parallel, device=DEVICE)

    if args.stage in ["features", "full"]:
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
