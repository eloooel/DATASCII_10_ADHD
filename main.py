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
from training import DataSplitter
from feature_extraction import SchaeferParcellation, run_feature_extraction_stage
from preprocessing.preprocess import _process_subject
from utils import run_parallel
from utils import DataDiscovery
from models import GNNSTANHybrid
from training import TrainingOptimizationModule
from evaluation import ADHDModelEvaluator
from typing import Dict, Any, List

# --- Configuration ---
RAW_DIR = Path("./data/raw")
PREPROC_OUT = Path("./data/preprocessed")
PARCELLATED_OUT = Path("./data/parcellated")
FEATURES_OUT = Path("./data/features")
TRAINED_OUT = Path("./data/trained")
METADATA_OUT = RAW_DIR / "subjects_metadata.csv"
DEMOGRAPHICS = RAW_DIR / "demographics.csv"
FEATURE_MANIFEST = FEATURES_OUT / "feature_manifest.csv"
SPLITS_DIR = Path("./data/splits")

SPLIT_CONFIG = {
    'train_size': 0.8,
    'n_splits': 5,
    'random_state': 42,
    'stratify': True
}

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

    try:
        metadata = pd.read_csv(metadata_out)
        print(f"Loaded metadata for {len(metadata)} subjects")

        # Add device and output directory to metadata
        metadata['device'] = str(device)
        metadata['out_dir'] = str(preproc_out)

        if parallel:
            results = run_parallel(
                tasks=metadata.to_dict('records'),
                worker_fn=_process_subject,
                max_workers=None
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


def run_feature_extraction(metadata_out: Path, preproc_out: Path, 
                         parcellated_out: Path, feature_out_dir: Path, 
                         parallel: bool = True):
    """Run feature extraction stage of the pipeline"""
    print("\nRunning Feature Extraction...")
    
    try:
        # Initialize parcellation
        parcellation = SchaeferParcellation()
        parcellation.load_parcellation()
        
        # Run feature extraction with parcellation step
        results = run_feature_extraction_stage(
            metadata_csv=metadata_out,
            preproc_dir=preproc_out,
            parcellated_dir=parcellated_out,  # Add intermediate directory
            feature_out_dir=feature_out_dir,
            atlas_labels=parcellation.roi_labels,
            parallel=parallel
        )
        
        # Print summary
        success = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - success
        print(f"\nFeature extraction complete. Success: {success}, Failed: {failed}")
        
        return results
        
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        raise


def run_splitting(
    feature_manifest: Path, 
    splits_dir: Path, 
    split_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Create dataset splits for training"""
    try:
        print("\nCreating dataset splits...")
        
        splitter = DataSplitter(
            train_size=split_config['train_size'],
            n_splits=split_config['n_splits'],
            random_state=split_config['random_state'],
            stratify=split_config['stratify']
        )
        
        splits = splitter.split_dataset(
            features_path=feature_manifest,
            splits_dir=splits_dir
        )
        
        return splits
    except Exception as e:
        print(f"Error in splitting: {str(e)}")
        raise

def run_training(feature_manifest: Path, demographics: Path, 
                model_config: dict, training_config: dict,
                splits_path: Path, device: torch.device = None):
    """Run training and evaluation with the specified configuration"""
    if not feature_manifest.exists():
        raise FileNotFoundError(f"Feature manifest not found: {feature_manifest}")
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_path}")

    # Use global DEVICE if none provided
    device = device or DEVICE

    print("\nLoading data for training...")
    
    # Load splits
    splitter = DataSplitter()
    splits = splitter.load_splits(splits_path)
    
    # Load feature data
    feature_data = pd.read_csv(feature_manifest)
    
    # Get train/test sets using splits
    train_data = feature_data.iloc[splits['train_idx']]
    test_data = feature_data.iloc[splits['test_idx']]
    
    # Initialize trainer with splits
    trainer = TrainingOptimizationModule(
        model_config=model_config,
        training_config=training_config
    )
    
    # Run training with cross-validation splits
    results = trainer.run_training(
        train_data=train_data,
        test_data=test_data,
        cv_splits=splits['cv_splits']
    )
    
    print("\nTraining complete.")

    # Evaluate trained model
    evaluator = ADHDModelEvaluator(
        model_path=trainer.best_model_path,
        model_config=model_config,
        device=device
    )
    
    # Evaluate using the test data
    metrics = evaluator.evaluate(
        fc_matrices=test_data['fc_matrices'].values,
        roi_timeseries=test_data['timeseries'].values,
        labels=test_data['diagnosis'].values,
        sites=test_data.get('site', None)
    )

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
                       choices=["preprocessing", "features", "split", "training", "full"],
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
    if DEVICE.type == 'cuda':
        print(f"Using device: {DEVICE} ({torch.cuda.get_device_name()})")
    else:
        print(f"Using device: {DEVICE}")

    # Create necessary directories
    for dir_path in [PREPROC_OUT, PARCELLATED_OUT, FEATURES_OUT, TRAINED_OUT, SPLITS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Ensure metadata exists
    ensure_metadata(RAW_DIR, METADATA_OUT)

    # Run pipeline stages
    try:
        if args.stage in ["preprocessing", "full"]:
            run_preprocessing(METADATA_OUT, PREPROC_OUT, parallel=args.parallel, device=DEVICE)

        if args.stage in ["features", "full"]:
            run_feature_extraction(
                metadata_out=METADATA_OUT,
                preproc_out=PREPROC_OUT,
                parcellated_out=PARCELLATED_OUT,
                feature_out_dir=FEATURES_OUT,
                parallel=args.parallel
            )

        if args.stage in ["split", "training", "full"]:
            splits = run_splitting(FEATURE_MANIFEST, SPLITS_DIR, SPLIT_CONFIG)

        if args.stage in ["training", "full"]:
            run_training(
                feature_manifest=FEATURE_MANIFEST,
                demographics=DEMOGRAPHICS,
                model_config=MODEL_CONFIG,
                training_config=TRAINING_CONFIG,
                splits_path=SPLITS_DIR / "splits.json",
                device=DEVICE
            )

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)
