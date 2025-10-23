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
    'hidden_dim': 128,
    'num_classes': 2,
    'num_heads': 4,
    'dropout': 0.3,
    'gnn': {
        'hidden_dims': [128, 64, 32],
        'dropout': 0.3,
        'pool_ratios': [0.8, 0.6]
    },
    'stan': {
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.3
    },
    'fusion': {
        'fusion_dim': 128,
        'dropout': 0.3
    },
    'classifier_dropout': 0.5
}

TRAINING_CONFIG = {
    'batch_size': 4,
    'learning_rate': 1e-3,
    'optimizer': 'adam',
    'weight_decay': 1e-5,
    'epochs': 50,
    'early_stopping_patience': 10,
    'early_stopping_min_delta': 0.001,
    'gradient_clip': 1.0,
    'num_workers': 0,
    'use_focal_loss': True,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'run_loso': True,
    'output_dir': str(TRAINED_OUT / "checkpoints"),

    # Memory optimization settings
    'use_amp': True,
    'use_gradient_checkpointing': True,
    'gradient_accumulation_steps': 4
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
    """Run preprocessing for all subjects in small batches"""
    print("\nRunning Preprocessing...")
    
    try:
        metadata = pd.read_csv(metadata_out)
        print(f"Loaded metadata for {len(metadata)} subjects")

        # Process in small batches to avoid memory overload
        batch_size = 4 if parallel else 1
        all_results = []
        
        for i in range(0, len(metadata), batch_size):
            batch = metadata.iloc[i:i+batch_size].copy()
            batch['device'] = 'cpu'  # Force CPU
            batch['out_dir'] = str(preproc_out)
            
            print(f"\nProcessing batch {i//batch_size + 1}/{(len(metadata)-1)//batch_size + 1}")
            
            if parallel and len(batch) > 1:
                results = run_parallel(
                    func=_process_subject,
                    items=batch.to_dict('records'),
                    max_workers=2,  # Maximum 2 workers
                    desc=f"Batch {i//batch_size + 1}"
                )
            else:
                # Sequential for small batches
                results = []
                for _, row in batch.iterrows():
                    result = _process_subject(row)
                    results.append(result)
                    
                    # Force cleanup between subjects
                    import gc
                    gc.collect()
            
            all_results.extend(results)
            
            # Pause between batches to let system recover
            import time
            time.sleep(2)
        
        # Print summary
        success = sum(1 for r in all_results if r["status"] == "success")
        failed = len(all_results) - success
        print(f"\nPreprocessing complete. Success: {success}, Failed: {failed}")
        
        return all_results
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise


def run_feature_extraction(metadata_out: Path, preproc_out: Path, 
                         feature_out_dir: Path, parallel: bool = True):
    """Run feature extraction stage of the pipeline"""
    print("\nRunning Feature Extraction...")
    
    try:
        # Initialize parcellation
        parcellation = SchaeferParcellation()
        parcellation.load_parcellation()
        
        # Run feature extraction
        results = run_feature_extraction_stage(
            metadata_csv=metadata_out,
            preproc_dir=preproc_out,
            feature_out_dir=feature_out_dir,
            atlas_labels=parcellation.roi_labels,
            parallel=parallel
        )
        
        # Create feature manifest for training
        from feature_extraction import create_feature_manifest
        metadata = pd.read_csv(metadata_out)
        manifest_path = create_feature_manifest(feature_out_dir, metadata)
        print(f"Created feature manifest at {manifest_path}")
        
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
    
    # Initialize trainer
    trainer = TrainingOptimizationModule(
        model_config=model_config,
        training_config=training_config,
        device=device
    )
    
    # Create output directory
    output_dir = Path(training_config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run cross-validation training
    print("\n" + "="*70)
    print("Running Cross-Validation Training")
    print("="*70)
    cv_results = trainer.run_cv_training(
        feature_data=feature_data,
        cv_splits=splits['cv_splits'],
        save_dir=output_dir / 'cv'
    )
    
    # Run LOSO validation if requested AND if splits exist
    if training_config.get('run_loso', True) and 'loso_splits' in splits:
        # Check if LOSO splits are actually available
        if len(splits.get('loso_splits', [])) > 0:
            print("\n" + "="*70)
            print("Running Leave-One-Site-Out Validation")
            print("="*70)
            loso_results = trainer.run_loso_training(
                feature_data=feature_data,
                loso_splits=splits['loso_splits'],
                save_dir=output_dir / 'loso'
            )
        else:
            print("\n" + "="*70)
            print("LOSO Validation Skipped - Only 1 site detected")
            print("="*70)
            loso_results = None
    
    print("\nTraining complete.")
    
    # Final evaluation on test set
    print("\n" + "="*70)
    print("Final Test Set Evaluation")
    print("="*70)
    
    test_data = feature_data.iloc[splits['test_idx']]
    
    # Load best model
    if trainer.best_model_path and trainer.best_model_path.exists():
        checkpoint = torch.load(trainer.best_model_path)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from {trainer.best_model_path}")
    
    # Evaluate
    from torch.utils.data import DataLoader
    from training.dataset import ADHDDataset
    
    test_dataset = ADHDDataset(test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.get('batch_size', 32),
        shuffle=False,
        num_workers=training_config.get('num_workers', 4)
    )
    
    _, test_acc, test_metrics = trainer.validate(test_loader)
    
    print(f"\nTest Set Results:")
    print(f"Accuracy: {test_acc:.2f}%")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-Score: {test_metrics['f1']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")
    
    # Save final results
    import json
    final_results = {
        'cv_results': cv_results,
        'test_metrics': test_metrics,
        'test_accuracy': test_acc,
        'model_config': model_config,
        'training_config': training_config
    }
    
    results_path = output_dir / 'final_results.json'
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_path}")
    
    return final_results
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
    for dir_path in [PREPROC_OUT, FEATURES_OUT, TRAINED_OUT, SPLITS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Ensure metadata exists
    ensure_metadata(RAW_DIR, METADATA_OUT)

    # Run pipeline stages
    try:
        # If a feature manifest already exists, treat "full" as skipping preprocessing+feature extraction.
        manifest_exists = FEATURE_MANIFEST.exists()

        # Preprocessing: run if explicitly requested, or if "full" and no manifest exists
        if args.stage == "preprocessing" or (args.stage == "full" and not manifest_exists):
            run_preprocessing(METADATA_OUT, PREPROC_OUT, parallel=args.parallel, device=DEVICE)
        else:
            print("Skipping preprocessing (feature manifest found). To force, run --stage preprocessing or delete the manifest.")

        # Feature extraction: run if explicitly requested, or if "full" and not manifest_exists
        if args.stage == "features" or (args.stage == "full" and not manifest_exists):
            run_feature_extraction(
                metadata_out=METADATA_OUT,
                preproc_out=PREPROC_OUT,
                feature_out_dir=FEATURES_OUT,
                parallel=args.parallel
            )
        else:
            print("Skipping feature extraction (feature manifest found). To force, run --stage features or delete the manifest.")
 
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
