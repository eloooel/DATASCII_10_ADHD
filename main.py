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
import argparse
import sys
import json
import pandas as pd
import torch
from pathlib import Path
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
    'batch_size': 2,
    'learning_rate': 1e-3,
    'optimizer': 'adam',
    'weight_decay': 1e-5,
    'epochs': 50,
    'early_stopping_patience': 10,
    'early_stopping_min_delta': 0.001,
    'gradient_clip': 1.0,
    'num_workers': 2,
    'use_focal_loss': True,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'run_loso': True,
    'output_dir': str(TRAINED_OUT / "checkpoints"),

    # Memory optimization settings
    'use_amp': True,
    'use_gradient_checkpointing': True,
    'gradient_accumulation_steps': 8
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

        batch_size = TRAINING_CONFIG.get('batch_size', 8) if parallel else 1
        all_results = []
        total_batches = (len(metadata) - 1) // batch_size + 1
        
        # Single overview progress bar
        with tqdm(total=len(metadata), desc="Preprocessing", unit="subj") as pbar:
            for i in range(0, len(metadata), batch_size):
                batch_num = i // batch_size + 1
                batch = metadata.iloc[i:i+batch_size].copy()
                batch['device'] = 'cpu'  # Force CPU
                batch['out_dir'] = str(preproc_out)
                
                if parallel and len(batch) > 1:
                    # Get current site for display
                    current_site = batch.iloc[0].get("site", batch.iloc[0].get("dataset", "Unknown"))
                    pbar.set_postfix_str(f"{current_site} Batch {batch_num}/{total_batches}")
                    
                    results = run_parallel(
                        func=_process_subject,
                        items=batch.to_dict('records'),
                        desc=None 
                    )
                    
                    # Update progress bar for the entire batch
                    pbar.update(len(batch))
                    
                    # Check for failed results
                    failed_results = [r for r in results if r.get("status") == "failed" or r.get("status") == "error"]
                    if failed_results:
                        print(f"\nParallel Failure: {len(failed_results)} subjects failed")
                        for fail in failed_results[:5]:  # Show first 5
                            print(f"  - {fail.get('subject_id', 'unknown')}: {fail.get('error', 'unknown error')}")
                    
                else:
                    # SEQUENTIAL MODE: Show individual subject details
                    results = []
                    for _, row in batch.iterrows():
                        site = row.get("site", row.get("dataset", "UnknownSite"))
                        subject_id = row.get("subject_id", "unknown")
                        pbar.set_postfix_str(f"{site} {subject_id}")
                        
                        result = _process_subject(row)
                        results.append(result)
                        
                        # Update progress after each subject
                        pbar.update(1)
                        
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
        
        # Print detailed summary at the end
        success_subjects = [r for r in all_results if r["status"] == "success"]
        failed_subjects = [r for r in all_results if r["status"] == "failed"]
        
        print(f"\nPreprocessing Summary")
        print(f"Success: {len(success_subjects)} subjects")
        print(f"Failed: {len(failed_subjects)} subjects")
        
        if failed_subjects:
            print(f"\nFailed subjects:")
            for fail in failed_subjects[:10]:  # Show first 10
                print(f"  - {fail['subject_id']} ({fail.get('site', 'unknown')}): {fail.get('error', 'unknown error')}")
            if len(failed_subjects) > 10:
                print(f"  ... and {len(failed_subjects) - 10} more")
        
        return all_results
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise


def run_feature_extraction(metadata_out: Path, preproc_out: Path, 
                         feature_out_dir: Path, parallel: bool = True):
    """Run feature extraction stage of the pipeline with consistent progress display"""
    print("\nRunning Feature Extraction...")
    
    try:
        # Load metadata
        metadata = pd.read_csv(metadata_out)
        print(f"Loaded metadata for {len(metadata)} subjects")
        
        # Debugging output
        print(f"Sites in metadata: {metadata['site'].value_counts()}")

        # Check which subjects have preprocessing files
        missing_preprocessing = []
        for _, row in metadata.iterrows():
            site = row['site']
            subject_id = row['subject_id']
            func_path = preproc_out / site / subject_id / "func_preproc.nii.gz"
            mask_path = preproc_out / site / subject_id / "mask.nii.gz"
            
            if not func_path.exists() or not mask_path.exists():
                missing_preprocessing.append(f"{site}/{subject_id}")

        print(f"Missing preprocessing files for {len(missing_preprocessing)} subjects:")
        for subj in missing_preprocessing[:10]:  # Show first 10
            print(f"  - {subj}")

        # Try both possible atlas directory structures
        parcellation_candidates = [
            Path("atlas_schaefer-200/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii"),
            Path("atlas/Schaefer-200/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii"),
            Path("atlas/Schaefer-200/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"),
        ]
        
        parcellation_path = None
        for candidate in parcellation_candidates:
            if candidate.exists():
                parcellation_path = candidate
                print(f"✅ Found atlas: {parcellation_path}")
                break
        
        if parcellation_path is None:
            print(f"❌ CRITICAL: Atlas file not found!")
            print(f"Searched in:")
            for candidate in parcellation_candidates:
                print(f"  - {candidate.absolute()}")
            print("\nAvailable atlas files:")
            for atlas_dir in [Path("atlas"), Path("atlas_schaefer-200")]:
                if atlas_dir.exists():
                    for f in atlas_dir.rglob("*.nii*"):
                        print(f"  - {f}")
            raise ValueError(f"CRITICAL: Could not find Schaefer parcellation. "
                           "Feature extraction cannot proceed without real parcellation atlas.")
        
        parcellation = SchaeferParcellation(parcellation_path)
        
        if not parcellation.load_parcellation():
            raise ValueError(f"CRITICAL: Could not load Schaefer parcellation from {parcellation_path}. "
                           "Feature extraction cannot proceed without real parcellation atlas. "
                           "Please ensure the Schaefer atlas file is available.")
        
        print("Schaefer parcellation loaded successfully")
        atlas_labels = parcellation.roi_labels

        # Use dynamic batch size from config
        batch_size = TRAINING_CONFIG.get('batch_size', 8) if parallel else 1
        all_results = []
        total_batches = (len(metadata) - 1) // batch_size + 1
        
        with tqdm(total=len(metadata), desc="Feature Extraction", unit="subj") as pbar:
            for i in range(0, len(metadata), batch_size):
                batch_num = i // batch_size + 1
                batch = metadata.iloc[i:i+batch_size].copy()
                
                if parallel and len(batch) > 1:
                    # PARALLEL MODE: Show batch progress with current site
                    current_site = batch.iloc[0]['site'] if 'site' in batch.columns else batch.iloc[0].get('dataset', 'Unknown')
                    
                    pbar.set_postfix_str(f"{current_site} Batch {batch_num}/{total_batches}")
                    
                    # Add required parameters to each row for the worker
                    batch_with_params = []
                    for _, row in batch.iterrows():
                        row_dict = row.to_dict()
                        row_dict['preproc_dir'] = str(preproc_out)
                        row_dict['feature_out_dir'] = str(feature_out_dir)
                        row_dict['atlas_labels'] = atlas_labels
                        row_dict['parcellation_path'] = str(parcellation_path)
                        batch_with_params.append(row_dict)

                    results = run_parallel(
                        func=_extract_features_worker_wrapper,
                        items=batch_with_params,
                        desc=None
                    )
                    
                    # Update progress for entire batch
                    pbar.update(len(batch))
                    
                else:
                    # SEQUENTIAL MODE: Show individual subject details
                    results = []
                    for _, row in batch.iterrows():
                        site = row.get("site", row.get("dataset", "UnknownSite"))
                        subject_id = row.get("subject_id", "unknown")
                        pbar.set_postfix_str(f"{site} {subject_id}")
                        
                        # Extract features for individual subject
                        from feature_extraction.parcellation_and_feature_extraction import extract_features_worker
                        result = extract_features_worker(
                            row, preproc_out, feature_out_dir, atlas_labels, parcellation_path  # ✅ Add path
                        )
                        results.append(result)
                        
                        # Update progress after each subject
                        pbar.update(1)
                
                all_results.extend(results)
                
                # Check for parcellation failures
                parcellation_failures = [r for r in results if r.get("error_type") == "parcellation_unavailable"]
                if parcellation_failures:
                    print(f"\nCritical: {len(parcellation_failures)} subjects failed due to missing parcellation atlas")
                    print("Feature extraction cannot proceed without real Schaefer parcellation.")
                    break
        
        # Create feature manifest for training
        from feature_extraction import create_feature_manifest
        manifest_path = create_feature_manifest(feature_out_dir, metadata)
        print(f"Created feature manifest at {manifest_path}")
        
        # Print summary with error breakdown
        success = sum(1 for r in all_results if r["status"] == "success")
        failed = len(all_results) - success
        
        # Categorize failures
        parcellation_fails = sum(1 for r in all_results if r.get("error_type") == "parcellation_unavailable")
        preprocessing_fails = sum(1 for r in all_results if r.get("error_type") == "missing_preprocessing")
        other_fails = failed - parcellation_fails - preprocessing_fails
        
        print(f"\nFeature extraction complete:")
        print(f"Success: {success}")
        print(f"Failed: {failed}")
        if parcellation_fails:
            print(f"    - Parcellation unavailable: {parcellation_fails}")
        if preprocessing_fails:
            print(f"    - Missing preprocessing files: {preprocessing_fails}")
        if other_fails:
            print(f"    - Other errors: {other_fails}")
        
        print_detailed_error_summary(all_results, "FEATURE EXTRACTION")
        
        return all_results
        
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        raise


def _extract_features_worker_wrapper(row_dict):
    """Wrapper function that can be pickled for multiprocessing"""
    from feature_extraction.parcellation_and_feature_extraction import extract_features_worker
    from pathlib import Path
    
    # Extract parameters from row_dict
    preproc_dir = Path(row_dict.pop('preproc_dir'))
    feature_out_dir = Path(row_dict.pop('feature_out_dir'))
    atlas_labels = row_dict.pop('atlas_labels')
    parcellation_path = Path(row_dict.pop('parcellation_path'))  # ✅ Add this
    
    # Call the actual worker function
    return extract_features_worker(row_dict, preproc_dir, feature_out_dir, atlas_labels, parcellation_path)

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

def print_detailed_error_summary(all_results, stage_name="PROCESSING"):
    # Categorize results
    success_results = [r for r in all_results if r["status"] == "success"]
    failed_results = [r for r in all_results if r["status"] == "failed"]
    
    if not failed_results:
        print(f"\nAll {len(success_results)} subjects processed successfully!")
        return
    
    print(f"\n{'='*60}")
    print(f"{stage_name} DETAILED SUMMARY")
    print(f"{'='*60}")
    
    print(f"Success: {len(success_results)} subjects")
    print(f"Failed: {len(failed_results)} subjects")
    
    error_types = {}
    for result in failed_results:
        error_type = result.get("error_type", "unknown")
        if error_type not in error_types:
            error_types[error_type] = []
        error_types[error_type].append(result)
    
    print(f"\nError Breakdown by Type:")
    for error_type, errors in error_types.items():
        print(f"  {error_type}: {len(errors)} subjects")
    
    for error_type, errors in error_types.items():
        print(f"\n--- {error_type.upper()} ({len(errors)} subjects) ---")
        
        # Show ALL subjects (no truncation)
        for i, error in enumerate(errors):
            print(f"  {i+1}. {error['site']}/{error['subject_id']}")
            print(f"     Error: {error['error']}")
            
            if 'error_details' in error and error['error_details']:
                for key, value in error['error_details'].items():
                    if key == 'available_files' and isinstance(value, list):
                        print(f"     {key}: {', '.join(value[:3])}{'...' if len(value) > 3 else ''}")
                    elif key != 'traceback':
                        print(f"     {key}: {value}")
            print()


# main exec
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ADHD GNN-STAN pipeline")
    parser.add_argument("--stage", type=str,
                       choices=["preprocessing", "features", "split", "training", "full", "retry"],
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
    parser.add_argument("--retry-cleanup", action="store_true", default=True,
                       help="Cleanup corrupted files before retry (default: True)")
    parser.add_argument("--no-retry-cleanup", action="store_false", dest="retry_cleanup",
                       help="Don't cleanup corrupted files before retry")
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
        # Handle retry stage
        if args.stage == "retry":
            from preprocessing.retry_failed import retry_failed_preprocessing
            
            print("="*70)
            print("RETRYING FAILED PREPROCESSING")
            print("="*70)
            
            results = retry_failed_preprocessing(
                manifest_path=str(METADATA_OUT),
                output_dir=str(PREPROC_OUT),
                cleanup=args.retry_cleanup,
                n_jobs=2 if args.parallel else 1
            )
            
            # Save retry results
            retry_results_path = PREPROC_OUT / "retry_results.csv"
            results.to_csv(retry_results_path, index=False)
            print(f"\n💾 Retry results saved to: {retry_results_path}")
            sys.exit(0)
        
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
