"""
Extract predictions from trained models to CSV format (no pandas dependency)
Creates a comprehensive CSV with all predictions across all folds
"""

import json
import csv
from pathlib import Path

def extract_predictions_from_results(results_path, model_name):
    """
    Extract predictions from results.json file
    
    Args:
        results_path: Path to results.json
        model_name: Name of the model (e.g., 'V6', 'V7', 'V8')
    
    Returns:
        List of prediction dictionaries
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    all_predictions = []
    
    if 'fold_results' not in results:
        print(f"Warning: No fold_results found in {results_path}")
        return []
    
    for fold_idx, fold in enumerate(results['fold_results']):
        test_site = fold.get('test_site', f'fold_{fold_idx}')
        test_metrics = fold.get('test_metrics', {})
        
        # Extract predictions
        true_labels = test_metrics.get('true_labels', [])
        predictions = test_metrics.get('predictions', [])
        probabilities = test_metrics.get('probabilities', [])
        
        if not true_labels or not predictions:
            print(f"Warning: No predictions found for fold {fold_idx} ({test_site})")
            continue
        
        # Create records for this fold
        for idx, (true_label, pred_label, prob) in enumerate(zip(true_labels, predictions, probabilities)):
            # prob is [prob_class_0, prob_class_1]
            if isinstance(prob, list):
                prob_hc = float(prob[0])  # Healthy Control
                prob_adhd = float(prob[1])  # ADHD
            else:
                # If it's a single value, assume it's ADHD probability
                prob_adhd = float(prob)
                prob_hc = 1.0 - prob_adhd
            
            all_predictions.append({
                'model': model_name,
                'fold': fold_idx,
                'test_site': test_site,
                'subject_index': idx,
                'true_label': int(true_label),
                'predicted_label': int(pred_label),
                'probability_adhd': prob_adhd,
                'probability_hc': prob_hc,
                'correct': int(true_label == pred_label)
            })
    
    return all_predictions


def save_to_csv(predictions, output_path, fieldnames):
    """Save predictions to CSV file"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def main():
    # Define model paths
    data_dir = Path("data/trained")
    
    models = {
        'V6': data_dir / "baseline_accurate_v6" / "run_1" / "results.json",
        'V7': data_dir / "baseline_accurate_v7" / "run_1" / "results.json",
        'V8': data_dir / "baseline_accurate_v8" / "run_1" / "results.json"
    }
    
    # Extract predictions from all models
    all_predictions = []
    model_predictions = {}
    
    print("\n" + "="*70)
    print("Extracting predictions from trained models...")
    print("="*70)
    
    fieldnames = ['model', 'fold', 'test_site', 'subject_index', 'true_label', 
                  'predicted_label', 'probability_adhd', 'probability_hc', 'correct']
    
    for model_name, model_path in models.items():
        if model_path.exists():
            print(f"\n📊 Extracting from {model_name}...")
            predictions = extract_predictions_from_results(model_path, model_name)
            if predictions:
                model_predictions[model_name] = predictions
                all_predictions.extend(predictions)
                print(f"   ✓ Extracted {len(predictions)} predictions")
            else:
                print(f"   ⚠️  No predictions found")
        else:
            print(f"\n⚠️  Not found: {model_path}")
    
    # Save predictions
    if all_predictions:
        output_dir = Path("data/predictions")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save combined predictions
        combined_path = output_dir / "all_model_predictions.csv"
        save_to_csv(all_predictions, combined_path, fieldnames)
        print(f"\n✅ Combined predictions saved to: {combined_path}")
        print(f"   Total predictions: {len(all_predictions)}")
        
        # Save per-model predictions
        for model_name, predictions in model_predictions.items():
            model_path = output_dir / f"predictions_{model_name}.csv"
            save_to_csv(predictions, model_path, fieldnames)
            print(f"   {model_name}: {model_path} ({len(predictions)} predictions)")
        
        # Create summary statistics
        print("\n" + "="*70)
        print("Prediction Summary:")
        print("="*70)
        
        for model_name, predictions in model_predictions.items():
            total = len(predictions)
            correct = sum(1 for p in predictions if p['correct'])
            adhd_samples = sum(1 for p in predictions if p['true_label'] == 1)
            hc_samples = total - adhd_samples
            accuracy = (correct / total * 100) if total > 0 else 0
            
            print(f"\n{model_name}:")
            print(f"  Total predictions: {total}")
            print(f"  Correct: {correct}")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  ADHD samples: {adhd_samples}")
            print(f"  HC samples: {hc_samples}")
        
        # Save summary
        summary_path = output_dir / "prediction_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Prediction Summary\n")
            f.write("="*70 + "\n\n")
            for model_name, predictions in model_predictions.items():
                total = len(predictions)
                correct = sum(1 for p in predictions if p['correct'])
                adhd_samples = sum(1 for p in predictions if p['true_label'] == 1)
                hc_samples = total - adhd_samples
                accuracy = (correct / total * 100) if total > 0 else 0
                
                f.write(f"{model_name}:\n")
                f.write(f"  Total predictions: {total}\n")
                f.write(f"  Correct: {correct}\n")
                f.write(f"  Accuracy: {accuracy:.2f}%\n")
                f.write(f"  ADHD samples: {adhd_samples}\n")
                f.write(f"  HC samples: {hc_samples}\n\n")
        
        print(f"\n📊 Summary saved to: {summary_path}")
        
    else:
        print("\n❌ No predictions found in any model")


if __name__ == '__main__':
    main()
