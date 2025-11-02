import json

with open('data/trained/checkpoints/final_results.json', 'r') as f:
    results = json.load(f)

print('=' * 60)
print('TRAINING RESULTS SUMMARY')
print('=' * 60)
print()

# Calculate average CV accuracy from fold results
cv_results = results['cv_results']['cv_results']
avg_cv_acc = sum(fold['best_val_acc'] for fold in cv_results) / len(cv_results)

print('5-FOLD CROSS-VALIDATION:')
print(f'  Average Accuracy: {avg_cv_acc:.2f}%')
print()

# LOSO validation if exists
if 'loso_results' in results:
    print('LEAVE-ONE-SITE-OUT VALIDATION:')
    print(f'  Average Accuracy: {results["loso_results"]["avg_test_acc"]:.2f}%')
    print()

print('FINAL TEST SET EVALUATION:')
print(f'  Accuracy:  {results["test_accuracy"]:.2f}%')
print(f'  Precision: {results["test_metrics"]["precision"]:.4f}')
print(f'  Recall:    {results["test_metrics"]["recall"]:.4f}')
print(f'  F1-Score:  {results["test_metrics"]["f1"]:.4f}')
print(f'  AUC:       {results["test_metrics"]["auc"]:.4f}')
print()

print('=' * 60)
print('COMPARISON TO PREVIOUS (BUGGY) RESULTS:')
print('=' * 60)
print('BEFORE (with diagnosis labeling bug):')
print('  Accuracy: 21.05%  ← Model predicted all "non-ADHD"')
print('  Precision: 0.0')
print('  Recall: 0.0')
print('  F1: 0.0')
print('  AUC: NaN')
print()
print('AFTER (with corrected labels):')
print(f'  Accuracy: {results["test_accuracy"]:.2f}%  ← Improved!')
print(f'  Precision: {results["test_metrics"]["precision"]:.4f}')
print(f'  Recall: {results["test_metrics"]["recall"]:.4f}')
print(f'  F1: {results["test_metrics"]["f1"]:.4f}')
print(f'  AUC: {results["test_metrics"]["auc"]:.4f}')
print()
print('=' * 60)
