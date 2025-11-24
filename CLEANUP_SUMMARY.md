Repository Cleanup Summary
==========================
Date: 2025-11-24 18:02:18

Files Deleted:
--------------
Root Directory:
- check_v9_distribution.py (single-use validation script)
- train_v6_with_predictions.py (one-time training script)
- update_notebook_with_real_data.py (one-time update script)
- verify_balanced_batches.py (single-use verification script)
- CLEANUP_LOG.txt (temporary log)
- MULTI_SITE_UPDATE.txt (temporary documentation)
- TASK_COMPLETION_REPORT.md (temporary report)

Scripts Directory:
- check_preprocessed_integrity.py (one-time validation)
- merge_peking_sites.py (one-time data processing)
- update_metadata_all_sites.py (one-time metadata update)
- validate_metadata.py (one-time validation)
- train_baseline_accurate.py (superseded by versioned training scripts)
- run-corruption-check.sh (debug script)
- run-debug.sh (debug script)

Directories Deleted:
--------------------
- feature_extraction_outputs/ (temporary outputs)

.gitignore Updates:
-------------------
Added to ignore:
- uv.lock
- thesis-adhd/ (virtual environment)
- .python-version
- data/trained/*/run_*/ (intermediate training runs)
- data/trained/checkpoints/
- data/attention_weights/
- data/roi_ranking/
- data/site_configs/
- data/ablation_results/
- experiments/ (temporary comparison outputs)
- feature_extraction_outputs/
- figures/*.png and figures/*.pdf (generated figures)
- *.log, *.tmp files
- *_temp.py, *_test.py (temporary scripts)
- Model checkpoints (*.pth, *.pt) except best_model.pth

Files Kept:
-----------
Core Scripts:
- extract_attention_weights.py (thesis analysis)
- extract_predictions_to_csv.py (thesis analysis)
- extract_table_data.py (thesis analysis)
- run_ablation_study.py (thesis analysis)
- run_comparison_with_stats.py (thesis analysis)
- summarize_results.py (results aggregation)

Important Outputs:
- data/predictions/*.csv (final model predictions)
- experiments/statistical_comparison/ (final statistical analysis)
- thesis_visualizations.ipynb (thesis figures)

Documentation:
- README.md (main documentation)
- ATTENTION_EXTRACTION_README.md
- DUPLICATE_PREVENTION_GUIDE.md
- GPU_SETUP.md
- MULTI_SITE_SETUP_GUIDE.md
- ROI_RANKING_AND_SITE_SELECTION.md
- THESIS_TOOLS_README.md
- THESIS_VISUALIZATION_REAL_DATA.md

