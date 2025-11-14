"""
Visualization Utilities for ADHD Classification Pipeline

Provides comprehensive plotting functions for:
- Attention maps (spatial and temporal)
- Confusion matrices
- ROC curves and PR curves
- Training/validation curves
- Feature importance visualization
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report, ConfusionMatrixDisplay
)


# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_attention_maps(
    attention_weights: np.ndarray,
    roi_names: Optional[List[str]] = None,
    title: str = "Attention Weights",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'YlOrRd'
) -> plt.Figure:
    """
    Plot attention weight heatmap
    
    Args:
        attention_weights: Attention weights (n_heads, seq_len, seq_len) or (seq_len, seq_len)
        roi_names: Optional list of ROI names
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        cmap: Colormap
    """
    # Handle multi-head attention
    if attention_weights.ndim == 3:
        n_heads = attention_weights.shape[0]
        fig, axes = plt.subplots(1, n_heads, figsize=(figsize[0] * n_heads / 4, figsize[1]))
        if n_heads == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            sns.heatmap(
                attention_weights[i],
                cmap=cmap,
                square=True,
                cbar=True,
                xticklabels=roi_names if roi_names and len(roi_names) < 50 else False,
                yticklabels=roi_names if roi_names and len(roi_names) < 50 else False,
                ax=ax,
                vmin=0,
                vmax=1
            )
            ax.set_title(f'{title} - Head {i+1}')
            ax.set_xlabel('Key')
            ax.set_ylabel('Query')
    else:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            attention_weights,
            cmap=cmap,
            square=True,
            cbar=True,
            xticklabels=roi_names if roi_names and len(roi_names) < 50 else False,
            yticklabels=roi_names if roi_names and len(roi_names) < 50 else False,
            ax=ax,
            vmin=0,
            vmax=1
        )
        ax.set_title(title)
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention map saved to {save_path}")
    
    return fig


def plot_spatial_attention(
    gnn_attention: np.ndarray,
    roi_coordinates: Optional[np.ndarray] = None,
    title: str = "GNN Spatial Attention",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot GNN spatial attention weights
    
    Args:
        gnn_attention: Graph attention weights (n_nodes, n_nodes) or (n_heads, n_nodes, n_nodes)
        roi_coordinates: Optional 3D coordinates of ROIs for brain visualization
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    if gnn_attention.ndim == 3:
        # Average across heads
        attention = np.mean(gnn_attention, axis=0)
    else:
        attention = gnn_attention
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        attention,
        cmap='viridis',
        square=True,
        cbar=True,
        ax=ax,
        vmin=0,
        vmax=np.percentile(attention, 95)  # Clip outliers
    )
    ax.set_title(f'{title} (Averaged across heads)')
    ax.set_xlabel('Target ROI')
    ax.set_ylabel('Source ROI')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spatial attention map saved to {save_path}")
    
    return fig


def plot_temporal_attention(
    stan_attention: np.ndarray,
    timepoint_labels: Optional[List[str]] = None,
    title: str = "STAN Temporal Attention",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot STAN temporal attention weights
    
    Args:
        stan_attention: Temporal attention weights (n_heads, n_timepoints, n_timepoints)
        timepoint_labels: Optional timepoint labels
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    if stan_attention.ndim == 3:
        # Average across heads
        attention = np.mean(stan_attention, axis=0)
    else:
        attention = stan_attention
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(attention, cmap='plasma', aspect='auto', vmin=0, vmax=1)
    ax.set_title(f'{title} (Averaged across heads)')
    ax.set_xlabel('Time (TR)')
    ax.set_ylabel('Time (TR)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Temporal attention map saved to {save_path}")
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ['Control', 'ADHD'],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
    normalize: bool = True
) -> plt.Figure:
    """
    Plot confusion matrix with percentages and counts
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        normalize: Whether to show normalized values
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cm_norm,
            annot=cm,  # Show counts
            fmt='d',
            cmap='Blues',
            square=True,
            cbar=True,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            vmin=0,
            vmax=1
        )
        
        # Add percentages as text
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = ax.text(j + 0.5, i + 0.7, f'({cm_norm[i, j]*100:.1f}%)',
                             ha="center", va="center", color="gray", fontsize=9)
    else:
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            square=True,
            cbar=True,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
    
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot ROC curve with AUC
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot Precision-Recall curve
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(recall, precision, color='blue', lw=2,
            label=f'PR curve (AUC = {pr_auc:.3f})')
    
    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    ax.plot([0, 1], [baseline, baseline], color='navy', lw=2, 
            linestyle='--', label=f'Baseline ({baseline:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curve saved to {save_path}")
    
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    title: str = "Training History",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Plot training and validation curves
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_accs: Training accuracies per epoch
        val_accs: Validation accuracies per epoch
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    top_k: int = 20,
    title: str = "Top ROI Importance",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot feature importance (e.g., ROI rankings)
    
    Args:
        feature_names: Feature names (ROI names)
        importance_scores: Importance scores
        top_k: Number of top features to show
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    # Sort by importance
    sorted_indices = np.argsort(importance_scores)[-top_k:][::-1]
    top_features = [feature_names[i] for i in sorted_indices]
    top_scores = importance_scores[sorted_indices]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, top_scores, color='steelblue')
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, top_scores)):
        ax.text(score + 0.002, i, f'{score:.3f}', 
                va='center', fontsize=9, color='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    return fig


def plot_site_comparison(
    site_results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'sensitivity', 'specificity', 'auc'],
    title: str = "Performance by Site",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot performance comparison across sites
    
    Args:
        site_results: Dictionary mapping site names to metric dictionaries
        metrics: Metrics to plot
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    sites = list(site_results.keys())
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        values = [site_results[site].get(metric, 0) for site in sites]
        
        axes[i].bar(range(len(sites)), values, color='teal', alpha=0.7)
        axes[i].set_xticks(range(len(sites)))
        axes[i].set_xticklabels(sites, rotation=45, ha='right')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_title(f'{metric.capitalize()} by Site')
        axes[i].grid(axis='y', alpha=0.3)
        axes[i].set_ylim([0, 1])
        
        # Add value labels
        for j, val in enumerate(values):
            axes[i].text(j, val + 0.02, f'{val:.3f}', 
                        ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Site comparison plot saved to {save_path}")
    
    return fig


def plot_cross_validation_results(
    fold_results: List[Dict[str, float]],
    metrics: List[str] = ['accuracy', 'sensitivity', 'specificity', 'auc', 'f1'],
    title: str = "Cross-Validation Results",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot cross-validation results across folds
    
    Args:
        fold_results: List of result dictionaries per fold
        metrics: Metrics to plot
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    n_folds = len(fold_results)
    fold_ids = [f"Fold {i+1}" for i in range(n_folds)]
    
    # Prepare data
    data = {metric: [] for metric in metrics}
    for fold_result in fold_results:
        for metric in metrics:
            data[metric].append(fold_result.get(metric, 0))
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(n_folds)
    width = 0.15
    multiplier = 0
    
    for metric in metrics:
        offset = width * multiplier
        ax.bar(x + offset, data[metric], width, label=metric.capitalize())
        multiplier += 1
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(fold_ids)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cross-validation results saved to {save_path}")
    
    return fig


def create_visualization_report(
    results: Dict[str, Any],
    output_dir: Path,
    experiment_name: str = "experiment"
) -> None:
    """
    Create comprehensive visualization report from validation results
    
    Args:
        results: Validation results dictionary
        output_dir: Output directory for visualizations
        experiment_name: Name of experiment
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Creating visualization report for: {experiment_name}")
    
    # 1. Confusion Matrix
    if 'fold_results' in results:
        # Aggregate predictions from all folds
        all_true = []
        all_pred = []
        for fold in results['fold_results']:
            if 'test_metrics' in fold and 'predictions' in fold['test_metrics']:
                all_true.extend(fold['test_metrics']['true_labels'])
                all_pred.extend(fold['test_metrics']['predictions'])
        
        if all_true and all_pred:
            plot_confusion_matrix(
                np.array(all_true),
                np.array(all_pred),
                title=f"Confusion Matrix - {experiment_name}",
                save_path=output_dir / f'{experiment_name}_confusion_matrix.png'
            )
            plt.close()
    
    # 2. ROC Curve
    if 'fold_results' in results:
        all_true = []
        all_proba = []
        for fold in results['fold_results']:
            if 'test_metrics' in fold and 'probabilities' in fold['test_metrics']:
                all_true.extend(fold['test_metrics']['true_labels'])
                all_proba.extend(fold['test_metrics']['probabilities'])
        
        if all_true and all_proba:
            plot_roc_curve(
                np.array(all_true),
                np.array(all_proba),
                title=f"ROC Curve - {experiment_name}",
                save_path=output_dir / f'{experiment_name}_roc_curve.png'
            )
            plt.close()
            
            # Precision-Recall curve
            plot_precision_recall_curve(
                np.array(all_true),
                np.array(all_proba),
                title=f"Precision-Recall Curve - {experiment_name}",
                save_path=output_dir / f'{experiment_name}_pr_curve.png'
            )
            plt.close()
    
    # 3. Cross-validation results
    if 'fold_results' in results:
        fold_metrics = []
        for fold in results['fold_results']:
            if 'test_metrics' in fold:
                fold_metrics.append(fold['test_metrics'])
        
        if fold_metrics:
            plot_cross_validation_results(
                fold_metrics,
                title=f"Cross-Validation Results - {experiment_name}",
                save_path=output_dir / f'{experiment_name}_cv_results.png'
            )
            plt.close()
    
    # 4. Site comparison (for LOSO)
    if 'validation_type' in results and 'loso' in results['validation_type'].lower():
        if 'fold_results' in results:
            site_results = {}
            for fold in results['fold_results']:
                if 'test_site' in fold and 'test_metrics' in fold:
                    site = fold['test_site']
                    site_results[site] = fold['test_metrics']
            
            if site_results:
                plot_site_comparison(
                    site_results,
                    title=f"Performance by Site - {experiment_name}",
                    save_path=output_dir / f'{experiment_name}_site_comparison.png'
                )
                plt.close()
    
    print(f"✅ Visualization report saved to: {output_dir}")


if __name__ == '__main__':
    # Example usage
    print("Visualization utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  - plot_attention_maps()")
    print("  - plot_spatial_attention()")
    print("  - plot_temporal_attention()")
    print("  - plot_confusion_matrix()")
    print("  - plot_roc_curve()")
    print("  - plot_precision_recall_curve()")
    print("  - plot_training_curves()")
    print("  - plot_feature_importance()")
    print("  - plot_site_comparison()")
    print("  - plot_cross_validation_results()")
    print("  - create_visualization_report()")
