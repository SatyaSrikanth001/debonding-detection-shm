import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    mean_absolute_error,
    r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classification(y_true, y_pred, classes=None):
    """Evaluate classification performance"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, target_names=classes)
    }
    return metrics

def evaluate_regression(y_true, y_pred):
    """Evaluate regression performance"""
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

def plot_confusion_matrix(cm, classes):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()

def evaluate_model(model, X_test, y_test, task_type='classification', class_names=None):
    """Enhanced evaluation with better visualization"""
    y_pred = model.predict(X_test)
    
    if task_type == 'classification':
        # Generate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(
                y_test, y_pred, 
                target_names=class_names,
                output_dict=True
            )
        }
        
        # Enhanced visualization
        plt.figure(figsize=(12, 6))
        
        # Confusion Matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(
            metrics['confusion_matrix'], 
            annot=True, fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        
        # Classification Metrics
        plt.subplot(1, 2, 2)
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        sns.heatmap(
            report_df.iloc[:-3, :-1], 
            annot=True, 
            cmap='YlGnBu'
        )
        plt.title('Classification Metrics')
        plt.tight_layout()
        plt.savefig('classification_results.png')
        plt.close()
        
        return metrics
        
    elif task_type == 'regression':
        return {
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    else:
        raise ValueError("task_type must be 'classification' or 'regression'")