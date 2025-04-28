from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imbalnce_handling import get_class_weights

def evaluate_classification_model(y_true, y_pred, y_prob=None):
    """Evaluate a classification model with appropriate metrics for imbalanced data"""
    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred = (y_pred > 0.5).astype(int)
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    
    # Add AUC if probability estimates are available
    if y_prob is not None:
        n_classes = len(np.unique(y_true))
        
        if n_classes == 2:  # Binary classification
            # Handle both 1D and 2D probability arrays
            if y_prob.ndim == 1 or y_prob.shape[1] == 1:
                results['auc'] = roc_auc_score(y_true, y_prob)
            else:
                results['auc'] = roc_auc_score(y_true, y_prob[:, 1])
        else:  # Multiclass
            results['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
    
    return results

def evaluate_regression_model(y_true, y_pred):
    """Evaluate a regression model with appropriate metrics"""
    results = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }
    return results

def stratified_kfold_validation(X, y, model_fn, n_splits=5, problem_type='classification', random_state=42):
    """Perform stratified k-fold cross-validation for classification or regression"""
    results = []
    
    if problem_type == 'classification':
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Calculate class weights for handling imbalance
            class_weights = get_class_weights(y_train)
            
            # Train model
            model = model_fn(X_train, y_train, problem_type='classification', class_weights=class_weights)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Get probability estimates if available
            y_prob = None
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
            
            # Evaluate
            fold_results = evaluate_classification_model(y_test, y_pred, y_prob)
            results.append(fold_results)
    
    else:  # regression
        # For regression, use regular KFold, but keep the same API
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        # Create fake categorical target for stratification
        y_categorical = pd.qcut(y, q=n_splits, labels=False)
        
        for train_idx, test_idx in cv.split(X, y_categorical):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model = model_fn(X_train, y_train, problem_type='regression')
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            fold_results = evaluate_regression_model(y_test, y_pred)
            results.append(fold_results)
    
    # Calculate mean and std
    results_df = pd.DataFrame(results)
    mean_results = results_df.mean()
    std_results = results_df.std()
    
    return mean_results, std_results, results_df

def leave_one_out_validation(X, y, model_fn, problem_type='classification'):
    """Perform leave-one-out cross-validation"""
    cv = LeaveOneOut()
    
    predictions = []
    true_values = []
    probabilities = []
    
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        if problem_type == 'classification':
            # Calculate class weights for handling imbalance
            class_weights = get_class_weights(y_train)
            
            # Train model
            model = model_fn(X_train, y_train, problem_type='classification', class_weights=class_weights)
            
            # Predict
            y_pred = model.predict(X_test)
            predictions.append(y_pred[0])
            true_values.append(y_test[0])
            
            # Get probability estimates if available
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
                probabilities.append(y_prob[0])
        
        else:  # regression
            # Train model
            model = model_fn(X_train, y_train, problem_type='regression')
            
            # Predict
            y_pred = model.predict(X_test)
            predictions.append(y_pred[0])
            true_values.append(y_test[0])
    
    # Convert to arrays
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    
    if problem_type == 'classification':
        if len(probabilities) > 0:
            probabilities = np.array(probabilities)
            results = evaluate_classification_model(true_values, predictions, probabilities)
        else:
            results = evaluate_classification_model(true_values, predictions)
    else:  # regression
        results = evaluate_regression_model(true_values, predictions)
    
    return results, predictions, true_values

def compare_models(X, y, models_dict, problem_type='classification', cv_method='stratified', n_splits=5):
    """Compare multiple models using cross-validation"""
    results = {}
    
    for name, model_fn in models_dict.items():
        print(f"Evaluating {name}...")
        
        if cv_method == 'stratified':
            mean_results, std_results, _ = stratified_kfold_validation(
                X, y, model_fn, n_splits=n_splits, problem_type=problem_type)
            results[name] = {
                'mean': mean_results,
                'std': std_results
            }
        elif cv_method == 'loocv':
            results[name], _, _ = leave_one_out_validation(X, y, model_fn, problem_type=problem_type)
        else:
            raise ValueError("Unsupported cross-validation method")
    
    # Convert to dataframe for easier comparison
    if cv_method == 'stratified':
        mean_df = pd.DataFrame({name: results[name]['mean'] for name in models_dict.keys()})
        std_df = pd.DataFrame({name: results[name]['std'] for name in models_dict.keys()})
        return mean_df, std_df
    else:
        return pd.DataFrame(results)

def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance for a trained model"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
    else:
        raise ValueError("Model doesn't provide feature importance information")
    
    # Validate feature names length
    if len(feature_names) != len(importances):
        raise ValueError(f"Mismatch between features ({len(feature_names)}) "
                        f"and importances ({len(importances)})")
        
    # Sort feature importances
    indices = np.argsort(importances)[::-1]
    
    # Select top N features
    indices = indices[:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Top Features by Importance')
    plt.tight_layout()
    return plt
