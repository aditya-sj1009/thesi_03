from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def apply_oversampling(X, y, method='smote', random_state=42):
    """Apply oversampling to the minority class"""
    if method == 'smote':
        sampler = SMOTE(random_state=random_state)
    elif method == 'adasyn':
        sampler = ADASYN(random_state=random_state)
    else:
        raise ValueError("Unsupported oversampling method")
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled

def apply_undersampling(X, y, method='random', random_state=42):
    """Apply undersampling to the majority class"""
    if method == 'random':
        sampler = RandomUnderSampler(random_state=random_state)
    elif method == 'tomek':
        sampler = TomekLinks()
    else:
        raise ValueError("Unsupported undersampling method")
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled

def apply_hybrid_sampling(X, y, method='smotetomek', random_state=42):
    """Apply hybrid sampling methods"""
    if method == 'smotetomek':
        sampler = SMOTETomek(random_state=random_state)
    elif method == 'smoteenn':
        sampler = SMOTEENN(random_state=random_state)
    else:
        raise ValueError("Unsupported hybrid sampling method")
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled

def get_class_weights(y):
    """Calculate class weights for imbalanced classes"""
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, weights))
    return class_weights
