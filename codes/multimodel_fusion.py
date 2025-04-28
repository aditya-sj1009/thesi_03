import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class EarlyFusion(BaseEstimator, TransformerMixin):
    """Early fusion: simply concatenate features from different modalities"""
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X_list):
        """
        Parameters:
        X_list: List of feature arrays from different modalities
        """
        return np.hstack(X_list)

class HierarchicalFusion(BaseEstimator, TransformerMixin):
    """Hierarchical fusion: first combine skin regions, then combine with clinical data"""
    def __init__(self, region_indices, clinical_indices):
        self.region_indices = region_indices  # List of lists, each containing indices for one region
        self.clinical_indices = clinical_indices
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        Parameters:
        X: Combined feature array
        """
        # Extract region features
        region_features = []
        for indices in self.region_indices:
            region_features.append(X[:, indices])
        
        # Extract clinical features
        clinical_features = X[:, self.clinical_indices]
        
        # First-level fusion: combine region features
        combined_regions = np.hstack([np.mean(region, axis=1, keepdims=True) for region in region_features])
        
        # Second-level fusion: combine with clinical features
        return np.hstack((combined_regions, clinical_features))

def prepare_multimodal_data(image_features_by_region, clinical_features):
    """
    Prepare data for multimodal fusion
    
    Parameters:
    image_features_by_region: Dictionary with regions as keys and feature arrays as values
    clinical_features: Array of clinical features
    """
    # Early fusion
    regions = list(image_features_by_region.keys())
    X_early = np.hstack([image_features_by_region[region] for region in regions] + [clinical_features])
    
    # Prepare indices for hierarchical fusion
    start_idx = 0
    region_indices = []
    for region in regions:
        n_features = image_features_by_region[region].shape[1]
        region_indices.append(list(range(start_idx, start_idx + n_features)))
        start_idx += n_features
    
    clinical_indices = list(range(start_idx, start_idx + clinical_features.shape[1]))
    
    return X_early, region_indices, clinical_indices

