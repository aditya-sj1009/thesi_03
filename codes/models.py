import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import( 
Conv2D, MaxPooling2D, Flatten, Input, concatenate)

def train_svm_model(X_train, y_train, problem_type='classification', class_weights=None):
    """Train SVM model for classification or regression"""
    if problem_type == 'classification':
        model = SVC(kernel='rbf', probability=True, class_weight=class_weights)
    else:  # regression
        model = SVR(kernel='rbf')
    
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, problem_type='classification', class_weights=None):
    """Train Random Forest model for classification or regression"""
    if problem_type == 'classification':
        model = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42)
    else:  # regression
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train, problem_type='classification' , **kwargs):
    """Train Gradient Boosting model for classification or regression"""
    if problem_type == 'classification':
        model = GradientBoostingClassifier(random_state=42)
    else:  # regression
        model = GradientBoostingRegressor(random_state=42)
    
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train, problem_type='classification', class_weights=None):
    """Train LightGBM model for classification or regression"""
    if problem_type == 'classification':
        if class_weights:
            # Convert class weights to sample weights
            sample_weights = np.array([class_weights[y] for y in y_train])
            model = lgb.LGBMClassifier(random_state=42)
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model = lgb.LGBMClassifier(random_state=42)
            model.fit(X_train, y_train)
    else:  # regression
        model = lgb.LGBMRegressor(random_state=42)
        model.fit(X_train, y_train)
    
    return model

def create_neural_network(input_dim, problem_type='classification', num_classes=None):
    """Create a simple neural network for classification or regression"""
    model = Sequential()
    
    # Input layer
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Hidden layer
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Output layer
    if problem_type == 'classification':
        if num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
        else:
            model.add(Dense(num_classes, activation='softmax'))
            model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
    else:  # regression
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mean_squared_error',
                     metrics=['mae'])
    
    return model

def train_neural_network(X_train, y_train, X_val, y_val, problem_type='classification', class_weights=None):
    """Train a neural network with early stopping"""
    input_dim = X_train.shape[1]
    
    if problem_type == 'classification':
        num_classes = len(np.unique(y_train))
        model = create_neural_network(input_dim, problem_type, num_classes)
    else:
        model = create_neural_network(input_dim, problem_type)
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    if problem_type == 'classification' and class_weights:
        # Convert class weights dict to array for Keras
        sample_weights = np.array([class_weights[y] for y in y_train])
        model.fit(X_train, y_train, 
                 validation_data=(X_val, y_val),
                 epochs=100, 
                 batch_size=8,
                 sample_weight=sample_weights,
                 callbacks=[early_stopping],
                 verbose=0)
    else:
        model.fit(X_train, y_train, 
                 validation_data=(X_val, y_val),
                 epochs=100, 
                 batch_size=8,
                 callbacks=[early_stopping],
                 verbose=0)
    
    return model

def create_feature_extractor(input_shape):
    """Create CNN for feature extraction without classification layer"""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),  # Feature vector output
        Dropout(0.5)
    ])
    return model

def build_hybrid_cnn(input_shape):
    """CNN model for jaundice feature extraction"""
    model = Sequential()
    
    # Feature extraction backbone
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    
    # Feature compression
    model.add(Dense(128, activation='relu', name='cnn_features'))
    
    # Optional: Add classification head
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
