
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import joblib
import cv2
from models import (
    train_svm_model,
    train_random_forest,
    train_gradient_boosting,
    train_lightgbm,
    train_neural_network, 
    build_hybrid_cnn,
)
from sklearn.model_selection import train_test_split
from eval import (
    evaluate_classification_model,
    stratified_kfold_validation,
    compare_models,
    plot_feature_importance
)
from img_preprocessing import detect_color_card, extract_cnn_features

def main():
    
    data = np.load('train_test_split.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    feature_names_data = np.load('feature_names.npz')
    feature_names = feature_names_data['feature_names'].tolist()
    feature_names = [str(name) for name in feature_names]
    
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of feature names: {len(feature_names)}")
    assert X_train.shape[1] == len(feature_names), "Feature count mismatch!"
    
        
    # 4. Define models to evaluate
    models = {
        'SVM': lambda X, y, **kw: train_svm_model(X, y, problem_type='classification'),
        'Random Forest': train_random_forest,
        'Gradient Boosting': train_gradient_boosting,
        'LightGBM': train_lightgbm,
        'Neural Network': lambda X, y, **kw: train_neural_network(
            X, y, X_val=X_test, y_val=y_test, problem_type='classification'
        )
    }

    # 5. Cross-validate and compare models
    print("\n=== Cross-Validation Results ===")
    mean_results, std_results = compare_models(
        X_train, y_train,
        models_dict=models,
        problem_type='classification',
        cv_method='stratified'
    )
    
    # Print cross-validation results
    print("\nMean Validation Metrics:")
    print(mean_results)
    print("\nStandard Deviation:")
    print(std_results)

    # 6. Final evaluation on test set
    print("\n=== Final Test Evaluation ===")
    final_metrics = {}
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    for model_name, model_fn in models.items():
        # Train on full training set
        model = model_fn(X_train, y_train)
        
        # Predict on test set
        if model_name == 'Neural Network':
            y_proba = model.predict(X_test)
            y_pred = (y_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Evaluate
        metrics = evaluate_classification_model(y_test, y_pred, y_proba)
        final_metrics[model_name] = metrics
        
        
        # Save model
        joblib.dump(model, f'models/{model_name}.pkl')
        
        # Plot feature importance for tree-based models
        if model_name in ['Random Forest', 'Gradient Boosting', 'LightGBM']:
            plt = plot_feature_importance(model, feature_names)
            plt.savefig(f'reports/{model_name}_feature_importance.png')
            plt.close()

    # 7. Generate final report
    report_df = pd.DataFrame(final_metrics).T
    print("\n=== Final Test Metrics ===")
    print(report_df)
    
    # Save results
    report_df.to_csv('reports/model_performance.csv')
    print("\nSaved results to reports/model_performance.csv")

def hybrid_classification_pipeline(medical_data_df):
    # 1. Train CNN feature extractor
    X_img =[]
    X_clinical = []
    y = []
    img_paths = []
    for _,row in medical_data_df.iterrows():
        img_path = row['new_image_name']
        # Load and preprocess the image
        img_path_updated = os.path.join('data/img_dataset', img_path)
        img = cv2.imread(img_path_updated)
        img = detect_color_card(img)
        if( img is None):
            print(f"Image not found: {img_path_updated}")
            continue
        
        X_img.append(img)
        X_clinical.append(row.drop(['new_image_name']).values)
        y.append(row['Jaundice Decision'])
        img_paths.append(img_path)
    
    X_img_train, X_img_test, X_clinical_train, X_clinical_test,train_image_paths, test_image_paths, y_train, y_test = train_test_split(
    X_img, X_clinical,img_paths, y, test_size=0.2, stratify=y, random_state=42
)
        
    cnn_model = build_hybrid_cnn((128,128,3))
    # Add this function to models.py
    cnn_model.fit(X_img_train, y_train, epochs=10, validation_split=0.2)
    
    # 2. Extract CNN features
    cnn_features_train = [extract_cnn_features(path, cnn_model) for path in train_image_paths]
    cnn_features_test = [extract_cnn_features(path, cnn_model) for path in test_image_paths]
    
    # 3. Combine with clinical data
    X_hybrid_train = np.hstack([cnn_features_train, X_clinical_train])
    X_hybrid_test = np.hstack([cnn_features_test, X_clinical_test])
    
    # 4. Train traditional classifiers
    svm = train_svm_model(X_hybrid_train, y_train, problem_type='classification')
    rf = train_random_forest(X_hybrid_train, y_train, problem_type='classification')
    gb = train_gradient_boosting(X_hybrid_train, y_train, problem_type='classification')
    lgbm = train_lightgbm(X_hybrid_train, y_train, problem_type='classification')
    
    y_pred_svm = svm.predict(X_hybrid_test)
    y_pred_rf = rf.predict(X_hybrid_test)
    y_pred_gb = gb.predict(X_hybrid_test)
    y_pred_lgbm = lgbm.predict(X_hybrid_test)
    
    svm_metrics = evaluate_classification_model(y_test, y_pred_svm)
    rf_metrics = evaluate_classification_model(y_test, y_pred_rf)
    gb_metrics = evaluate_classification_model(y_test, y_pred_gb)
    lgbm_metrics = evaluate_classification_model(y_test, y_pred_lgbm)
    print("SVM Metrics:", svm_metrics)
    print("Random Forest Metrics:", rf_metrics)
    print("Gradient Boosting Metrics:", gb_metrics)
    print("LightGBM Metrics:", lgbm_metrics)


if __name__ == "__main__":
    
    hybrid_classification_pipeline(medical_data_df=pd.read_csv('data/clinical_data_updated.csv'))

