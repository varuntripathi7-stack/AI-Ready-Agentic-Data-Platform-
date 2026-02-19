#!/usr/bin/env python3
"""
Machine Learning Pipeline
Trains a logistic regression model to predict if a user will make a purchase.
Uses features from the feature engineering pipeline.
"""

import os # Used for file path operations and directory management
import json # Used for saving model performance metrics in JSON format
import pickle # Used for saving the trained model and scaler objects to disk
from datetime import datetime # Used for timestamping the training date in the metrics
import pandas as pd # Used for data manipulation and loading features from Delta Lake or parquet files
import numpy as np # Used for handling missing values and infinity in the feature data
from sklearn.model_selection import train_test_split # Used for splitting the dataset into training and testing sets
from sklearn.preprocessing import StandardScaler # Used for scaling the feature data before training the model
from sklearn.linear_model import LogisticRegression # Used for training a logistic regression model to predict purchase behavior


# Used for evaluating the performance of the trained model using various metrics and confusion matrix
# Includes:-  
# accuracy:- the proportion of correct predictions (both true positives and true negatives) out of all predictions made. It gives an overall measure of how well the model is performing.
# precision: the proportion of true positive predictions out of all positive predictions made by the model. It indicates how many of the predicted purchasers were actually purchasers.
#  recall: the proportion of true positive predictions out of all actual positive cases in the dataset. It indicates how many of the actual purchasers were correctly identified by the model.
#  F1 score:- the harmonic mean of precision and recall, providing a single metric that balances both. It is especially useful when there is an imbalance between the classes (e.g., more non-purchasers than purchasers).
#  ROC:- AUC (Area Under the Receiver Operating Characteristic Curve) measures the model's ability to distinguish between classes. It ranges from 0 to 1, with higher values indicating better performance. An AUC of 0.5 suggests no discriminative power (random guessing), while an AUC of 1.0 indicates perfect discrimination.
#  AUC:- the area under the ROC curve, which is a plot of the true positive rate against the false positive rate at various threshold settings. It provides a single metric to evaluate the model's performance across all classification thresholds.
#  classification report:- a detailed report that includes precision, recall, F1 score, and support (number of occurrences of each class in the test set) for each class. It provides a comprehensive overview of the model's performance on each class.
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
) 


# Used to suppress warnings during model training and evaluation, such as convergence warnings 
# from logistic regression or warnings about class imbalance. 
# This helps to keep the output clean and focused on the key results.
import warnings 
warnings.filterwarnings('ignore')

# Configuration
# Define file paths and feature/target columns. Adjust these paths as needed for your environment.
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Get the parent directory of the current file
FEATURES_PATH = os.path.join(BASE_PATH, "data/features/user_features") # Path to the directory containing the feature data (Delta Lake or parquet files)
MODEL_PATH = os.path.join(BASE_PATH, "data/models") # Path to the directory where the trained model, scaler, and metrics will be saved
MODEL_FILE = os.path.join(MODEL_PATH, "purchase_predictor.pkl") # Path to save the trained logistic regression model as a pickle file
SCALER_FILE = os.path.join(MODEL_PATH, "scaler.pkl") # Path to save the feature scaler used for scaling the training data, also as a pickle file
METRICS_FILE = os.path.join(MODEL_PATH, "metrics.json") # Path to save the model performance metrics in JSON format, which will include accuracy, precision, recall, F1 score, ROC AUC, confusion matrix, and feature importance






# Feature columns for training (excluding target and metadata)
# These are the features that will be used to train the logistic regression model.
FEATURE_COLUMNS = [
    "purchases_last_24h",
    "total_purchases",
    "total_revenue",
    "avg_order_value",
    "max_purchase_value",
    "min_purchase_value",
    "total_events",
    "view_count",
    "cart_count",
    "unique_products",
    "activity_span_hours",
    "events_per_hour",
    "view_to_cart_ratio",
    "cart_to_purchase_ratio",
    "overall_conversion"
]

TARGET_COLUMN = "is_purchaser" # The target variable we want to predict, indicating whether a user made a purchase (1) or not (0)




# This functions implement the machine learning pipeline 
# including:- 
# loading features  
# preparing data 
# training the model 
# evaluating performance
# saving artifacts. 

# Each function includes print statements to provide feedback on the process and results.
def load_features():
    """
    Load feature data from Delta Lake using pandas.
    Falls back to parquet if delta-rs not available.

    # Note: Delta Lake stores data in parquet format, so we can read it directly with pandas if delta-rs is not available.
    # This function first tries to read parquet files from the specified directory. 
    #     - If no parquet files are found, it attempts to use delta-rs to read the Delta Lake format. 
    #     - If both methods fail, it raises an error.
    
    """
    print(f"Loading features from: {FEATURES_PATH}")

    try:
        # Prefer delta-rs to read only active Delta files (avoids stale parquet)
        try:
            from deltalake import DeltaTable
            dt = DeltaTable(FEATURES_PATH)
            df = dt.to_pandas()
        except Exception:
            # Fallback: read parquet files directly
            import glob
            parquet_files = glob.glob(f"{FEATURES_PATH}/*.parquet")
            if parquet_files:
                df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
            else:
                raise FileNotFoundError(f"No parquet files found in {FEATURES_PATH}")
        
        print(f"✓ Loaded {len(df)} records")
        return df
    except Exception as e:
        print(f"✗ Failed to load features: {e}")
        raise # Re-raise the exception after logging the error message




# This function prepares the data for training by splitting the features and target variable. 
# It also handles missing values and infinity in the feature data, and prints the class distribution of the target variable to provide insight into potential class imbalance issues.
# The function returns the feature matrix (X), target vector (y), and the list of available feature names that were used for training.
def prepare_data(df):
    """
    Prepare data for training - split features and target.
    """
    print("\nPreparing data for training...")
    
    # Select only feature columns that exist
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns] # Create a list of feature columns that are present in the DataFrame, ensuring that we only use features that are available for training
    
    if len(available_features) < len(FEATURE_COLUMNS): # Check if any expected feature columns are missing from the DataFrame by comparing the length of the available features with the length of the expected feature columns
        missing = set(FEATURE_COLUMNS) - set(available_features) # Identify any expected feature columns that are missing from the DataFrame by comparing the list of expected feature columns with the list of available feature columns
        print(f"  Warning: Missing features: {missing}") # Print a warning message if any of the expected feature columns are missing from the DataFrame, which could impact model performance
    
    print(f"  Using {len(available_features)} features")
    
    # Extract features and target
    X = df[available_features].copy() # Create a feature matrix (X) by selecting the available feature columns from the DataFrame and making a copy to avoid modifying the original DataFrame
    y = df[TARGET_COLUMN].copy() # Create a target vector (y) by selecting the target column from the DataFrame and making a copy to avoid modifying the original DataFrame
    
    # Handle missing values and infinity
    import numpy as np
    X = X.replace([np.inf, -np.inf], np.nan) # Replace any infinite values in the feature matrix (X) with NaN to handle cases where features may have infinite values due to division by zero or other issues
    X = X.fillna(0) # Fill any missing values (NaN) in the feature matrix (X) with 0, which is a common strategy for handling missing data in machine learning pipelines. 
                    # This ensures that the model can be trained without errors due to missing values
     
    # Print class distribution
    print(f"\nClass distribution:")
    print(f"  Non-purchasers (0): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"  Purchasers (1): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
    
    return X, y, available_features



# This function trains a logistic regression model using the prepared feature matrix (X) and target vector (y). 
# It performs a train/test split to evaluate the model's performance on unseen data, scales the features using StandardScaler, and handles class imbalance by setting class_weight to 'balanced' in the logistic regression model. 
# The function returns the trained model, the scaler used for feature scaling, the scaled training and testing feature matrices, and the corresponding target vectors for training and testing.
def train_model(X, y):
    """
    Train logistic regression model with train/test split.
    """
    print("\nTraining model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
    ) # Split the feature matrix (X) and target vector (y) into training and testing sets using an 80/20 split. 
       # The random_state parameter is set to 42 for reproducibility, ensuring that the same 
         # split is generated each time the code is run.
    
    print(f"  Training samples: {len(X_train)}") 
    print(f"  Test samples: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler() # Create an instance of StandardScaler, which will be used to standardize the feature data by removing the mean and scaling to unit variance.
    X_train_scaled = scaler.fit_transform(X_train) # Fit the scaler to the training feature matrix (X_train) and transform it to create a scaled version of the training features (X_train_scaled).
    X_test_scaled = scaler.transform(X_test)  # Use the fitted scaler to transform the testing feature matrix (X_test) to create a scaled version of the testing features (X_test_scaled).
    
    # Train logistic regression
    model = LogisticRegression(
        max_iter=1000, # Set a higher maximum number of iterations to ensure convergence, especially if the dataset is large or has many features.
        random_state=42, # Set random_state for reproducibility, ensuring that the same model parameters are generated each time the code is run.
        class_weight='balanced',  # Handle class imbalance
        solver='lbfgs' # Use the 'lbfgs' solver, which is an optimization algorithm that is efficient for logistic regression and can handle L2 regularization.
    )
    
    model.fit(X_train_scaled, y_train) # Fit the logistic regression model to the scaled training feature matrix (X_train_scaled) and the corresponding target vector (y_train).
    print("✓ Model trained")
    
    return model, scaler, X_train_scaled, X_test_scaled, y_train, y_test





# This function evaluates the performance of the trained logistic regression model on the testing data. 
# It calculates various performance metrics such as accuracy, precision, recall, F1 score, and ROC AUC, and also generates a confusion matrix to visualize the true positives, true negatives, false positives, and false negatives. 
# Additionally, it extracts feature importance from the model coefficients and prints the top 5 most important features. The function returns a dictionary containing all the calculated metrics, the confusion matrix, and the feature importance for further analysis or saving to disk.
def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluate model performance and return metrics.    
    """
    print("\nEvaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test) # Use the trained logistic regression model to make predictions on the scaled testing feature matrix (X_test) and store the predicted class labels in y_pred.
    y_pred_proba = model.predict_proba(X_test)[:, 1] if len(model.classes_) > 1 else model.predict_proba(X_test)[:, 0] # Get predicted probabilities for the positive class (class 1) if there are multiple classes, otherwise get probabilities for the single class. This is used for calculating ROC AUC.
    
    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)), # Calculate the accuracy of the model by comparing the true labels (y_test) with the predicted labels (y_pred) and convert it to a float for easier storage in JSON format.
        "precision": float(precision_score(y_test, y_pred, zero_division=0)), # Calculate the precision of the model, which is the proportion of true positive predictions out of all positive predictions made by the model. The zero_division=0 parameter ensures that if there are no positive predictions, the precision will be set to 0 instead of raising an error.
        "recall": float(recall_score(y_test, y_pred, zero_division=0)), # Calculate the recall of the model, which is the proportion of true positive predictions out of all actual positive cases in the dataset. The zero_division=0 parameter ensures that if there are no actual positive cases, the recall will be set to 0 instead of raising an error.
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)), # Calculate the F1 score of the model, which is the harmonic mean of precision and recall. The zero_division=0 parameter ensures that if there are no positive predictions or actual positive cases, the F1 score will be set to 0 instead of raising an error.
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)) if len(np.unique(y_test)) > 1 else 0.0, # Calculate the ROC AUC score of the model using the true labels (y_test) and the predicted probabilities (y_pred_proba). If there is only one class in y_test, set ROC AUC to 0.0 since it cannot be calculated.
        "training_date": datetime.now().isoformat(), # Add a timestamp for when the model was trained, formatted as an ISO 8601 string for easy storage and readability.
        "n_samples": int(len(y_test)),  # Include the number of samples in the test set to provide context for the performance metrics, which can help in understanding the reliability of the results.
        "n_features": int(len(feature_names)) # Include the number of features used in the model to provide context for the performance metrics, which can help in understanding the complexity of the model and its potential for overfitting or underfitting.
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred) # Generate a confusion matrix to visualize the performance of the model in terms of true positives, true negatives, false positives, and false negatives by comparing the true labels (y_test) with the predicted labels (y_pred). The confusion matrix is stored as a list in the metrics dictionary for easy storage in JSON format.
    metrics["confusion_matrix"] = cm.tolist() # Convert the confusion matrix to a list format for JSON serialization, as numpy arrays cannot be directly serialized to JSON.
    
    # Feature importance (coefficients)
    feature_importance = dict(zip(feature_names, model.coef_[0].tolist())) # Extract feature importance from the logistic regression model coefficients by creating a dictionary that maps each feature name to its corresponding coefficient value. The coefficients indicate the strength and direction of the relationship between each feature and the target variable, with higher absolute values indicating more important features for predicting the target variable. This information is stored in the metrics dictionary for further analysis or saving to disk.
    metrics["feature_importance"] = feature_importance # Add the feature importance dictionary to the metrics dictionary for easy storage in JSON format, allowing for analysis of which features are most influential in the model's predictions.
    
    # Print results
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    print("=" * 60)
    
    print("\nConfusion Matrix:")
    print(f"  TN: {cm[0][0]:>6}  FP: {cm[0][1]:>6}")
    print(f"  FN: {cm[1][0]:>6}  TP: {cm[1][1]:>6}")
    
    print("\nTop 5 Important Features:")
    sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    for feature, importance in sorted_features[:5]:
        print(f"  {feature}: {importance:.4f}")
    
    return metrics



# This function saves the trained model, the scaler used for feature scaling, and the performance metrics to disk. 
# The model and scaler are saved as pickle files for easy loading in future inference or retraining, while the metrics are saved in JSON format for easy analysis and tracking of model performance over time.
def save_artifacts(model, scaler, metrics, feature_names): # The function takes the trained model, the scaler used for feature scaling, the performance metrics, and the list of feature names as input parameters to save them to disk.
    """
    Save model, scaler, and metrics to disk.

    """
    print(f"\nSaving artifacts to: {MODEL_PATH}")
    
    # Create directory if not exists
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Save model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Model saved: {MODEL_FILE}")
    
    # Save scaler
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  ✓ Scaler saved: {SCALER_FILE}")
    
    # Save metrics
    metrics["feature_names"] = feature_names
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Metrics saved: {METRICS_FILE}")



# This main function orchestrates the entire machine learning training pipeline by calling the individual functions for loading features, preparing data, training the model, evaluating performance, and saving artifacts.
def main():
    """
    Main function to run the ML training pipeline.
    """
    print("=" * 60)
    print("Machine Learning Training Pipeline")
    print("=" * 60)
    print(f"Features Path: {FEATURES_PATH}")
    print(f"Model Path: {MODEL_PATH}")
    print("=" * 60)
    
    # Load features
    df = load_features()
    
    # Prepare data
    X, y, feature_names = prepare_data(df)
    
    # Check if we have enough data
    if len(X) < 10:
        print(f"\n⚠ Warning: Only {len(X)} samples. Need more data for reliable training.")
        print("Generating synthetic training anyway for demonstration...")
    
    # Train model
    model, scaler, X_train, X_test, y_train, y_test = train_model(X, y)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, feature_names)
    
    # Save artifacts
    save_artifacts(model, scaler, metrics, feature_names)
    
    print("\n" + "=" * 60)
    print("ML Training Pipeline Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
