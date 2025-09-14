import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List, Any, Optional
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class MultiTargetAnomalyDetector:
    """Multi-target anomaly detection for payroll validation."""
    
    def __init__(self, config=None):
        self.config = config
        self.feature_engineer = None
        self.models = {}
        self.best_model = None
        self.target_columns = ['CODE_DAY_MATCH', 'CODE_PP_MATCH', 'SyncedWFMMeditch']
        
    def coerce_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map all targets to binary: 1 = anomaly (not matched), 0 = normal (matched)."""
        df_targets = df.copy()
        
        for target_col in self.target_columns:
            if target_col in df_targets.columns:
                # Convert to string and normalize
                target_series = df_targets[target_col].astype(str).str.strip().str.lower()
                
                # Mapping logic for each target
                def map_target(v):
                    if pd.isna(v) or v == 'nan':
                        return np.nan
                    elif 'not' in str(v).lower() or 'mismatch' in str(v).lower() or v == '0':
                        return 1  # Anomaly
                    elif 'match' in str(v).lower() or v == '1':
                        return 0  # Normal
                    else:
                        return np.nan
                
                df_targets[target_col] = target_series.apply(map_target)
                logger.info(f"Target {target_col} mapping: {df_targets[target_col].value_counts()}")
        
        return df_targets
    
    def add_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific features for anomaly detection."""
        df_features = df.copy()
        
        # Hours-related features (if available)
        hour_cols = [col for col in df.columns if 'hour' in col.lower()]
        if len(hour_cols) >= 2:
            df_features['hours_diff'] = df_features[hour_cols[0]] - df_features[hour_cols[1]]
            df_features['hours_absdiff'] = df_features['hours_diff'].abs()
            df_features['hours_ratio'] = np.where(
                df_features[hour_cols[1]].abs() > 1e-6,
                df_features[hour_cols[0]] / df_features[hour_cols[1]], 
                np.nan
            )
        
        # Time-based features
        time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        for col in time_cols[:1]:  # Process first time column found
            try:
                df_features[col] = pd.to_datetime(df_features[col], errors='coerce')
                df_features['day_of_week'] = df_features[col].dt.dayofweek
                df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
                df_features['month'] = df_features[col].dt.month
                break
            except:
                continue
        
        # Code matching features (look for columns with 'code' in name)
        code_cols = [col for col in df.columns if 'code' in col.lower()]
        if len(code_cols) >= 2:
            df_features['codes_match'] = (
                df_features[code_cols[0]].astype(str) == df_features[code_cols[1]].astype(str)
            ).astype(int)
        
        # Partial shift indicator
        if 'Partial' in df_features.columns:
            df_features['is_partial'] = df_features['Partial'].astype(int)
        
        # Paid hours analysis
        if 'PaidHours' in df_features.columns:
            df_features['paid_hours_log'] = np.log1p(df_features['PaidHours'].abs())
            df_features['is_zero_hours'] = (df_features['PaidHours'] == 0).astype(int)
        
        logger.info("Domain features added successfully")
        return df_features
    
    def build_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """Build preprocessing pipeline based on available columns."""
        
        # Identify column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target columns from features
        numeric_cols = [col for col in numeric_cols if col not in self.target_columns]
        categorical_cols = [col for col in categorical_cols if col not in self.target_columns]
        
        # Separate high and low cardinality categoricals
        cat_low_card = []
        cat_high_card = []
        
        for col in categorical_cols:
            unique_vals = df[col].nunique()
            if unique_vals <= 10:
                cat_low_card.append(col)
            elif unique_vals <= 100:
                cat_high_card.append(col)
            # Skip very high cardinality columns (>100 unique values)
        
        transformers = []
        
        if numeric_cols:
            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_transformer, numeric_cols))
        
        if cat_low_card:
            cat_low_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
            ])
            transformers.append(('cat_low', cat_low_transformer, cat_low_card))
        
        if cat_high_card:
            cat_high_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=50))
            ])
            transformers.append(('cat_high', cat_high_transformer, cat_high_card))
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        logger.info(f"Preprocessor built: {len(numeric_cols)} numeric, {len(cat_low_card)} low-card categorical, {len(cat_high_card)} high-card categorical")
        return preprocessor
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for training."""
        
        # Add domain features
        df_processed = self.add_domain_features(df)
        
        # Process targets
        df_with_targets = self.coerce_targets(df_processed)
        
        # Remove rows where all targets are NaN
        target_mask = df_with_targets[self.target_columns].notna().any(axis=1)
        df_clean = df_with_targets[target_mask].copy()
        
        # For rows with some missing targets, fill with 0 (assume normal if not specified)
        for col in self.target_columns:
            df_clean[col] = df_clean[col].fillna(0)
        
        # Extract targets
        y = df_clean[self.target_columns].values.astype(int)
        
        # Build preprocessor and transform features
        preprocessor = self.build_preprocessor(df_clean)
        X = preprocessor.fit_transform(df_clean)
        
        self.preprocessor = preprocessor
        
        logger.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} targets")
        logger.info(f"Target distributions: {[np.bincount(y[:, i]) for i in range(y.shape[1])]}")
        
        return X, y
    
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize multi-output models."""
        
        models = {
            'random_forest_multi': MultiOutputClassifier(
                RandomForestClassifier(
                    n_estimators=100,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
            ),
            'logistic_multi': MultiOutputClassifier(
                LogisticRegression(
                    class_weight='balanced',
                    random_state=42,
                    max_iter=1000
                )
            )
        }
        
        return models
    
    def evaluate_multi_output(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Evaluate multi-output predictions."""
        
        results = {}
        
        # Overall metrics
        overall_accuracy = np.mean([accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])
        results['overall_accuracy'] = overall_accuracy
        
        # Per-target metrics
        target_metrics = {}
        for i, target in enumerate(self.target_columns):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true[:, i], y_pred[:, i], average='binary', zero_division=0
            )
            
            target_metrics[target] = {
                'accuracy': accuracy_score(y_true[:, i], y_pred[:, i]),
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        results['target_metrics'] = target_metrics
        
        # Combined anomaly detection (any target is 1)
        y_true_any = (y_true.sum(axis=1) > 0).astype(int)
        y_pred_any = (y_pred.sum(axis=1) > 0).astype(int)
        
        if len(np.unique(y_true_any)) > 1:  # Check if we have both classes
            precision_any, recall_any, f1_any, _ = precision_recall_fscore_support(
                y_true_any, y_pred_any, average='binary', zero_division=0
            )
            
            results['combined_anomaly'] = {
                'accuracy': accuracy_score(y_true_any, y_pred_any),
                'precision': precision_any,
                'recall': recall_any,
                'f1_score': f1_any
            }
        
        return results
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train all multi-output models."""
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y[:, 0]  # Stratify on first target
        )
        
        # Initialize models
        models = self.initialize_models()
        
        results = {}
        best_f1 = 0
        
        logger.info("Starting multi-target model training...")
        
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Evaluate
                metrics = self.evaluate_multi_output(y_test, y_pred)
                
                results[model_name] = metrics
                self.models[model_name] = model
                
                # Track best model based on overall F1
                avg_f1 = np.mean([metrics['target_metrics'][target]['f1_score'] for target in self.target_columns])
                
                logger.info(f"{model_name} - Overall Accuracy: {metrics['overall_accuracy']:.3f}, Avg F1: {avg_f1:.3f}")
                
                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    self.best_model = model
                    self.best_model_name = model_name
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        logger.info(f"Best model: {self.best_model_name} with average F1 score: {best_f1:.3f}")
        return results
    
    def predict_multi_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for all targets."""
        
        if self.best_model is None:
            raise ValueError("No trained model available. Train model first.")
        
        # Process features same way as training
        df_processed = self.add_domain_features(df)
        X = self.preprocessor.transform(df_processed)
        
        # Get predictions and probabilities
        predictions = self.best_model.predict(X)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(self.best_model, 'predict_proba'):
            # For multi-output, predict_proba returns list of arrays
            prob_arrays = self.best_model.predict_proba(X)
            probabilities = np.column_stack([prob[:, 1] for prob in prob_arrays])
        
        # Create results dataframe
        results_df = df.copy()
        
        # Add predictions
        for i, target in enumerate(self.target_columns):
            results_df[f'{target}_Prediction'] = ['Anomaly' if p == 1 else 'Normal' for p in predictions[:, i]]
            
            if probabilities is not None:
                results_df[f'{target}_Confidence'] = probabilities[:, i]
        
        # Add combined anomaly flag
        results_df['Any_Anomaly'] = ['Yes' if pred.sum() > 0 else 'No' for pred in predictions]
        
        if probabilities is not None:
            # Combined confidence as max probability across targets
            results_df['Any_Anomaly_Confidence'] = probabilities.max(axis=1)
        
        logger.info(f"Multi-target predictions completed for {len(df)} samples")
        return results_df
    
    def save_model(self, filepath: str):
        """Save the trained model and preprocessor."""
        model_data = {
            'model': self.best_model,
            'preprocessor': self.preprocessor,
            'target_columns': self.target_columns,
            'best_model_name': self.best_model_name
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and preprocessor."""
        model_data = joblib.load(filepath)
        
        self.best_model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.target_columns = model_data['target_columns']
        self.best_model_name = model_data.get('best_model_name', 'loaded_model')
        
        logger.info(f"Model loaded from {filepath}")

# Usage example
def main():
    """Example usage of the multi-target anomaly detector."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load your data
    # df = pd.read_csv('your_payroll_data.csv')
    
    # Initialize detector
    detector = MultiTargetAnomalyDetector()
    
    # Train models
    # training_results = detector.train_models(df)
    
    # Make predictions
    # predictions_df = detector.predict_multi_target(df)
    
    # Save model
    # detector.save_model('multi_target_anomaly_model.joblib')
    
    print("Multi-target anomaly detection model ready!")

if __name__ == "__main__":
    main()