import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score
)
import joblib
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class SingleTargetAnomalyDetector:
    """
    Single target anomaly detection for payroll validation.
    Template that can be customized for different target columns.
    """
    
    def __init__(self, target_column: str, model_name: str = "anomaly_detector"):
        self.target_column = target_column
        self.model_name = model_name
        self.preprocessor = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.training_results = {}
        
    def coerce_target(self, y: pd.Series) -> pd.Series:
        """Map target to binary: 1 = not matched (anomaly), 0 = matched."""
        y = y.astype(str).str.strip().str.lower()
        
        def map_func(v):
            if pd.isna(v) or v == 'nan':
                return np.nan
            elif 'not' in str(v).lower() or 'mismatch' in str(v).lower():
                return 1  # Anomaly
            elif 'match' in str(v).lower():
                return 0  # Normal
            else:
                return np.nan
        
        result = y.map(map_func)
        logger.info(f"Target {self.target_column} mapping: {result.value_counts()}")
        return result
    
    def add_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific features for anomaly detection."""
        df_features = df.copy()
        
        # Hours-related features
        if 'ESP_Hours' in df_features.columns and 'WFM_Hours' in df_features.columns:
            df_features['hours_diff'] = df_features['ESP_Hours'] - df_features['WFM_Hours']
            df_features['hours_absdiff'] = df_features['hours_diff'].abs()
            
            # Avoid division by zero
            df_features['hours_ratio'] = np.where(
                df_features['WFM_Hours'].abs() > 1e-6,
                df_features['ESP_Hours'] / df_features['WFM_Hours'], 
                np.nan
            )
            
            df_features['hours_min'] = np.minimum(df_features['ESP_Hours'], df_features['WFM_Hours'])
            df_features['hours_max'] = np.maximum(df_features['ESP_Hours'], df_features['WFM_Hours'])
        
        # Time-based features using WorkedDay
        if 'WorkedDay' in df_features.columns:
            try:
                df_features['WorkedDay'] = pd.to_datetime(df_features['WorkedDay'], errors='coerce')
                df_features['day_of_week'] = df_features['WorkedDay'].dt.dayofweek
                df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
                df_features['month'] = df_features['WorkedDay'].dt.month
                df_features['day_of_month'] = df_features['WorkedDay'].dt.day
            except:
                pass
        
        # Code matching features
        if 'ESP_EarnCode' in df_features.columns and 'WFM_EarnCode' in df_features.columns:
            df_features['earn_match'] = (
                df_features['ESP_EarnCode'].astype(str) == df_features['WFM_EarnCode'].astype(str)
            ).astype(int)
        
        if 'ESP_CostCentre' in df_features.columns and 'WFM_CostCentre' in df_features.columns:
            df_features['costcentre_match'] = (
                df_features['ESP_CostCentre'].astype(str) == df_features['WFM_CostCentre'].astype(str)
            ).astype(int)
        
        # Interaction features
        if 'hours_absdiff' in df_features.columns and 'earn_match' in df_features.columns:
            df_features['absdiff_x_earnmismatch'] = df_features['hours_absdiff'] * (1 - df_features['earn_match'])
        
        if 'hours_absdiff' in df_features.columns and 'costcentre_match' in df_features.columns:
            df_features['absdiff_x_ccmismatch'] = df_features['hours_absdiff'] * (1 - df_features['costcentre_match'])
        
        # Handle missing values for hours columns
        for col in ['ESP_Hours', 'WFM_Hours']:
            if col in df_features.columns:
                df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
        
        logger.info("Domain features added successfully")
        return df_features
    
    def build_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """Build preprocessing pipeline."""
        
        # Identify column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target column from features
        numeric_cols = [col for col in numeric_cols if col != self.target_column]
        categorical_cols = [col for col in categorical_cols if col != self.target_column]
        
        # Remove ID columns and other non-predictive columns
        id_cols = ['Number', 'WorkedDay']
        numeric_cols = [col for col in numeric_cols if col not in id_cols]
        categorical_cols = [col for col in categorical_cols if col not in id_cols]
        
        # Separate high and low cardinality categoricals
        cat_low_card = []
        cat_high_card = []
        
        for col in categorical_cols:
            unique_vals = df[col].nunique()
            if unique_vals <= 10:
                cat_low_card.append(col)
            elif unique_vals <= 50:
                cat_high_card.append(col)
            # Skip very high cardinality columns
        
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
                ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=20))
            ])
            transformers.append(('cat_high', cat_high_transformer, cat_high_card))
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        self.preprocessor = preprocessor
        logger.info(f"Preprocessor built: {len(numeric_cols)} numeric, {len(cat_low_card)} low-card categorical, {len(cat_high_card)} high-card categorical")
        return preprocessor
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for training."""
        
        # Add domain features
        df_processed = self.add_domain_features(df)
        
        # Process target
        y = self.coerce_target(df_processed[self.target_column])
        
        # Remove rows where target is NaN
        valid_mask = ~y.isna()
        df_clean = df_processed[valid_mask].copy()
        y_clean = y[valid_mask].astype(int)
        
        # Build preprocessor and transform features
        preprocessor = self.build_preprocessor(df_clean)
        X = preprocessor.fit_transform(df_clean)
        
        logger.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target distribution: {np.bincount(y_clean)}")
        
        return X, y_clean
    
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize models with their configurations."""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'logistic_regression': LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
        }
        return models
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance."""
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        if y_proba is not None:
            auc = roc_auc_score(y_test, y_proba)
            metrics['auc_roc'] = float(auc)
        
        # Classification report
        class_report = classification_report(
            y_test, y_pred, 
            target_names=['Match', 'Not Matched'],
            output_dict=True
        )
        metrics['classification_report'] = class_report
        
        return metrics
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train all models and select the best one."""
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize models
        models = self.initialize_models()
        
        results = {}
        best_f1 = 0
        
        logger.info(f"Starting model training for {self.target_column}...")
        
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Create pipeline
                pipeline = Pipeline([
                    ('pre', self.preprocessor),
                    ('clf', model)
                ])
                
                # Hyperparameter tuning
                if model_name == 'random_forest':
                    param_grid = {
                        'clf__n_estimators': [100, 200],
                        'clf__max_depth': [10, 20, None],
                        'clf__min_samples_split': [2, 5]
                    }
                elif model_name == 'logistic_regression':
                    param_grid = {
                        'clf__C': [0.1, 1, 10],
                        'clf__max_iter': [1000, 2000]
                    }
                
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                search = RandomizedSearchCV(
                    pipeline, param_grid, n_iter=10, cv=cv,
                    scoring='f1', random_state=42, n_jobs=-1
                )
                
                # Fit the model
                search.fit(X_train, y_train)
                
                # Evaluate
                metrics = self.evaluate_model(search, X_test, y_test)
                
                results[model_name] = metrics
                self.models[model_name] = search
                
                logger.info(f"{model_name} - F1: {metrics['f1_score']:.3f}, Accuracy: {metrics['accuracy']:.3f}")
                
                # Track best model
                if metrics['f1_score'] > best_f1:
                    best_f1 = metrics['f1_score']
                    self.best_model = search
                    self.best_model_name = model_name
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        self.training_results = results
        logger.info(f"Best model: {self.best_model_name} with F1 score: {best_f1:.3f}")
        return results
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on new data."""
        
        if self.best_model is None:
            raise ValueError("No trained model available. Train model first.")
        
        # Process features same way as training
        df_processed = self.add_domain_features(df)
        
        # Get predictions and probabilities
        predictions = self.best_model.predict(df_processed)
        probabilities = None
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(df_processed)[:, 1]
        
        # Create results dataframe
        results_df = df.copy()
        results_df[f'{self.target_column}_Prediction'] = ['Anomaly' if p == 1 else 'Match' for p in predictions]
        
        if probabilities is not None:
            results_df[f'{self.target_column}_Confidence'] = probabilities
        
        logger.info(f"Predictions completed for {len(df)} samples")
        return results_df
    
    def save_model(self, filepath: str):
        """Save the trained model and preprocessor."""
        model_data = {
            'model': self.best_model,
            'preprocessor': self.preprocessor,
            'target_column': self.target_column,
            'model_name': self.model_name,
            'best_model_name': self.best_model_name,
            'training_results': self.training_results
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and preprocessor."""
        model_data = joblib.load(filepath)
        
        self.best_model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.target_column = model_data['target_column']
        self.model_name = model_data.get('model_name', 'loaded_model')
        self.best_model_name = model_data.get('best_model_name', 'unknown')
        self.training_results = model_data.get('training_results', {})
        
        logger.info(f"Model loaded from {filepath}")


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Single Target Anomaly Detection")
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--target', required=True, help='Target column name')
    parser.add_argument('--output', default='results', help='Output directory')
    parser.add_argument('--model-name', default='anomaly_detector', help='Model name')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Load data
        logger.info(f"Loading data from {args.input}")
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df):,} records")
        
        # Initialize detector
        detector = SingleTargetAnomalyDetector(args.target, args.model_name)
        
        # Train model
        results = detector.train_models(df)
        
        # Make predictions
        predictions_df = detector.predict(df)
        
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True)
        
        # Save results
        predictions_df.to_csv(output_path / f"{args.model_name}_predictions.csv", index=False)
        detector.save_model(str(output_path / f"{args.model_name}_model.joblib"))
        
        # Save training results
        with open(output_path / f"{args.model_name}_training_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        anomaly_count = (predictions_df[f'{args.target}_Prediction'] == 'Anomaly').sum()
        print(f"\nResults for {args.target}:")
        print(f"Total records: {len(df):,}")
        print(f"Anomalies detected: {anomaly_count:,} ({anomaly_count/len(df):.2%})")
        print(f"Best model: {detector.best_model_name}")
        print(f"F1 Score: {detector.training_results[detector.best_model_name]['f1_score']:.3f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())