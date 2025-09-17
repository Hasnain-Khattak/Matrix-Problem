import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score
)
import joblib
import json
import argparse
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SingleTargetDetector:
    """Complete anomaly detector for a single target."""
    
    def __init__(self, target_column: str, model_name: str):
        self.target_column = target_column
        self.model_name = model_name
        self.preprocessor = None
        self.best_model = None
        self.best_model_name = None
        self.training_results = {}
        
    def coerce_target(self, y: pd.Series) -> pd.Series:
        """Map target to binary: 1 = anomaly, 0 = normal."""
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
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add domain-specific features."""
        df_features = df.copy()
        
        # Hours-related features
        if 'ESP_Hours' in df_features.columns and 'WFM_Hours' in df_features.columns:
            df_features['hours_diff'] = df_features['ESP_Hours'] - df_features['WFM_Hours']
            df_features['hours_absdiff'] = df_features['hours_diff'].abs()
            df_features['hours_ratio'] = np.where(
                df_features['WFM_Hours'].abs() > 1e-6,
                df_features['ESP_Hours'] / df_features['WFM_Hours'], 
                np.nan
            )
        
        # Time features
        if 'WorkedDay' in df_features.columns:
            try:
                df_features['WorkedDay'] = pd.to_datetime(df_features['WorkedDay'], errors='coerce')
                df_features['day_of_week'] = df_features['WorkedDay'].dt.dayofweek
                df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
                df_features['month'] = df_features['WorkedDay'].dt.month
            except:
                pass
        
        # Code matching
        if 'ESP_EarnCode' in df_features.columns and 'WFM_EarnCode' in df_features.columns:
            df_features['earn_match'] = (
                df_features['ESP_EarnCode'].astype(str) == df_features['WFM_EarnCode'].astype(str)
            ).astype(int)
        
        if 'ESP_CostCentre' in df_features.columns and 'WFM_CostCentre' in df_features.columns:
            df_features['costcentre_match'] = (
                df_features['ESP_CostCentre'].astype(str) == df_features['WFM_CostCentre'].astype(str)
            ).astype(int)
        
        # Clean numeric columns
        for col in ['ESP_Hours', 'WFM_Hours']:
            if col in df_features.columns:
                df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
        
        return df_features
    
    def build_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """Build preprocessing pipeline."""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target and ID columns
        exclude_cols = [self.target_column, 'Number', 'WorkedDay']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        # Separate categorical by cardinality
        cat_low_card = []
        cat_high_card = []
        
        for col in categorical_cols:
            unique_vals = df[col].nunique()
            if unique_vals <= 10:
                cat_low_card.append(col)
            elif unique_vals <= 50:
                cat_high_card.append(col)
        
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
        return preprocessor
    
    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Train the model."""
        
        # Add features
        train_processed = self.add_features(train_df)
        test_processed = self.add_features(test_df)
        
        # Process targets
        y_train = self.coerce_target(train_processed[self.target_column])
        y_test = self.coerce_target(test_processed[self.target_column])
        
        # Remove rows with missing targets
        train_mask = ~y_train.isna()
        test_mask = ~y_test.isna()
        
        train_clean = train_processed[train_mask].copy()
        test_clean = test_processed[test_mask].copy()
        y_train_clean = y_train[train_mask].astype(int)
        y_test_clean = y_test[test_mask].astype(int)
        
        # Build preprocessor
        preprocessor = self.build_preprocessor(train_clean)
        X_train = preprocessor.fit_transform(train_clean)
        X_test = preprocessor.transform(test_clean)
        
        # Train models
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
        
        results = {}
        best_f1 = 0
        
        for model_name, model in models.items():
            logger.info(f"Training {model_name} for {self.target_column}...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('pre', preprocessor),
                ('clf', model)
            ])
            
            # Simple hyperparameter tuning
            if model_name == 'random_forest':
                param_grid = {
                    'clf__n_estimators': [100, 200],
                    'clf__max_depth': [10, 20, None]
                }
            else:
                param_grid = {
                    'clf__C': [0.1, 1, 10]
                }
            
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            search = RandomizedSearchCV(
                pipeline, param_grid, n_iter=6, cv=cv,
                scoring='f1', random_state=42, n_jobs=-1
            )
            
            # Fit model
            search.fit(train_clean, y_train_clean)
            
            # Evaluate on test set
            y_pred = search.predict(test_clean)
            y_proba = search.predict_proba(test_clean)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_clean, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test_clean, y_pred, average='binary', zero_division=0
            )
            auc = roc_auc_score(y_test_clean, y_proba)
            
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc_roc': float(auc),
                'confusion_matrix': confusion_matrix(y_test_clean, y_pred).tolist()
            }
            
            results[model_name] = metrics
            
            logger.info(f"{model_name} - F1: {f1:.3f}, Accuracy: {accuracy:.3f}")
            
            # Track best model
            if f1 > best_f1:
                best_f1 = f1
                self.best_model = search
                self.best_model_name = model_name
        
        self.training_results = results
        logger.info(f"Best model for {self.target_column}: {self.best_model_name} (F1: {best_f1:.3f})")
        return results
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions."""
        if self.best_model is None:
            raise ValueError("Model not trained")
        
        df_processed = self.add_features(df)
        predictions = self.best_model.predict(df_processed)
        probabilities = self.best_model.predict_proba(df_processed)[:, 1]
        
        results_df = df.copy()
        results_df[f'{self.target_column}_Prediction'] = ['Anomaly' if p == 1 else 'Match' for p in predictions]
        results_df[f'{self.target_column}_Confidence'] = probabilities
        
        return results_df
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        model_data = {
            'model': self.best_model,
            'preprocessor': self.preprocessor,
            'target_column': self.target_column,
            'model_name': self.model_name,
            'best_model_name': self.best_model_name,
            'training_results': self.training_results
        }
        joblib.dump(model_data, filepath)

def create_folder_structure():
    """Create the 3 model folders."""
    folders = [
        'Anomaly-Detection_CODE_DAY_MATCH',
        'Anomaly-Detection_CODE_PP_MATCH', 
        'Anomaly-Detection_SyncedWFMMeditech'
    ]
    
    for folder in folders:
        folder_path = Path(folder)
        folder_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (folder_path / 'data').mkdir(exist_ok=True)
        (folder_path / 'models').mkdir(exist_ok=True)
        (folder_path / 'outputs').mkdir(exist_ok=True)
        
        logger.info(f"Created folder: {folder}")

def split_and_train_all_models(input_file: str):
    """Split data and train all 3 models."""
    
    # Load data
    logger.info(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df):,} records")
    
    # Model configurations
    model_configs = [
        {
            'target': 'ESPorMTMandWFMHoursEarnCodeDayMatch',
            'folder': 'Anomaly-Detection_CODE_DAY_MATCH',
            'name': 'CODE_DAY_MATCH'
        },
        {
            'target': 'ESPorMTMandWFMHoursEarnCodePPMatch',
            'folder': 'Anomaly-Detection_CODE_PP_MATCH',
            'name': 'CODE_PP_MATCH'
        },
        {
            'target': 'SyncedWFMMeditech',
            'folder': 'Anomaly-Detection_SyncedWFMMeditech',
            'name': 'SyncedWFMMeditech'
        }
    ]
    
    all_results = {}
    
    for config in model_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {config['name']}")
        logger.info(f"{'='*60}")
        
        target = config['target']
        folder = config['folder']
        
        # Check if target exists
        if target not in df.columns:
            logger.error(f"Target column '{target}' not found!")
            continue
        
        # Filter valid records
        valid_mask = df[target].notna()
        df_valid = df[valid_mask].copy()
        
        if len(df_valid) < 10:
            logger.warning(f"Too few records for {target}: {len(df_valid)}")
            continue
        
        logger.info(f"Valid records: {len(df_valid):,}")
        logger.info(f"Target distribution: {df_valid[target].value_counts().to_dict()}")
        
        # Encode target for stratification
        y_encoded = df_valid[target].astype(str).str.lower().map({
            'match': 0,
            'not matched': 1
        })
        
        # Split data
        train_df, test_df = train_test_split(
            df_valid, 
            test_size=0.2, 
            random_state=42,
            stratify=y_encoded
        )
        
        logger.info(f"Split - Train: {len(train_df):,}, Test: {len(test_df):,}")
        
        # Save split data
        train_df.to_csv(f"{folder}/data/train_data.csv", index=False)
        test_df.to_csv(f"{folder}/data/test_data.csv", index=False)
        
        # Initialize and train model
        detector = SingleTargetDetector(target, config['name'])
        training_results = detector.train(train_df, test_df)
        
        # Save model
        model_file = f"{folder}/models/{config['name']}_model.joblib"
        detector.save_model(model_file)
        
        # Make test predictions
        test_predictions = detector.predict(test_df)
        
        # Calculate final metrics
        y_true = detector.coerce_target(test_df[target])
        y_pred = test_predictions[f'{target}_Prediction'].map({'Anomaly': 1, 'Match': 0})
        
        anomaly_count = (y_pred == 1).sum()
        anomaly_rate = anomaly_count / len(test_df)
        
        final_results = {
            'target_column': target,
            'folder': folder,
            'total_records': len(df_valid),
            'train_records': len(train_df),
            'test_records': len(test_df),
            'best_model': detector.best_model_name,
            'training_results': training_results,
            'test_anomalies': int(anomaly_count),
            'test_anomaly_rate': float(anomaly_rate)
        }
        
        # Save results
        with open(f"{folder}/outputs/{config['name']}_results.json", 'w') as f:
            json.dump(final_results, f, indent=2)
        
        test_predictions.to_csv(f"{folder}/outputs/{config['name']}_test_predictions.csv", index=False)
        
        # Generate summary report
        summary_report = f"""
{config['name']} Training Results
{'='*50}
Training Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Data Summary:
  Total Valid Records: {len(df_valid):,}
  Training Records: {len(train_df):,}
  Test Records: {len(test_df):,}
  Target Column: {target}

Best Model: {detector.best_model_name}
Test Performance:
  F1 Score: {training_results[detector.best_model_name]['f1_score']:.3f}
  Accuracy: {training_results[detector.best_model_name]['accuracy']:.3f}
  Precision: {training_results[detector.best_model_name]['precision']:.3f}
  Recall: {training_results[detector.best_model_name]['recall']:.3f}
  ROC AUC: {training_results[detector.best_model_name]['auc_roc']:.3f}

Anomaly Detection:
  Anomalies Found: {anomaly_count:,} ({anomaly_rate:.2%})
  
Files Created:
  Model: {model_file}
  Test Predictions: {folder}/outputs/{config['name']}_test_predictions.csv
  Results: {folder}/outputs/{config['name']}_results.json
"""
        
        with open(f"{folder}/outputs/{config['name']}_summary.txt", 'w') as f:
            f.write(summary_report)
        
        all_results[config['name']] = final_results
        
        # Print progress
        print(f"\n✅ {config['name']} COMPLETED")
        print(f"   F1 Score: {training_results[detector.best_model_name]['f1_score']:.3f}")
        print(f"   Anomalies: {anomaly_count:,} ({anomaly_rate:.2%})")
        print(f"   Files saved to: {folder}/")
    
    return all_results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Complete 3-Model Anomaly Detection Setup")
    parser.add_argument('--input', required=True, help='Input CSV file (anomaly_dataset.csv)')
    
    args = parser.parse_args()
    
    try:
        # Check input file
        if not Path(args.input).exists():
            logger.error(f"Input file not found: {args.input}")
            return 1
        
        print("🚀 Starting Complete 3-Model Setup...")
        print(f"Input file: {args.input}")
        
        # Create folder structure
        create_folder_structure()
        
        # Train all models
        results = split_and_train_all_models(args.input)
        
        # Print final summary
        print("\n" + "="*80)
        print("🎉 ALL 3 MODELS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        for model_name, result in results.items():
            best_model = result['best_model']
            f1_score = result['training_results'][best_model]['f1_score']
            anomaly_count = result['test_anomalies']
            anomaly_rate = result['test_anomaly_rate']
            
            print(f"\n📊 {model_name}:")
            print(f"   Folder: {result['folder']}/")
            print(f"   Best Model: {best_model}")
            print(f"   F1 Score: {f1_score:.3f}")
            print(f"   Test Anomalies: {anomaly_count:,} ({anomaly_rate:.2%})")
        
        print(f"\n📁 Each folder contains:")
        print(f"   ├── models/ (trained .joblib files)")
        print(f"   ├── outputs/ (predictions and results)")
        print(f"   └── data/ (train/test splits)")
        
        print(f"\n🎯 Use trained models:")
        print(f"   from joblib import load")
        print(f"   model = load('Anomaly-Detection_CODE_DAY_MATCH/models/CODE_DAY_MATCH_model.joblib')")
        
        return 0
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())