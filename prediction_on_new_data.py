import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import json
import argparse
import sys
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelPredictor:
    """Load trained models and make predictions on new data."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_data = None
        self.model = None
        
    def load_model(self):
        """Load the trained model."""
        try:
            self.model_data = joblib.load(self.model_path)
            self.model = self.model_data['model']
            self.target_column = self.model_data['target_column']
            self.model_name = self.model_data['model_name']
            logger.info(f"Loaded model: {self.model_name} for {self.target_column}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            return False
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the same features as used during training."""
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
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Add features (same as training)
        df_processed = self.add_features(df)
        
        # Make predictions
        predictions = self.model.predict(df_processed)
        probabilities = self.model.predict_proba(df_processed)[:, 1]
        
        # Create results dataframe
        results_df = df.copy()
        results_df[f'{self.target_column}_Prediction'] = ['Anomaly' if p == 1 else 'Match' for p in predictions]
        results_df[f'{self.target_column}_Confidence'] = probabilities
        
        return results_df
    
    def evaluate_predictions(self, df: pd.DataFrame, predictions_df: pd.DataFrame) -> dict:
        """Evaluate predictions if true labels are available."""
        
        if self.target_column not in df.columns:
            logger.warning(f"Target column {self.target_column} not found. Skipping evaluation.")
            return None
        
        # Map targets to binary
        def coerce_target(y: pd.Series) -> pd.Series:
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
            return y.map(map_func)
        
        y_true = coerce_target(df[self.target_column])
        y_pred = predictions_df[f'{self.target_column}_Prediction'].map({'Anomaly': 1, 'Match': 0})
        
        # Remove missing values
        valid_mask = ~y_true.isna()
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        
        if len(y_true_clean) == 0:
            return None
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_clean, y_pred_clean)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_clean, y_pred_clean, average='binary', zero_division=0
        )
        
        cm = confusion_matrix(y_true_clean, y_pred_clean)
        
        evaluation = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'true_positives': int(cm[1, 1]),
            'false_positives': int(cm[0, 1]),
            'true_negatives': int(cm[0, 0]),
            'false_negatives': int(cm[1, 0]),
            'total_samples': len(y_true_clean)
        }
        
        return evaluation

def predict_single_model(model_folder: str, new_data_file: str, output_file: str = None):
    """Predict using a single model."""
    
    # Find model file
    model_path = Path(model_folder) / 'models'
    model_files = list(model_path.glob('*.joblib'))
    
    if not model_files:
        logger.error(f"No model files found in {model_path}")
        return None
    
    model_file = model_files[0]  # Use first model found
    
    # Load data
    logger.info(f"Loading new data: {new_data_file}")
    df = pd.read_csv(new_data_file)
    logger.info(f"New data: {len(df):,} records")
    
    # Initialize predictor and load model
    predictor = ModelPredictor(str(model_file))
    if not predictor.load_model():
        return None
    
    # Make predictions
    logger.info("Making predictions...")
    predictions_df = predictor.predict(df)
    
    # Evaluate if target column exists
    evaluation = predictor.evaluate_predictions(df, predictions_df)
    
    # Save predictions
    if output_file is None:
        output_file = f"{model_folder}/outputs/new_data_predictions.csv"
    
    predictions_df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved: {output_file}")
    
    # Count anomalies
    pred_col = f'{predictor.target_column}_Prediction'
    anomaly_count = (predictions_df[pred_col] == 'Anomaly').sum()
    anomaly_rate = anomaly_count / len(predictions_df)
    
    results = {
        'model_folder': model_folder,
        'model_file': str(model_file),
        'target_column': predictor.target_column,
        'input_file': new_data_file,
        'output_file': output_file,
        'total_records': len(df),
        'anomalies_detected': int(anomaly_count),
        'anomaly_rate': float(anomaly_rate),
        'evaluation': evaluation,
        'prediction_date': pd.Timestamp.now().isoformat()
    }
    
    # Save results
    results_file = f"{model_folder}/outputs/new_data_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def predict_all_models(new_data_file: str, output_dir: str = "new_predictions"):
    """Predict using all 3 trained models."""
    
    model_folders = [
        'Anomaly-Detection_CODE_DAY_MATCH',
        'Anomaly-Detection_CODE_PP_MATCH',
        'Anomaly-Detection_SyncedWFMMeditech'
    ]
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    all_results = {}
    
    for folder in model_folders:
        if not Path(folder).exists():
            logger.warning(f"Model folder not found: {folder}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Predicting with {folder}")
        logger.info(f"{'='*60}")
        
        # Predict with this model
        output_file = f"{output_dir}/{folder}_predictions.csv"
        results = predict_single_model(folder, new_data_file, output_file)
        
        if results:
            all_results[folder] = results
            
            # Print summary
            print(f"\n{folder}:")
            print(f"  Anomalies: {results['anomalies_detected']:,} ({results['anomaly_rate']:.2%})")
            
            if results['evaluation']:
                eval_data = results['evaluation']
                print(f"  Accuracy: {eval_data['accuracy']:.3f}")
                print(f"  F1 Score: {eval_data['f1_score']:.3f}")
    
    # Save combined results
    combined_results_file = f"{output_dir}/all_models_results.json"
    with open(combined_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Predict on New Data using Trained Models")
    parser.add_argument('--input', required=True, help='New testing data CSV file')
    parser.add_argument('--model', help='Specific model folder (optional)')
    parser.add_argument('--output-dir', default='new_predictions', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        if not Path(args.input).exists():
            logger.error(f"Input file not found: {args.input}")
            return 1
        
        if args.model:
            # Predict with single model
            logger.info(f"Using single model: {args.model}")
            output_file = f"{args.output_dir}/{args.model}_predictions.csv"
            results = predict_single_model(args.model, args.input, output_file)
            
            if results:
                print(f"\nPrediction Results:")
                print(f"Model: {results['target_column']}")
                print(f"Input: {results['input_file']}")
                print(f"Output: {results['output_file']}")
                print(f"Anomalies: {results['anomalies_detected']:,} ({results['anomaly_rate']:.2%})")
                
                if results['evaluation']:
                    eval_data = results['evaluation']
                    print(f"Accuracy: {eval_data['accuracy']:.3f}")
                    print(f"F1 Score: {eval_data['f1_score']:.3f}")
        else:
            # Predict with all models
            logger.info("Using all trained models")
            all_results = predict_all_models(args.input, args.output_dir)
            
            print(f"\n{'='*60}")
            print("PREDICTION SUMMARY - ALL MODELS")
            print(f"{'='*60}")
            print(f"Input file: {args.input}")
            print(f"Output directory: {args.output_dir}")
            
            for model_name, results in all_results.items():
                model_short = model_name.replace('Anomaly-Detection_', '')
                print(f"\n{model_short}:")
                print(f"  Anomalies: {results['anomalies_detected']:,} ({results['anomaly_rate']:.2%})")
                
                if results['evaluation']:
                    eval_data = results['evaluation']
                    print(f"  Accuracy: {eval_data['accuracy']:.3f}")
                    print(f"  F1 Score: {eval_data['f1_score']:.3f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())