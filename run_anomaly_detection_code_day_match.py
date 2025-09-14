import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Import the base detector class
sys.path.append('.')
from single_target_anomaly_detector import SingleTargetAnomalyDetector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeDayMatchDetector(SingleTargetAnomalyDetector):
    """
    Specialized detector for ESPorMTMandWFMHoursEarnCodeDayMatch anomalies.
    Focuses on day-level matching patterns and earn code consistency.
    """
    
    def __init__(self):
        super().__init__(
            target_column='ESPorMTMandWFMHoursEarnCodeDayMatch',
            model_name='CODE_DAY_MATCH_Detector'
        )
    
    def add_specialized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to day-level matching."""
        df = self.add_domain_features(df)  # Base features
        
        # Day-specific features
        if 'WorkedDay' in df.columns:
            df['WorkedDay'] = pd.to_datetime(df['WorkedDay'], errors='coerce')
            
            # Day-level patterns
            df['is_monday'] = (df['WorkedDay'].dt.dayofweek == 0).astype(int)
            df['is_friday'] = (df['WorkedDay'].dt.dayofweek == 4).astype(int)
            df['day_of_month'] = df['WorkedDay'].dt.day
            df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
            df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
        
        # Earn code specific features
        if 'ESP_EarnCode' in df.columns and 'WFM_EarnCode' in df.columns:
            # Check for common earn code patterns that might cause day matching issues
            df['both_codes_null'] = (df['ESP_EarnCode'].isna() & df['WFM_EarnCode'].isna()).astype(int)
            df['one_code_null'] = (df['ESP_EarnCode'].isna() ^ df['WFM_EarnCode'].isna()).astype(int)
        
        # Hours consistency on day level
        if 'ESP_Hours' in df.columns and 'WFM_Hours' in df.columns:
            df['large_hour_discrepancy'] = (df['hours_absdiff'] > 2.0).astype(int)
            df['zero_hours_mismatch'] = ((df['ESP_Hours'] == 0) ^ (df['WFM_Hours'] == 0)).astype(int)
        
        logger.info("Day-specific features added")
        return df
    
    def add_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Override to use specialized features."""
        return self.add_specialized_features(df)


def main():
    """Main function for CODE_DAY_MATCH detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CODE_DAY_MATCH Anomaly Detection")
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', default='CODE_DAY_MATCH_results', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        # Load data
        logger.info(f"Loading data for CODE_DAY_MATCH analysis: {args.input}")
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df):,} records")
        
        # Check for target column
        target_col = 'ESPorMTMandWFMHoursEarnCodeDayMatch'
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in data")
            return 1
        
        # Show target distribution
        target_dist = df[target_col].value_counts()
        logger.info(f"Target distribution: {target_dist.to_dict()}")
        
        # Initialize specialized detector
        detector = CodeDayMatchDetector()
        
        # Train model
        logger.info("Training CODE_DAY_MATCH model...")
        training_results = detector.train_models(df)
        
        # Make predictions
        logger.info("Making predictions...")
        predictions_df = detector.predict(df)
        
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True)
        
        # Save results
        pred_file = output_path / "CODE_DAY_MATCH_predictions.csv"
        model_file = output_path / "CODE_DAY_MATCH_model.joblib"
        results_file = output_path / "CODE_DAY_MATCH_training_results.json"
        
        predictions_df.to_csv(pred_file, index=False)
        detector.save_model(str(model_file))
        
        # Save training results
        import json
        with open(results_file, 'w') as f:
            json.dump(training_results, f, indent=2)
        
        # Generate summary
        anomaly_count = (predictions_df[f'{target_col}_Prediction'] == 'Anomaly').sum()
        anomaly_rate = anomaly_count / len(df)
        
        # Save summary report
        summary_report = f"""
CODE_DAY_MATCH Anomaly Detection Results
========================================
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Records: {len(df):,}
Anomalies Detected: {anomaly_count:,} ({anomaly_rate:.2%})

Best Model: {detector.best_model_name}
Model Performance:
  F1 Score: {training_results[detector.best_model_name]['f1_score']:.3f}
  Accuracy: {training_results[detector.best_model_name]['accuracy']:.3f}
  Precision: {training_results[detector.best_model_name]['precision']:.3f}
  Recall: {training_results[detector.best_model_name]['recall']:.3f}

Target Distribution:
{target_dist.to_string()}

Key Focus: Day-level earn code matching patterns
- Specialized features for day-of-week patterns
- Enhanced earn code consistency checking
- Hours discrepancy analysis for daily records
"""
        
        with open(output_path / "CODE_DAY_MATCH_summary.txt", 'w') as f:
            f.write(summary_report)
        
        # Print results
        print("\n" + "="*60)
        print("CODE_DAY_MATCH ANOMALY DETECTION RESULTS")
        print("="*60)
        print(f"Total Records: {len(df):,}")
        print(f"Anomalies Detected: {anomaly_count:,} ({anomaly_rate:.2%})")
        print(f"Best Model: {detector.best_model_name}")
        print(f"F1 Score: {training_results[detector.best_model_name]['f1_score']:.3f}")
        print(f"Results saved to: {output_path}/")
        
        return 0
        
    except Exception as e:
        logger.error(f"CODE_DAY_MATCH analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())