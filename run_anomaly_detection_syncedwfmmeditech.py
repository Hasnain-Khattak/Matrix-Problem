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

class SyncedWFMMeditechDetector(SingleTargetAnomalyDetector):
    """
    Specialized detector for SyncedWFMMeditech anomalies.
    Focuses on WFM to Meditech system synchronization issues.
    """
    
    def __init__(self):
        super().__init__(
            target_column='SyncedWFMMeditech',
            model_name='SyncedWFMMeditech_Detector'
        )
    
    def add_specialized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to WFM-Meditech synchronization."""
        df = self.add_domain_features(df)  # Base features
        
        # WFM-specific features
        wfm_cols = [col for col in df.columns if 'WFM' in col]
        logger.info(f"WFM columns found: {wfm_cols}")
        
        # System completeness features
        if 'WFM_Hours' in df.columns:
            df['wfm_hours_missing'] = df['WFM_Hours'].isna().astype(int)
            df['wfm_zero_hours'] = (df['WFM_Hours'] == 0).astype(int)
            df['wfm_negative_hours'] = (df['WFM_Hours'] < 0).astype(int)
            df['wfm_extreme_hours'] = (df['WFM_Hours'] > 16).astype(int)
        
        if 'WFM_EarnCode' in df.columns:
            df['wfm_earncode_missing'] = df['WFM_EarnCode'].isna().astype(int)
            
        if 'WFM_CostCentre' in df.columns:
            df['wfm_costcentre_missing'] = df['WFM_CostCentre'].isna().astype(int)
        
        # Data quality patterns that affect synchronization
        df['wfm_data_completeness'] = 0
        if 'WFM_Hours' in df.columns:
            df['wfm_data_completeness'] += (~df['WFM_Hours'].isna()).astype(int)
        if 'WFM_EarnCode' in df.columns:
            df['wfm_data_completeness'] += (~df['WFM_EarnCode'].isna()).astype(int)
        if 'WFM_CostCentre' in df.columns:
            df['wfm_data_completeness'] += (~df['WFM_CostCentre'].isna()).astype(int)
        
        # ESP vs WFM system consistency (affecting sync)
        if 'ESP_Hours' in df.columns and 'WFM_Hours' in df.columns:
            # Patterns that suggest sync issues
            df['esp_present_wfm_missing'] = (df['ESP_Hours'].notna() & df['WFM_Hours'].isna()).astype(int)
            df['esp_missing_wfm_present'] = (df['ESP_Hours'].isna() & df['WFM_Hours'].notna()).astype(int)
            df['both_systems_missing'] = (df['ESP_Hours'].isna() & df['WFM_Hours'].isna()).astype(int)
            
            # Sync quality indicators
            df['systems_hour_agreement'] = (df['hours_absdiff'] < 0.1).astype(int)
            df['systems_major_disagreement'] = (df['hours_absdiff'] > 2.0).astype(int)
        
        # Employee-level sync patterns
        if 'Number' in df.columns:
            # Calculate employee-level sync statistics
            emp_sync_stats = df.groupby('Number').agg({
                'wfm_data_completeness': ['mean', 'min'],
                'hours_absdiff': ['mean', 'max', 'std']
            }).fillna(0)
            
            # Flatten column names
            emp_sync_stats.columns = ['_'.join(col) for col in emp_sync_stats.columns]
            emp_sync_stats = emp_sync_stats.add_prefix('emp_sync_')
            
            # Merge back to main dataset
            df = df.merge(emp_sync_stats, left_on='Number', right_index=True, how='left')
            
            # Employee sync reliability features
            df['emp_unreliable_sync'] = (df['emp_sync_wfm_data_completeness_mean'] < 0.8).astype(int)
            df['emp_high_variance_sync'] = (df['emp_sync_hours_absdiff_std'] > 1.0).astype(int)
        
        # Temporal sync patterns
        if 'WorkedDay' in df.columns:
            df['WorkedDay'] = pd.to_datetime(df['WorkedDay'], errors='coerce')
            
            # Day patterns that might affect sync
            df['is_system_maintenance_day'] = ((df['WorkedDay'].dt.dayofweek == 6) |  # Sunday
                                             (df['WorkedDay'].dt.day == 1)).astype(int)  # First of month
            
            # Batch processing patterns
            df['is_batch_day'] = (df['WorkedDay'].dt.dayofweek.isin([0, 6])).astype(int)  # Monday/Sunday
            
            # End of period sync challenges
            df['is_period_end'] = ((df['WorkedDay'].dt.day >= 28) | 
                                 (df['WorkedDay'].dt.day <= 2)).astype(int)
        
        # Job category sync patterns
        if 'O_Desc' in df.columns:
            # Some job types might have more sync issues
            df['job_desc_length'] = df['O_Desc'].astype(str).str.len()
            df['job_has_special_chars'] = df['O_Desc'].astype(str).str.contains('[^a-zA-Z0-9 ]', na=False).astype(int)
            
            # Healthcare-specific patterns
            df['is_nurse_role'] = df['O_Desc'].astype(str).str.contains('nurse', case=False, na=False).astype(int)
            df['is_tech_role'] = df['O_Desc'].astype(str).str.contains('tech', case=False, na=False).astype(int)
        
        logger.info("WFM-Meditech sync-specific features added")
        return df
    
    def add_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Override to use specialized features."""
        return self.add_specialized_features(df)


def main():
    """Main function for SyncedWFMMeditech detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SyncedWFMMeditech Anomaly Detection")
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', default='SyncedWFMMeditech_results', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        # Load data
        logger.info(f"Loading data for SyncedWFMMeditech analysis: {args.input}")
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df):,} records")
        
        # Check for target column
        target_col = 'SyncedWFMMeditech'
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in data")
            return 1
        
        # Show target distribution
        target_dist = df[target_col].value_counts()
        logger.info(f"Target distribution: {target_dist.to_dict()}")
        
        # Initialize specialized detector
        detector = SyncedWFMMeditechDetector()
        
        # Train model
        logger.info("Training SyncedWFMMeditech model...")
        training_results = detector.train_models(df)
        
        # Make predictions
        logger.info("Making predictions...")
        predictions_df = detector.predict(df)
        
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True)
        
        # Save results
        pred_file = output_path / "SyncedWFMMeditech_predictions.csv"
        model_file = output_path / "SyncedWFMMeditech_model.joblib"
        results_file = output_path / "SyncedWFMMeditech_training_results.json"
        
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
SyncedWFMMeditech Anomaly Detection Results
===========================================
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

Key Focus: WFM to Meditech system synchronization
- WFM data completeness analysis
- System-to-system consistency checking
- Employee-level sync reliability patterns
- Temporal sync issue identification
- Healthcare-specific sync challenges
"""
        
        with open(output_path / "SyncedWFMMeditech_summary.txt", 'w') as f:
            f.write(summary_report)
        
        # Print results
        print("\n" + "="*60)
        print("SYNCEDWFMMEDITECH ANOMALY DETECTION RESULTS")
        print("="*60)
        print(f"Total Records: {len(df):,}")
        print(f"Anomalies Detected: {anomaly_count:,} ({anomaly_rate:.2%})")
        print(f"Best Model: {detector.best_model_name}")
        print(f"F1 Score: {training_results[detector.best_model_name]['f1_score']:.3f}")
        print(f"Results saved to: {output_path}/")
        
        return 0
        
    except Exception as e:
        logger.error(f"SyncedWFMMeditech analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())