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

class CodePPMatchDetector(SingleTargetAnomalyDetector):
    """
    Specialized detector for ESPorMTMandWFMHoursEarnCodePPMatch anomalies.
    Focuses on pay period-level aggregation and earn code consistency.
    """
    
    def __init__(self):
        super().__init__(
            target_column='ESPorMTMandWFMHoursEarnCodePPMatch',
            model_name='CODE_PP_MATCH_Detector'
        )
    
    def add_specialized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to pay period-level matching."""
        df = self.add_domain_features(df)  # Base features
        
        # Pay period specific features
        if 'WorkedDay' in df.columns:
            df['WorkedDay'] = pd.to_datetime(df['WorkedDay'], errors='coerce')
            
            # Pay period patterns (assuming bi-weekly pay periods)
            df['week_of_year'] = df['WorkedDay'].dt.isocalendar().week
            df['pay_period'] = ((df['week_of_year'] - 1) // 2) + 1
            df['week_in_pp'] = df['week_of_year'] % 2  # Week 1 or 2 of pay period
            
            # Month-end/beginning effects on pay periods
            df['is_month_boundary'] = ((df['WorkedDay'].dt.day <= 3) | 
                                     (df['WorkedDay'].dt.day >= 28)).astype(int)
        
        # Employee-level aggregation features (proxy for PP-level patterns)
        if 'Number' in df.columns:
            # Calculate employee-level statistics
            emp_stats = df.groupby('Number').agg({
                'ESP_Hours': ['sum', 'mean', 'count'],
                'WFM_Hours': ['sum', 'mean', 'count'],
                'hours_absdiff': ['sum', 'mean', 'max']
            }).fillna(0)
            
            # Flatten column names
            emp_stats.columns = ['_'.join(col) for col in emp_stats.columns]
            emp_stats = emp_stats.add_prefix('emp_')
            
            # Merge back to main dataset
            df = df.merge(emp_stats, left_on='Number', right_index=True, how='left')
            
            # Employee-level consistency features
            df['emp_total_hour_variance'] = np.where(
                df['emp_ESP_Hours_count'] > 0,
                df['emp_hours_absdiff_sum'] / df['emp_ESP_Hours_count'],
                0
            )
            
            # High vs low activity employees
            df['emp_high_activity'] = (df['emp_ESP_Hours_count'] > df['emp_ESP_Hours_count'].median()).astype(int)
        
        # Pay period aggregation effects
        if 'ESP_Hours' in df.columns and 'WFM_Hours' in df.columns:
            # Cumulative effects that might show up in PP matching
            df['cumulative_hour_bias'] = df.groupby('Number')['hours_diff'].cumsum()
            df['running_avg_hour_diff'] = df.groupby('Number')['hours_diff'].expanding().mean().reset_index(level=0, drop=True)
            
            # PP-level thresholds (larger discrepancies more likely in PP aggregation)
            df['major_hour_discrepancy'] = (df['hours_absdiff'] > 4.0).astype(int)
            df['extreme_hour_discrepancy'] = (df['hours_absdiff'] > 8.0).astype(int)
        
        # Earn code patterns in PP context
        if 'ESP_EarnCode' in df.columns and 'WFM_EarnCode' in df.columns:
            # Employee earn code consistency
            emp_code_consistency = df.groupby('Number')['earn_match'].mean()
            df = df.merge(emp_code_consistency.rename('emp_earn_consistency'), 
                         left_on='Number', right_index=True, how='left')
            
            # Variable vs consistent earn code usage
            df['emp_low_earn_consistency'] = (df['emp_earn_consistency'] < 0.8).astype(int)
        
        logger.info("Pay period-specific features added")
        return df
    
    def add_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Override to use specialized features."""
        return self.add_specialized_features(df)


def main():
    """Main function for CODE_PP_MATCH detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CODE_PP_MATCH Anomaly Detection")
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', default='CODE_PP_MATCH_results', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        # Load data
        logger.info(f"Loading data for CODE_PP_MATCH analysis: {args.input}")
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df):,} records")
        
        # Check for target column
        target_col = 'ESPorMTMandWFMHoursEarnCodePPMatch'
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in data")
            return 1
        
        # Show target distribution
        target_dist = df[target_col].value_counts()
        logger.info(f"Target distribution: {target_dist.to_dict()}")
        
        # Initialize specialized detector
        detector = CodePPMatchDetector()
        
        # Train model
        logger.info("Training CODE_PP_MATCH model...")
        training_results = detector.train_models(df)
        
        # Make predictions
        logger.info("Making predictions...")
        predictions_df = detector.predict(df)
        
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True)
        
        # Save results
        pred_file = output_path / "CODE_PP_MATCH_predictions.csv"
        model_file = output_path / "CODE_PP_MATCH_model.joblib"
        results_file = output_path / "CODE_PP_MATCH_training_results.json"
        
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
CODE_PP_MATCH Anomaly Detection Results
=======================================
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

Key Focus: Pay period-level earn code aggregation patterns
- Employee-level consistency analysis
- Pay period boundary effects
- Cumulative hour discrepancy tracking
- Aggregation-specific threshold detection
"""
        
        with open(output_path / "CODE_PP_MATCH_summary.txt", 'w') as f:
            f.write(summary_report)
        
        # Print results
        print("\n" + "="*60)
        print("CODE_PP_MATCH ANOMALY DETECTION RESULTS")
        print("="*60)
        print(f"Total Records: {len(df):,}")
        print(f"Anomalies Detected: {anomaly_count:,} ({anomaly_rate:.2%})")
        print(f"Best Model: {detector.best_model_name}")
        print(f"F1 Score: {training_results[detector.best_model_name]['f1_score']:.3f}")
        print(f"Results saved to: {output_path}/")
        
        return 0
        
    except Exception as e:
        logger.error(f"CODE_PP_MATCH analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())