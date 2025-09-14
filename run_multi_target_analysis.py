#!/usr/bin/env python3
"""
Multi-Target Analysis Runner for your new dataset
Works with: ESPorMTMandWFMHoursEarnCodeDayMatch, ESPorMTMandWFMHoursEarnCodePPMatch, SyncedWFMMeditech
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import argparse
import sys

# Import the MultiTargetAnomalyDetector class
# (Make sure multi_target_anomaly_detector.py is in the same directory)
sys.path.append('.')
from multi_target_anomaly_detector import MultiTargetAnomalyDetector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run multi-target anomaly detection on your new dataset."""
    
    parser = argparse.ArgumentParser(description="Multi-Target Payroll Anomaly Detection")
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', default='multi_target_results', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        # Load data
        logger.info(f"Loading data from {args.input}")
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns")
        
        # Print column info for verification
        logger.info("Available columns:")
        for i, col in enumerate(df.columns):
            logger.info(f"  {i}: {col}")
        
        # Initialize detector with your actual target columns
        detector = MultiTargetAnomalyDetector()
        
        # Override the target columns to match your actual data
        detector.target_columns = ['ESPorMTMandWFMHoursEarnCodeDayMatch', 
                                  'ESPorMTMandWFMHoursEarnCodePPMatch', 
                                  'SyncedWFMMeditech']
        
        # Check if target columns exist
        missing_targets = [col for col in detector.target_columns if col not in df.columns]
        if missing_targets:
            logger.error(f"Missing target columns: {missing_targets}")
            return 1
        
        logger.info(f"Target columns found: {detector.target_columns}")
        
        # Show target distribution
        for target in detector.target_columns:
            target_dist = df[target].value_counts()
            logger.info(f"{target} distribution: {target_dist.to_dict()}")
        
        # Train models
        logger.info("Training multi-target models...")
        training_results = detector.train_models(df)
        
        # Print training results
        logger.info("Training Results:")
        for model_name, metrics in training_results.items():
            logger.info(f"{model_name}:")
            logger.info(f"  Overall Accuracy: {metrics['overall_accuracy']:.3f}")
            for target, target_metrics in metrics['target_metrics'].items():
                logger.info(f"  {target}: F1={target_metrics['f1_score']:.3f}, Acc={target_metrics['accuracy']:.3f}")
        
        # Make predictions
        logger.info("Making predictions...")
        predictions_df = detector.predict_multi_target(df)
        
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True)
        
        # Save predictions
        predictions_file = output_path / "multi_target_predictions.csv"
        predictions_df.to_csv(predictions_file, index=False)
        logger.info(f"Predictions saved to {predictions_file}")
        
        # Save model
        model_file = output_path / "multi_target_model.joblib"
        detector.save_model(str(model_file))
        logger.info(f"Model saved to {model_file}")
        
        # Generate analysis summary
        summary = analyze_results(predictions_df, detector.target_columns)
        
        # Save summary
        summary_file = output_path / "analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(generate_summary_report(summary, len(df)))
        logger.info(f"Summary report saved to {summary_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("MULTI-TARGET ANOMALY DETECTION RESULTS")
        print("="*60)
        print(f"Total Records: {len(df):,}")
        print(f"Records with Any Anomaly: {summary['any_anomaly_count']:,} ({summary['any_anomaly_rate']:.1%})")
        print("\nPer-Target Results:")
        for target in detector.target_columns:
            target_short = target.replace('ESPorMTMandWFMHoursEarnCode', '').replace('Match', '')
            if target_short == '': target_short = 'Day'
            count = summary['target_results'][target]['anomaly_count']
            rate = summary['target_results'][target]['anomaly_rate']
            print(f"  {target_short}: {count:,} anomalies ({rate:.1%})")
        
        print(f"\nResults saved to: {args.output}/")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

def analyze_results(predictions_df: pd.DataFrame, target_columns: list) -> dict:
    """Analyze prediction results and generate summary statistics."""
    
    summary = {
        'total_records': len(predictions_df),
        'target_results': {},
        'any_anomaly_count': 0,
        'any_anomaly_rate': 0.0
    }
    
    # Analyze each target
    for target in target_columns:
        pred_col = f'{target}_Prediction'
        conf_col = f'{target}_Confidence'
        
        if pred_col in predictions_df.columns:
            anomaly_count = (predictions_df[pred_col] == 'Anomaly').sum()
            anomaly_rate = anomaly_count / len(predictions_df)
            
            target_result = {
                'anomaly_count': int(anomaly_count),
                'anomaly_rate': float(anomaly_rate),
                'normal_count': int((predictions_df[pred_col] == 'Normal').sum())
            }
            
            if conf_col in predictions_df.columns:
                target_result['avg_confidence'] = float(predictions_df[conf_col].mean())
                target_result['high_conf_anomalies'] = int(
                    ((predictions_df[pred_col] == 'Anomaly') & (predictions_df[conf_col] > 0.8)).sum()
                )
            
            summary['target_results'][target] = target_result
    
    # Analyze combined results
    if 'Any_Anomaly' in predictions_df.columns:
        any_anomaly_count = (predictions_df['Any_Anomaly'] == 'Yes').sum()
        summary['any_anomaly_count'] = int(any_anomaly_count)
        summary['any_anomaly_rate'] = float(any_anomaly_count / len(predictions_df))
    
    return summary

def generate_summary_report(summary: dict, total_records: int) -> str:
    """Generate a human-readable summary report."""
    
    report_lines = [
        "=" * 80,
        "MULTI-TARGET PAYROLL ANOMALY DETECTION REPORT",
        "=" * 80,
        f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Records Analyzed: {total_records:,}",
        "",
        "TARGET ANALYSIS RESULTS:",
        "-" * 50,
    ]
    
    for target, results in summary['target_results'].items():
        # Shorten target name for readability
        target_name = target.replace('ESPorMTMandWFMHoursEarnCode', '').replace('Match', '')
        if target_name == '':
            target_name = 'Day Match'
        elif target_name == 'PP':
            target_name = 'Pay Period Match'
        elif target_name == 'SyncedWFMMeditech':
            target_name = 'WFM Meditech Sync'
        
        report_lines.extend([
            f"{target_name}:",
            f"  Anomalies Detected: {results['anomaly_count']:,} ({results['anomaly_rate']:.2%})",
            f"  Normal Records: {results['normal_count']:,}",
        ])
        
        if 'avg_confidence' in results:
            report_lines.append(f"  Average Confidence: {results['avg_confidence']:.3f}")
        
        if 'high_conf_anomalies' in results:
            report_lines.append(f"  High Confidence Anomalies: {results['high_conf_anomalies']:,}")
        
        report_lines.append("")
    
    report_lines.extend([
        "COMBINED ANOMALY ANALYSIS:",
        "-" * 50,
        f"Records with ANY Target Anomaly: {summary['any_anomaly_count']:,} ({summary['any_anomaly_rate']:.2%})",
        f"Records with ALL Targets Normal: {total_records - summary['any_anomaly_count']:,}",
        "",
        "KEY INSIGHTS:",
        "-" * 50,
    ])
    
    # Generate insights
    highest_anomaly_target = max(summary['target_results'].items(), 
                                key=lambda x: x[1]['anomaly_rate'])
    lowest_anomaly_target = min(summary['target_results'].items(), 
                               key=lambda x: x[1]['anomaly_rate'])
    
    report_lines.extend([
        f"• Highest anomaly rate: {highest_anomaly_target[0]} ({highest_anomaly_target[1]['anomaly_rate']:.2%})",
        f"• Lowest anomaly rate: {lowest_anomaly_target[0]} ({lowest_anomaly_target[1]['anomaly_rate']:.2%})",
    ])
    
    if summary['any_anomaly_rate'] > 0.1:
        report_lines.append(f"• High overall anomaly rate ({summary['any_anomaly_rate']:.1%}) - investigate data quality")
    elif summary['any_anomaly_rate'] < 0.01:
        report_lines.append(f"• Very low anomaly rate ({summary['any_anomaly_rate']:.1%}) - systems operating well")
    
    report_lines.extend([
        "",
        "RECOMMENDATIONS:",
        "-" * 50,
        "• Focus investigation on high-confidence anomaly predictions",
        "• Review records where multiple targets show anomalies",
        "• Consider retraining model if data patterns change significantly",
        "",
        "=" * 80,
        "End of Report"
    ])
    
    return "\n".join(report_lines)

if __name__ == "__main__":
    sys.exit(main())