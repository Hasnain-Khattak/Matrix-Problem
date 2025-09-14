
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleMatrixAnalyzer:
    """
    Simplified Matrix analyzer that works with PP14 data format.
    Avoids timestamp parsing issues by using Length column directly.
    """
    
    def __init__(self):
        self.results = {}
    
    def apply_break_logic(self, duration_hours: float) -> float:
        """Apply break deductions based on shift duration."""
        if pd.isna(duration_hours):
            return duration_hours
            
        if duration_hours <= 5:
            return duration_hours  # No break
        elif duration_hours <= 8:
            return duration_hours - 0.5  # 30-minute break
        else:
            return duration_hours - 1.0  # 1-hour break
    
    def check_factor1_is_partial(self, df: pd.DataFrame) -> pd.Series:
        """Factor 1: Is Shift Split into Partials?"""
        is_partial = (df['Partial'] == 1)
        logger.info(f"Factor 1 - Partial shifts: {is_partial.sum():,} out of {len(df):,}")
        return is_partial
    
    def check_factor2_has_off_segment(self, df: pd.DataFrame) -> pd.Series:
        """Factor 2: Has OFF Segment?"""
        has_off = (df['PayCodeType'] == 2)
        logger.info(f"Factor 2 - OFF segments: {has_off.sum():,} out of {len(df):,}")
        return has_off
    
    def check_factor3_duration_equals_paid(self, df: pd.DataFrame) -> pd.Series:
        """Factor 3: Does Shift Duration = Paid Hours?"""
        
        # Use Length column for duration
        duration_hours = df['Length']
        
        # Apply break logic
        duration_with_breaks = duration_hours.apply(self.apply_break_logic)
        
        # Compare with paid hours (using 0.5 hour tolerance)
        paid_hours = df['PaidHours']
        tolerance = 0.5
        duration_equals_paid = (abs(paid_hours - duration_with_breaks) <= tolerance)
        
        logger.info(f"Factor 3 - Duration equals paid: {duration_equals_paid.sum():,} out of {len(df):,}")
        return duration_equals_paid
    
    def check_factor4_working_5hrs_or_less(self, df: pd.DataFrame) -> pd.Series:
        """Factor 4: Working Segment is 5 Hours or Less?"""
        duration = df['Length']
        is_5hrs_or_less = duration <= 5
        
        logger.info(f"Factor 4 - 5hrs or less: {is_5hrs_or_less.sum():,} out of {len(df):,}")
        return is_5hrs_or_less
    
    def check_factor5_off_duration_equals_paid(self, df: pd.DataFrame) -> pd.Series:
        """Factor 5: Off Shift Duration = Paid Hours of OFF?"""
        
        # Check if it's an off shift
        has_off = (df['PayCodeType'] == 2)
        
        # For off shifts, check duration equals paid
        duration_hours = df['Length']
        duration_with_breaks = duration_hours.apply(self.apply_break_logic)
        paid_hours = df['PaidHours']
        duration_equals_paid = (abs(paid_hours - duration_with_breaks) <= 0.5)
        
        # Only applies to off shifts
        off_duration_equals_paid = has_off & duration_equals_paid
        
        logger.info(f"Factor 5 - OFF duration equals paid: {off_duration_equals_paid.sum():,} out of {len(df):,}")
        return off_duration_equals_paid
    
    def predict_system_accuracy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict which system (WFM, ESP, or both) will handle each shift correctly."""
        
        df_pred = df.copy()
        
        # Initialize predictions
        df_pred['wfm_expected_correct'] = True
        df_pred['esp_expected_correct'] = True
        df_pred['predicted_outcome'] = 'Both_Correct'
        
        # Rule 1: Complex partial shifts may cause WFM issues
        complex_partials = (
            df_pred['factor1_is_partial'] & 
            df_pred['factor2_has_off_segment'] & 
            ~df_pred['factor3_duration_equals_paid']
        )
        df_pred.loc[complex_partials, 'wfm_expected_correct'] = False
        df_pred.loc[complex_partials, 'predicted_outcome'] = 'ESP_Only_Correct'
        
        # Rule 2: Simple duration mismatches may affect ESP
        simple_duration_issues = (
            ~df_pred['factor1_is_partial'] & 
            ~df_pred['factor3_duration_equals_paid'] &
            df_pred['factor4_working_5hrs_or_less']
        )
        df_pred.loc[simple_duration_issues, 'esp_expected_correct'] = False
        df_pred.loc[simple_duration_issues, 'predicted_outcome'] = 'WFM_Only_Correct'
        
        # Rule 3: OFF segment issues
        off_issues = (
            df_pred['factor2_has_off_segment'] & 
            ~df_pred['factor5_off_duration_equals_paid']
        )
        df_pred.loc[off_issues, 'wfm_expected_correct'] = False
        df_pred.loc[off_issues, 'esp_expected_correct'] = False
        df_pred.loc[off_issues, 'predicted_outcome'] = 'Both_Incorrect'
        
        # Calculate prediction confidence
        df_pred['prediction_confidence'] = self.calculate_prediction_confidence(df_pred)
        
        return df_pred
    
    def calculate_prediction_confidence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate confidence score for matrix predictions."""
        
        confidence_scores = []
        
        for _, row in df.iterrows():
            # Base confidence on factor alignment
            aligned_factors = 0
            if row['factor3_duration_equals_paid']:
                aligned_factors += 1
            if row['factor5_off_duration_equals_paid'] or not row['factor2_has_off_segment']:
                aligned_factors += 1
            
            base_confidence = aligned_factors / 2
            
            # Adjust based on complexity
            complexity_penalty = 0
            if row['factor1_is_partial']:
                complexity_penalty += 0.1
            if row['factor2_has_off_segment']:
                complexity_penalty += 0.1
            
            final_confidence = max(0.1, base_confidence - complexity_penalty)
            confidence_scores.append(final_confidence)
        
        return pd.Series(confidence_scores, index=df.index)
    
    def determine_analysis_scope(self, df: pd.DataFrame) -> pd.Series:
        """Determine which records should be included in detailed analysis."""
        
        # Include in scope if WFM is expected to have issues or uncertainty exists
        in_scope = (
            ~df['wfm_expected_correct'] |  # WFM expected to fail
            (df['factor1_is_partial'] & df['factor2_has_off_segment']) |  # Complex cases
            ~df['factor3_duration_equals_paid']  # Duration mismatches
        )
        
        logger.info(f"Analysis scope: {in_scope.sum():,} out of {len(df):,} records ({in_scope.mean():.2%})")
        return in_scope
    
    def analyze_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run complete matrix analysis on the data."""
        
        logger.info(f"Starting matrix analysis on {len(df):,} records")
        
        # Calculate all 5 factors
        df['factor1_is_partial'] = self.check_factor1_is_partial(df)
        df['factor2_has_off_segment'] = self.check_factor2_has_off_segment(df)
        df['factor3_duration_equals_paid'] = self.check_factor3_duration_equals_paid(df)
        df['factor4_working_5hrs_or_less'] = self.check_factor4_working_5hrs_or_less(df)
        df['factor5_off_duration_equals_paid'] = self.check_factor5_off_duration_equals_paid(df)
        
        # Apply matrix rules
        df = self.predict_system_accuracy(df)
        
        # Determine analysis scope
        df['in_analysis_scope'] = self.determine_analysis_scope(df)
        
        # Calculate risk and complexity scores
        df['matrix_risk_score'] = self.calculate_risk_score(df)
        df['complexity_score'] = self.calculate_complexity_score(df)
        
        logger.info("Matrix analysis completed")
        return df
    
    def calculate_risk_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate risk score based on matrix factors."""
        
        risk_weights = {
            'factor1_is_partial': 0.2,
            'factor2_has_off_segment': 0.3,
            'factor3_duration_equals_paid': -0.4,  # Negative because matching reduces risk
            'factor4_working_5hrs_or_less': 0.1,
            'factor5_off_duration_equals_paid': -0.2
        }
        
        risk_scores = pd.Series(0.0, index=df.index)
        
        for factor, weight in risk_weights.items():
            risk_scores += df[factor].astype(int) * weight
        
        # Normalize to 0-1 range
        if risk_scores.max() > risk_scores.min():
            risk_scores = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min())
        else:
            risk_scores = pd.Series(0.5, index=df.index)
        
        return risk_scores
    
    def calculate_complexity_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate complexity score based on number of active factors."""
        
        factor_cols = ['factor1_is_partial', 'factor2_has_off_segment', 'factor4_working_5hrs_or_less']
        complexity_scores = df[factor_cols].astype(int).sum(axis=1) / len(factor_cols)
        
        return complexity_scores
    
    def generate_summary(self, df: pd.DataFrame) -> dict:
        """Generate summary statistics for the analysis."""
        
        summary = {
            'total_records': len(df),
            'factor_distributions': {
                'partial_shifts': int(df['factor1_is_partial'].sum()),
                'has_off_segments': int(df['factor2_has_off_segment'].sum()),
                'duration_matches_paid': int(df['factor3_duration_equals_paid'].sum()),
                'working_5hrs_or_less': int(df['factor4_working_5hrs_or_less'].sum()),
                'off_duration_matches_paid': int(df['factor5_off_duration_equals_paid'].sum())
            },
            'prediction_distribution': df['predicted_outcome'].value_counts().to_dict(),
            'analysis_scope': {
                'in_scope_count': int(df['in_analysis_scope'].sum()),
                'in_scope_percentage': float(df['in_analysis_scope'].mean() * 100),
                'expected_wfm_issues': int((~df['wfm_expected_correct']).sum()),
                'expected_esp_issues': int((~df['esp_expected_correct']).sum())
            },
            'confidence_stats': {
                'mean_confidence': float(df['prediction_confidence'].mean()),
                'high_confidence_count': int((df['prediction_confidence'] > 0.8).sum()),
                'low_confidence_count': int((df['prediction_confidence'] < 0.3).sum())
            },
            'risk_analysis': {
                'high_risk_count': int((df['matrix_risk_score'] > 0.7).sum()),
                'medium_risk_count': int(((df['matrix_risk_score'] > 0.3) & (df['matrix_risk_score'] <= 0.7)).sum()),
                'low_risk_count': int((df['matrix_risk_score'] <= 0.3).sum())
            }
        }
        
        return summary
    
    def export_results(self, df: pd.DataFrame, output_dir: str = "results"):
        """Export analysis results to files."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate summary
        summary = self.generate_summary(df)
        
        # Export detailed results CSV
        df.to_csv(output_path / "matrix_analysis_results.csv", index=False)
        logger.info(f"Detailed results exported to {output_path / 'matrix_analysis_results.csv'}")
        
        # Export summary JSON
        with open(output_path / "matrix_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary exported to {output_path / 'matrix_summary.json'}")
        
        # Export filtered data for ML analysis
        filtered_df = df[df['in_analysis_scope']].copy()
        filtered_df.to_csv(output_path / "filtered_for_ml_analysis.csv", index=False)
        logger.info(f"Filtered data exported to {output_path / 'filtered_for_ml_analysis.csv'}")
        
        # Generate and export text report
        report = self.generate_text_report(summary)
        with open(output_path / "matrix_analysis_report.txt", 'w') as f:
            f.write(report)
        logger.info(f"Text report exported to {output_path / 'matrix_analysis_report.txt'}")
        
        return summary
    
    def generate_text_report(self, summary: dict) -> str:
        """Generate human-readable text report."""
        
        report_lines = [
            "=" * 80,
            "MATRIX PAYROLL VALIDATION ANALYSIS REPORT",
            "=" * 80,
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Records Processed: {summary['total_records']:,}",
            "",
            "MATRIX FACTOR ANALYSIS:",
            "-" * 40,
            f"Partial Shifts: {summary['factor_distributions']['partial_shifts']:,} ({summary['factor_distributions']['partial_shifts']/summary['total_records']:.1%})",
            f"OFF Segments: {summary['factor_distributions']['has_off_segments']:,} ({summary['factor_distributions']['has_off_segments']/summary['total_records']:.1%})",
            f"Duration Matches Paid Hours: {summary['factor_distributions']['duration_matches_paid']:,} ({summary['factor_distributions']['duration_matches_paid']/summary['total_records']:.1%})",
            f"Working 5hrs or Less: {summary['factor_distributions']['working_5hrs_or_less']:,} ({summary['factor_distributions']['working_5hrs_or_less']/summary['total_records']:.1%})",
            f"OFF Duration Matches Paid: {summary['factor_distributions']['off_duration_matches_paid']:,} ({summary['factor_distributions']['off_duration_matches_paid']/summary['total_records']:.1%})",
            "",
            "SYSTEM ACCURACY PREDICTIONS:",
            "-" * 40,
        ]
        
        for outcome, count in summary['prediction_distribution'].items():
            report_lines.append(f"{outcome}: {count:,} ({count/summary['total_records']:.1%})")
        
        report_lines.extend([
            "",
            "ANALYSIS SCOPE FOR ML:",
            "-" * 40,
            f"Records Requiring Detailed Analysis: {summary['analysis_scope']['in_scope_count']:,} ({summary['analysis_scope']['in_scope_percentage']:.1f}%)",
            f"Expected WFM Issues: {summary['analysis_scope']['expected_wfm_issues']:,}",
            f"Expected ESP Issues: {summary['analysis_scope']['expected_esp_issues']:,}",
            "",
            "CONFIDENCE ANALYSIS:",
            "-" * 40,
            f"Average Confidence: {summary['confidence_stats']['mean_confidence']:.3f}",
            f"High Confidence Predictions (>0.8): {summary['confidence_stats']['high_confidence_count']:,}",
            f"Low Confidence Predictions (<0.3): {summary['confidence_stats']['low_confidence_count']:,}",
            "",
            "RISK ANALYSIS:",
            "-" * 40,
            f"High Risk Records (>0.7): {summary['risk_analysis']['high_risk_count']:,}",
            f"Medium Risk Records (0.3-0.7): {summary['risk_analysis']['medium_risk_count']:,}",
            f"Low Risk Records (≤0.3): {summary['risk_analysis']['low_risk_count']:,}",
            "",
            "KEY INSIGHTS:",
            "-" * 40,
        ])
        
        # Generate insights
        scope_pct = summary['analysis_scope']['in_scope_percentage']
        if scope_pct > 40:
            report_lines.append(f"• High proportion ({scope_pct:.1f}%) of shifts need detailed analysis")
        elif scope_pct < 10:
            report_lines.append(f"• Low proportion ({scope_pct:.1f}%) of problematic shifts - systems performing well")
        
        partial_pct = summary['factor_distributions']['partial_shifts'] / summary['total_records'] * 100
        if partial_pct > 20:
            report_lines.append(f"• High partial shift rate ({partial_pct:.1f}%) - review adjacency logic")
        
        off_pct = summary['factor_distributions']['has_off_segments'] / summary['total_records'] * 100
        if off_pct > 25:
            report_lines.append(f"• High OFF segment rate ({off_pct:.1f}%) - focus on OFF processing rules")
        
        report_lines.extend([
            "",
            "=" * 80,
            "End of Report"
        ])
        
        return "\n".join(report_lines)


def main():
    """Main function to run the simple matrix analyzer."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Matrix Analyzer for PP14 Data")
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', default='results', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        # Load data
        logger.info(f"Loading data from {args.input}")
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns")
        
        # Initialize analyzer
        analyzer = SimpleMatrixAnalyzer()
        
        # Run analysis
        df_analyzed = analyzer.analyze_data(df)
        
        # Export results
        summary = analyzer.export_results(df_analyzed, args.output)
        
        # Print summary
        print("\n" + "="*60)
        print("MATRIX ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total Records: {summary['total_records']:,}")
        print(f"Analysis Scope: {summary['analysis_scope']['in_scope_count']:,} ({summary['analysis_scope']['in_scope_percentage']:.1f}%)")
        print(f"Expected WFM Issues: {summary['analysis_scope']['expected_wfm_issues']:,}")
        print(f"Expected ESP Issues: {summary['analysis_scope']['expected_esp_issues']:,}")
        print(f"\nResults exported to: {args.output}/")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())