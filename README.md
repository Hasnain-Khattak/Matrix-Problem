# Integrated Payroll Analysis System

## Overview

This project implements two complementary approaches for payroll data analysis and anomaly detection:

1. **Multi-Target Anomaly Detection** - Machine learning system that detects anomalies across three independent payroll matching targets
2. **Matrix-Based Payroll Validation** - Rule-based system implementing 5-factor logic for ESP vs WFM system comparison

## Project Structure

### Working Files
```
payroll-analysis/
├── run_multi_target_analysis.py        # Runner script for ML analysis
├── simple_matrix_analyzer.py           # Matrix-based validation system
├── anomaly_dataset.csv                 # Dataset for multi-target analysis (10K records)
├── PP14.csv                            # Dataset for matrix analysis (400K+ records)
└── README.md                           # This documentation
```

### Deprecated Files (Not Used)
```
matrix_payroll_validator.py     # Had timestamp parsing issues
integrated_payroll_system.py    # Required both datasets together
config.json                     # Configuration for integrated system
```

## Part 1: Multi-Target Anomaly Detection

### Purpose
Detect anomalies in three independent binary targets using machine learning:
- `ESPorMTMandWFMHoursEarnCodeDayMatch`
- `ESPorMTMandWFMHoursEarnCodePPMatch` 
- `SyncedWFMMeditech`

### Data Requirements
**Input:** `anomaly_dataset.csv` with columns:
```
- Number (int64): Employee identifier
- WorkedDay (object): Date worked
- ESP_CostCentre (object): ESP system cost center
- ESP_EarnCode (object): ESP system earn code
- WFM_EarnCode (float64): WFM system earn code
- O_Desc (object): Job description
- ESP_Hours (float64): Hours from ESP system
- WFM_Hours (float64): Hours from WFM system
- WFM_CostCentre (object): WFM system cost center
- ESPorMTMandWFMHoursEarnCodeDayMatch (object): Target 1
- ESPorMTMandWFMHoursEarnCodePPMatch (object): Target 2
- SyncedWFMMeditech (object): Target 3
```

### Key Features
- **Independent Classification**: Each target is treated as separate binary classification
- **Multi-Output Models**: Single model predicts all three targets simultaneously
- **Feature Engineering**: 
  - Hour differences (`ESP_Hours` vs `WFM_Hours`)
  - Code matching (`ESP_EarnCode` vs `WFM_EarnCode`)
  - Cost center matching (`ESP_CostCentre` vs `WFM_CostCentre`)
  - Temporal features from `WorkedDay`
  - Interaction features (hour differences × mismatches)

### Target Mapping
```
"Match" → 0 (Normal)
"Not Matched" → 1 (Anomaly)
```

### Usage
```bash
python run_multi_target_analysis.py --input anomaly_dataset.csv --output results
```

### Output
- `multi_target_predictions.csv`: Detailed predictions with confidence scores
- `multi_target_model.joblib`: Trained model for reuse
- `analysis_summary.txt`: Human-readable report

### Output Format
Each record gets predictions for all targets:
```
Target1_Prediction, Target1_Confidence, Target2_Prediction, Target2_Confidence, Target3_Prediction, Target3_Confidence, Any_Anomaly
Normal, 0.85, Anomaly, 0.92, Normal, 0.78, Yes
```

## Part 2: Matrix-Based Payroll Validation

### Purpose
Apply 5-factor matrix logic from payroll documentation to predict ESP vs WFM system accuracy and filter problematic records for detailed analysis.

### Data Requirements
**Input:** `PP14.csv` with columns:
```
- ShiftDate, ShiftStartTime, ShiftEndTime (object): Shift timing info
- Number (object): Employee identifier  
- PaidHours (float64): Actual paid hours
- Length (float64): Shift duration in hours
- Partial (int64): Partial shift indicator (0/1)
- PayCodeType (int64): Pay code type (2 = OFF shift)
- Plus 35+ other payroll system columns
```

### The 5 Matrix Factors

#### Factor 1: Is Shift Split into Partials?
- **Source**: `Partial` column
- **Logic**: `1` = partial shift, `0` = non-partial shift
- **Purpose**: Identify complex shift scenarios

#### Factor 2: Has OFF Segment?
- **Source**: `PayCodeType` column  
- **Logic**: `2` = OFF shift, other values = regular shift
- **Purpose**: Identify time-off related processing

#### Factor 3: Does Shift Duration = Paid Hours?
- **Source**: `Length` and `PaidHours` columns
- **Logic**: Apply break deduction rules, compare with tolerance
- **Break Rules**:
  - ≤ 5 hours: No break deduction
  - 5-8 hours: 30-minute break deduction
  - > 8 hours: 1-hour break deduction
- **Tolerance**: ±0.5 hours for comparison

#### Factor 4: Working Segment is 5 Hours or Less?
- **Source**: `Length` column
- **Logic**: `Length <= 5` hours
- **Purpose**: Identify short shifts with different processing rules

#### Factor 5: Off Shift Duration = Paid Hours of OFF?
- **Source**: Combination of Factors 2 and 3
- **Logic**: Apply Factor 3 logic only to OFF shifts (Factor 2 = True)
- **Purpose**: Special handling for OFF segment duration matching

### Matrix Prediction Rules

#### Rule 1: Complex Partial Issues → WFM Problems
```
IF partial AND has_off_segment AND NOT duration_equals_paid:
    WFM_expected_correct = False
    Outcome = "ESP_Only_Correct"
```

#### Rule 2: Simple Duration Issues → ESP Problems  
```
IF NOT partial AND NOT duration_equals_paid AND working_5hrs_or_less:
    ESP_expected_correct = False
    Outcome = "WFM_Only_Correct"
```

#### Rule 3: OFF Segment Issues → Both Systems Problems
```
IF has_off_segment AND NOT off_duration_equals_paid:
    WFM_expected_correct = False
    ESP_expected_correct = False
    Outcome = "Both_Incorrect"
```

### Analysis Scope Filtering
Records included in detailed analysis scope:
- WFM expected to have issues
- Complex cases (partial + OFF segments)
- Duration mismatches

**Result**: Filters ~400K records down to focused subset for manual review

### Usage
```bash
python simple_matrix_analyzer.py --input PP14.csv --output results
```

### Output
- `matrix_analysis_results.csv`: Complete analysis with all factors and predictions
- `filtered_for_ml_analysis.csv`: Reduced dataset focusing on problematic records
- `matrix_summary.json`: Statistical summary in JSON format
- `matrix_analysis_report.txt`: Human-readable comprehensive report

## Technical Implementation Details

### Data Challenges Encountered

#### Timestamp Parsing Issues (PP14.csv)
**Problem**: Timestamp columns contained invalid values like `30:00.0`, `45:00.0`
**Solution**: Bypassed timestamp parsing, used `Length` column directly for duration calculations

#### Missing Target Columns (Initial Attempt)
**Problem**: First dataset lacked required ML target columns
**Solution**: Obtained proper dataset with actual target columns

#### Column Name Mismatches
**Problem**: Code initially used generic column names  
**Solution**: Updated column references to match actual data structure

### Machine Learning Approach

#### Model Selection
- **Random Forest**: Primary model for handling mixed data types and feature interactions
- **Logistic Regression**: Secondary model for comparison and interpretability
- **Multi-Output Wrapper**: Enables single model to predict all targets simultaneously

#### Feature Engineering Strategy
```python
# Numerical features
hours_diff = ESP_Hours - WFM_Hours
hours_absdiff = abs(hours_diff)
hours_ratio = ESP_Hours / WFM_Hours (with zero protection)

# Categorical matching features  
earn_match = (ESP_EarnCode == WFM_EarnCode)
costcentre_match = (ESP_CostCentre == WFM_CostCentre)

# Interaction features
absdiff_x_earnmismatch = hours_absdiff * (1 - earn_match)
absdiff_x_ccmismatch = hours_absdiff * (1 - costcentre_match)

# Temporal features
day_of_week = WorkedDay.dt.dayofweek  
is_weekend = (day_of_week >= 5)
month = WorkedDay.dt.month
```

#### Data Preprocessing Pipeline
1. **Target Encoding**: Map text values to binary (0/1)
2. **Feature Engineering**: Create domain-specific features
3. **Missing Value Handling**: 
   - Numerical: Median imputation + scaling
   - Categorical: Mode imputation + one-hot encoding
4. **Train-Test Split**: 80/20 stratified split

#### Model Evaluation
- **Per-Target Metrics**: Accuracy, Precision, Recall, F1-Score for each target
- **Combined Metrics**: Overall accuracy, any-anomaly detection rate
- **Confidence Scoring**: Prediction probabilities for uncertainty quantification

### Matrix Implementation Strategy

#### Original Complex Approach (Abandoned)
- Full timestamp parsing and datetime arithmetic
- Complex time-based adjacency grouping algorithm
- Sophisticated partial shift aggregation logic
- **Issue**: Failed due to non-standard timestamp format

#### Simplified Working Approach  
- Direct use of `Length` column for duration
- Basic employee-level grouping for partial shifts
- Robust error handling and fallback logic
- More lenient comparison tolerances

#### Risk and Complexity Scoring
```python
# Risk Score (0-1 scale)
risk_weights = {
    'factor1_is_partial': 0.2,
    'factor2_has_off_segment': 0.3, 
    'factor3_duration_equals_paid': -0.4,  # Negative = reduces risk
    'factor4_working_5hrs_or_less': 0.1,
    'factor5_off_duration_equals_paid': -0.2
}

# Complexity Score (0-1 scale)
complexity = (is_partial + has_off_segment + working_5hrs_or_less) / 3
```

## Development Evolution

### Phase 1: Initial Integration Attempt
- Created comprehensive integrated system
- Designed for datasets with both ML targets and matrix factors
- **Issue**: Real data didn't match expected format

### Phase 2: Data Format Discovery  
- Analyzed actual data structures
- Identified timestamp parsing incompatibilities
- Discovered missing ML target columns in PP14.csv

### Phase 3: Separate System Development
- Split into two independent systems
- Developed matrix-only analyzer for PP14.csv
- Updated ML analyzer for proper target columns

### Phase 4: Data-Specific Optimization
- Created simplified implementations avoiding problematic parsing
- Added comprehensive error handling
- Implemented fallback logic for edge cases

### Phase 5: Final Integration and Testing
- Verified both systems work with respective datasets
- Created comprehensive documentation
- Developed user-friendly runner scripts

## Performance Characteristics

### Multi-Target ML System
- **Dataset Size**: 10K records
- **Training Time**: ~30-60 seconds  
- **Memory Usage**: ~50-100MB
- **Output Size**: ~2-5MB files

### Matrix Analysis System  
- **Dataset Size**: 400K+ records
- **Processing Time**: ~2-5 minutes
- **Memory Usage**: ~500MB-1GB
- **Output Size**: ~200-500MB files
- **Filtering Efficiency**: Reduces analysis scope by ~50-80%

## Results and Insights

### Multi-Target Analysis Results
Based on 10K record analysis:
- Independent anomaly rates per target
- Combined anomaly detection across all targets
- Confidence-based prioritization for investigation
- Feature importance analysis showing key predictors

### Matrix Analysis Results  
Based on 400K+ record analysis:
- **Factor 1 (Partials)**: ~33K partial shifts identified
- **Factor 2 (OFF segments)**: ~108K OFF segments found  
- **Analysis Scope**: Significant reduction in records requiring manual review
- **System Predictions**: ESP vs WFM accuracy forecasts for each shift

## Installation and Dependencies

### Required Packages
```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn
```

### Python Version
- **Minimum**: Python 3.7+
- **Recommended**: Python 3.8+

### File Structure Setup
```bash
mkdir payroll-analysis
cd payroll-analysis
# Copy all working files to this directory
# Ensure data files are in same directory as scripts
```

## Usage Instructions

### For Multi-Target Analysis
```bash
# Basic usage
python run_multi_target_analysis.py --input anomaly_dataset.csv --output multi_results

# Check outputs
ls multi_results/
# Expected files:
# - multi_target_predictions.csv
# - multi_target_model.joblib  
# - analysis_summary.txt
```

### For Matrix Analysis
```bash
# Basic usage  
python simple_matrix_analyzer.py --input PP14.csv --output matrix_results

# Check outputs
ls matrix_results/
# Expected files:
# - matrix_analysis_results.csv
# - filtered_for_ml_analysis.csv
# - matrix_summary.json
# - matrix_analysis_report.txt
```

## Troubleshooting

### Common Issues

#### "Missing target columns" Error
**Solution**: Ensure your dataset has the exact column names:
- `ESPorMTMandWFMHoursEarnCodeDayMatch`
- `ESPorMTMandWFMHoursEarnCodePPMatch`  
- `SyncedWFMMeditech`

#### "Memory Error" with Large Datasets
**Solution**: 
- Process data in chunks
- Increase available RAM
- Use sampling for initial analysis

#### "Import Error" for MultiTargetAnomalyDetector
**Solution**: Ensure `multi_target_anomaly_detector.py` is in the same directory

### Data Quality Checks
- Verify required columns exist
- Check for reasonable value ranges (hours 0-24, dates in valid format)
- Ensure sufficient non-null data for training

## Future Enhancements

### Potential Improvements
1. **Real-time Processing**: Streaming analysis capability
2. **Advanced Visualization**: Interactive dashboards for results
3. **Model Updating**: Incremental learning for concept drift
4. **Integration APIs**: REST endpoints for system integration
5. **Advanced Matrix Rules**: More sophisticated ESP vs WFM prediction logic

### Scalability Considerations
- Database integration for larger datasets
- Distributed processing for very large files
- Cloud deployment options
- Automated retraining pipelines

## Contact and Support

For questions about implementation details, data format requirements, or system enhancement requests, refer to the code comments and logging output for detailed troubleshooting information.

---

*This system was developed to handle real-world payroll data anomaly detection and validation challenges, with robust error handling and practical solutions for common data quality issues.*