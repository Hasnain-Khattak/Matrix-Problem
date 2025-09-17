# Three Separate Anomaly Detection Models

## Overview

This system creates **three independent anomaly detection models** for payroll data validation. Each model specializes in detecting anomalies for a specific target column, providing focused analysis and predictions.

## Quick Start

### Single Command Setup
```bash
python complete_three_models_setup.py --input anomaly_dataset.csv
```

This command will:
- Create 3 separate model folders
- Split your data into train/test sets (80/20)
- Train specialized models for each target
- Save trained models and evaluation results

### Test on New Data
```bash
python prediction_on_new_data.py --input your_new_test_data.csv
```

## System Architecture

### Three Independent Models

#### 1. CODE_DAY_MATCH Model
- **Target**: `ESPorMTMandWFMHoursEarnCodeDayMatch`
- **Focus**: Day-level earn code matching anomalies
- **Folder**: `Anomaly-Detection_CODE_DAY_MATCH/`
- **Specialization**: Daily patterns, same-day discrepancies

#### 2. CODE_PP_MATCH Model  
- **Target**: `ESPorMTMandWFMHoursEarnCodePPMatch`
- **Focus**: Pay period-level earn code matching anomalies
- **Folder**: `Anomaly-Detection_CODE_PP_MATCH/`
- **Specialization**: Pay period aggregation, employee trends

#### 3. SyncedWFMMeditech Model
- **Target**: `SyncedWFMMeditech`
- **Focus**: WFM-Meditech synchronization anomalies
- **Folder**: `Anomaly-Detection_SyncedWFMMeditech/`
- **Specialization**: System sync issues, data completeness

## Data Requirements

### Input Data Format
Your CSV file must contain these columns:

**Required Columns:**
```
- ESPorMTMandWFMHoursEarnCodeDayMatch (target 1)
- ESPorMTMandWFMHoursEarnCodePPMatch (target 2)
- SyncedWFMMeditech (target 3)
- ESP_Hours (numeric)
- WFM_Hours (numeric)
- ESP_EarnCode (categorical)
- WFM_EarnCode (categorical)
- ESP_CostCentre (categorical)
- WFM_CostCentre (categorical)
- WorkedDay (date)
- Number (employee ID)
- O_Desc (job description)
```

### Target Value Format
Each target column should contain:
- `"Match"` - Normal cases
- `"Not Matched"` - Anomaly cases

## Folder Structure

After running the setup, you'll have:

```
├── Anomaly-Detection_CODE_DAY_MATCH/
│   ├── data/
│   │   ├── train_data.csv
│   │   ├── test_data.csv
│   │   └── data_info.json
│   ├── models/
│   │   └── CODE_DAY_MATCH_model.joblib
│   └── outputs/
│       ├── CODE_DAY_MATCH_results.json
│       ├── CODE_DAY_MATCH_test_predictions.csv
│       └── CODE_DAY_MATCH_summary.txt
│
├── Anomaly-Detection_CODE_PP_MATCH/
│   └── (same structure)
│
├── Anomaly-Detection_SyncedWFMMeditech/
│   └── (same structure)
│
├── complete_three_models_setup.py
└── prediction_on_new_data.py
│
└──anomaly_dataset.csv
│
└──test_data.csv
```

## Model Features

### Common Features (All Models)
- **Hours Analysis**: Differences between ESP_Hours and WFM_Hours
- **Code Matching**: Consistency between ESP and WFM earn codes
- **Cost Center Matching**: ESP vs WFM cost center alignment
- **Temporal Features**: Day of week, month, weekend indicators

### Specialized Features

**CODE_DAY_MATCH**:
- Day-of-week pattern analysis
- Daily consistency checks
- Same-day hour discrepancy detection

**CODE_PP_MATCH**:
- Employee-level consistency tracking
- Pay period boundary effects
- Cumulative discrepancy analysis

**SyncedWFMMeditech**:
- WFM data completeness analysis
- System-to-system consistency
- Healthcare-specific sync patterns

## Machine Learning Details

### Model Architecture
- **Primary**: Random Forest Classifier
- **Secondary**: Logistic Regression
- **Selection**: Best performing model (by F1-score) is automatically chosen
- **Training**: Stratified train/test split with hyperparameter tuning

### Evaluation Metrics
- **Accuracy**: Overall correct predictions
- **Precision**: True anomalies / Predicted anomalies
- **Recall**: True anomalies / Actual anomalies
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under ROC curve

### Output Format
Each prediction includes:
- **Prediction**: "Anomaly" or "Match"
- **Confidence**: Probability score (0-1)

## Usage Examples

### Training Models
```bash
# Train all 3 models at once
python complete_three_models_setup.py --input anomaly_dataset.csv

# Expected output:
# Processing CODE_DAY_MATCH...
# ✅ CODE_DAY_MATCH COMPLETED
# Processing CODE_PP_MATCH...
# ✅ CODE_PP_MATCH COMPLETED
# Processing SyncedWFMMeditech...
# ✅ SyncedWFMMeditech COMPLETED
```

### Testing on New Data
```bash
# Test all models
python prediction_on_new_data.py --input new_test_data.csv

# Test specific model
python prediction_on_new_data.py --input new_test_data.csv --model Anomaly-Detection_CODE_DAY_MATCH
```

### Loading Trained Models (Python)
```python
import joblib
import pandas as pd

# Load a trained model
model_data = joblib.load('Anomaly-Detection_CODE_DAY_MATCH/models/CODE_DAY_MATCH_model.joblib')
model = model_data['model']

# Load and predict on new data
new_data = pd.read_csv('new_data.csv')
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)[:, 1]
```

## Performance Expectations

### Typical Results
Based on 10K training records:

| Model | Typical F1-Score | Typical Accuracy | Training Time |
|-------|------------------|------------------|---------------|
| CODE_DAY_MATCH | 0.75-0.85 | 0.85-0.95 | 30-60 seconds |
| CODE_PP_MATCH | 0.70-0.80 | 0.80-0.90 | 30-60 seconds |
| SyncedWFMMeditech | 0.65-0.75 | 0.75-0.85 | 30-60 seconds |

### Memory and Storage
- **Training Memory**: ~100-200MB per model
- **Model Size**: ~5-15MB per trained model
- **Prediction Memory**: ~50MB for 10K records

## File Outputs

### Training Outputs
Each model folder contains:

**models/**
- `{MODEL_NAME}_model.joblib` - Trained model for predictions

**outputs/**
- `{MODEL_NAME}_results.json` - Complete training metrics
- `{MODEL_NAME}_test_predictions.csv` - Test set predictions
- `{MODEL_NAME}_summary.txt` - Human-readable report

**data/**
- `train_data.csv` - Training data (80% of input)
- `test_data.csv` - Test data (20% of input)
- `data_info.json` - Data split information

### Prediction Outputs
When testing new data:

**new_predictions/**
- `{MODEL_NAME}_predictions.csv` - Predictions with confidence scores
- `{MODEL_NAME}_results.json` - Prediction summary and metrics
- `all_models_results.json` - Combined results from all models

## Dependencies

### Required Python Packages
```bash
pip install pandas numpy scikit-learn joblib
```

### Python Version
- **Minimum**: Python 3.7+
- **Recommended**: Python 3.8+

## Troubleshooting

### Common Issues

**"Target column not found"**
- Ensure your CSV has the exact column names: `ESPorMTMandWFMHoursEarnCodeDayMatch`, `ESPorMTMandWFMHoursEarnCodePPMatch`, `SyncedWFMMeditech`

**"Too few records for training"**
- Each target needs at least 10 valid records (non-null values)
- Check your data for missing values in target columns

**"Model file not found"**
- Run the training script first: `python complete_three_models_setup.py --input your_data.csv`
- Check that model folders were created successfully

**Low model performance**
- Check data quality (missing values, inconsistent formats)
- Ensure sufficient examples of both "Match" and "Not Matched" cases
- Consider adding more training data

### Data Quality Checks
Before training, verify:
- Target columns have both "Match" and "Not Matched" values
- Numeric columns (ESP_Hours, WFM_Hours) contain reasonable values (0-24 range)
- Date column (WorkedDay) is in a parseable format
- At least 100+ records per target for reliable training

## Production Deployment

### Model Retraining
Models should be retrained when:
- New data patterns emerge
- Performance degrades over time
- Business rules change
- New data sources are added

### Monitoring
Track these metrics in production:
- Anomaly detection rates over time
- Model confidence scores
- Data drift in input features
- False positive/negative rates

### Scalability
For larger datasets:
- Process data in chunks for memory efficiency
- Use sampling for initial model development
- Consider distributed training for very large datasets

## Support and Maintenance

### Model Updates
To update a model with new data:
1. Combine old and new training data
2. Re-run: `python complete_three_models_setup.py --input updated_data.csv`
3. Compare new vs old model performance
4. Deploy better performing model

### Feature Engineering
To add new features, modify the `add_features()` function in the training script. Consider:
- Domain-specific business rules
- Interaction terms between existing features
- Temporal aggregations
- External data sources

---

This system provides a complete solution for multi-target payroll anomaly detection with independent, specialized models for each target type.