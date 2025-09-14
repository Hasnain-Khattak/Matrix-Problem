If you want to get the three models separatly then you have to run this

```python
python setup_three_models.py
```
# Model 1: CODE_DAY_MATCH
```python
cd Anomaly-Detection_CODE_DAY_MATCH/
python run_anomaly_detection_code_day_match.py --input ../anomaly_dataset.csv --output results/
```
# Model 2: CODE_PP_MATCH  
```python
cd ../Anomaly-Detection_CODE_PP_MATCH/
python run_anomaly_detection_code_pp_match.py --input ../anomaly_dataset.csv --output results/
```
# Model 3: SyncedWFMMeditech
```python
cd ../Anomaly-Detection_SyncedWFMMeditech/
python run_anomaly_detection_syncedwfmmeditech.py --input ../anomaly_dataset.csv --output results/
```