Predictive Maintenance System for Industrial Machines

Problem Statement
Predict potential machine failures using time-series sensor data (temperature, torque, wear, etc.) to reduce downtime and optimize maintenance.

Approach
Used LSTM (Long Short-Term Memory) neural networks to detect patterns in past sensor readings and predict future machine failure events.

Dataset
[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)

Model
- Scaled features using MinMaxScaler
- Created sequences of 10-timestep sliding windows
- LSTM model with sigmoid activation for binary classification

How to Run
```bash
pip install -r requirements.txt
python predictive_maintenance_lstm.py
