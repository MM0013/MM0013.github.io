---
layout: post
title: Real-time ROP Optimization in Drilling Using Machine Learning
image: "/posts/rop-optimization-title-img.png"
tags: [Python, Machine Learning, Drilling, Oil & Gas, Optimization]
---

In this project, we developed a real-time Rate of Penetration (ROP) optimization model using supervised machine learning techniques tailored for drilling operations. By analyzing historical drilling data and training regression models, this system can predict ROP under various drilling parameters and suggest optimal inputs in real time to improve drilling efficiency and reduce non-productive time (NPT).

If you're not familiar, ROP is a critical metric in drilling operations that measures how fast the drill bit penetrates the formation. Optimizing ROP without compromising wellbore stability or equipment integrity can lead to significant time and cost savings.


# Table of contents

- [00. Project Overview](#overview-main)
- [01. Dataset Overview](#data-overview)
- [02. Feature Engineering](#feature-overview)
- [03. Model Selection & Training](#model-selection-application)
- [04. Real-Time Optimization Engine](#real-time-opt-engine)
- [05. Visualization](#visualization)

___

# Project Overview  <a name="overview-main"></a>

This project demonstrates how machine learning, when combined with real-time drilling data and domain expertise, can significantly optimize ROP during drilling. It offers actionable insights to drilling engineers and supports faster, more cost-effective well construction.

# 1.Dataset Overview <a name="data-overview"></a>

The dataset used for this project consists of time-series surface drilling parameters captured during real-world drilling operations. Key features include:

- Weight on Bit (WOB)
- Rotary Speed (RPM)
- Standpipe Pressure (SPP)
- Torque
- Flow Rate
- Mud Weight
- Rate of Penetration (ROP)

These features were cleaned, resampled at 1-minute intervals, and filtered to remove static drilling periods.

```python
import pandas as pd

df = pd.read_csv("drilling_data.csv", parse_dates=['Time'])
df = df.set_index('Time').resample('1min').mean().dropna()
df = df[df['ROP'] > 0]  # Remove flat sections or non-drilling intervals

```

# 2.Feature Engineering <a name="feature-overview"></a>
 
We created domain-specific features such as Mechanical Specific Energy (MSE), normalized inputs, and rolling statistics to enhance model performance.


```python
import numpy as np

df['MSE'] = (df['WOB'] * 1000) / (np.pi * (8.5**2) / 4 * df['ROP']) + (120 * df['Torque']) / df['ROP']
df['WOB_per_RPM'] = df['WOB'] / df['RPM']
df['SPP_rolling'] = df['SPP'].rolling(window=5).mean()

```

# 3.Model Selection & Training <a name="model-selection-application"></a>

Several regressors were benchmarked (Random Forest, XGBoost, Gradient Boosting, and Linear Regression). XGBoost consistently outperformed others in both training and cross-validation.

```python

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

features = ['WOB', 'RPM', 'SPP', 'Torque', 'Flow Rate', 'MSE', 'WOB_per_RPM']
X = df[features]
y = df['ROP']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"Test MSE: {mse:.2f}")

```


# 4.Real-Time Optimization Engine <a name="real-time-opt-engine"></a>

After training, the model was integrated into a loop simulating real-time drilling. At each time step, current parameters are fed into the model to predict ROP and adjust drilling setpoints.

```python
def optimize_wob_rpm(current_data, model, bounds):
    best_pred = -np.inf
    best_wob, best_rpm = None, None
    
    for wob in np.arange(bounds['WOB'][0], bounds['WOB'][1], 0.5):
        for rpm in np.arange(bounds['RPM'][0], bounds['RPM'][1], 10):
            features = current_data.copy()
            features['WOB'] = wob
            features['RPM'] = rpm
            features['WOB_per_RPM'] = wob / rpm
            features['MSE'] = (wob * 1000) / (np.pi * (8.5**2) / 4 * features['ROP']) + (120 * features['Torque']) / features['ROP']
            pred = model.predict(pd.DataFrame([features[features.columns]]))[0]
            if pred > best_pred:
                best_pred = pred
                best_wob, best_rpm = wob, rpm
    return best_wob, best_rpm, best_pred
```

# 5.Visualization <a name="visualization"></a>

The optimization results were visualized with Plotly to demonstrate changes in ROP over time and optimal parameter suggestions.

```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index[-100:], y=preds[-100:], name='Predicted ROP'))
fig.add_trace(go.Scatter(x=df.index[-100:], y=y_test[-100:], name='Actual ROP'))
fig.update_layout(title='ROP Prediction vs Actual', xaxis_title='Time', yaxis_title='ROP (m/hr)')
fig.show()

```










