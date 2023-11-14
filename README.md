# Introduction to Data Mining Course Project 3

## ICC Cricket World Cup Predictions
**Team 29 : Data Wizards**
This project is based on the dataset for ICC Cricket World Cup 2023, which contains data including data for each delivery in each match in the WC2023, each match and the points table for WC 2023.

## Table of Contents

1. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
2. [Data Preprocessing](#data-preprocessing)
   - [Step 1: Data Loading](#step-1-data-loading)
   - [Step 2: Data Cleaning and Analysis](#step-2-Data-cleaning-and-analysis)
   - [step 3 : Data Transformation and Label Encoding](#Data-Transformation-and-Label-Encoding)
3. [Model Training and Evaluation](#model-training-and-evaluation)
   - [Classical Models](#classical-models)
   - [ODi Match Winner Prediction](#odi-match-winner)
   - [Team Composition](#team-composition-prediction)
   - [Sixes in Tournament](#sixes-in-the-tournament)

## Getting Started

### Prerequisites

List the software and libraries required to run the project.

- Python
- Jupyter Notebook
- Libraries (e.g., pandas, numpy, scikit-learn)

### Installation

Provide instructions on how to install or set up the necessary software and libraries.

```bash
pip install jupyter
pip install pandas
pip install seaborn
pip install scikit-learn
pip install tensorflow

```

In this section, we import the necessary Python libraries and modules for data analysis and machine learning. Each library has a specific purpose.

This setup prepares the environment for working with data and machine learning, importing libraries and modules commonly used for data analysis and model building. The 'warnings' filter is set to 'ignore' to maintain a cleaner output when running the code.

```python
import numpy as np
import pandas as pd
import os
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from keras.losses import mean_squared_error
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow import keras
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings("ignore")
```

## Data Preprocessing

Explain the data preprocessing steps performed in your project. Include code snippets with explanations.

### Step 1: Data Loading

```python
# Loading the Dataset
ball=pd.read_csv("/kaggle/input/icc-mens-world-cup-2023/deliveries.csv")
```

### Step 2: Data Cleaning and Analysis

In this section, we perform exploratory data analysis (EDA) on the hospital dataset `ball`. We are using various operations and visualizations to gain insights into the data and performing data cleaning and handle missing values in the `ball` dataset.

#### Checking for Missing and Unique Values

We start by checking the number of missing and unique values for each column in the original `ball` DataFrame.

```python
ball[ball['penalty'].notnull()]
ball['season'].unique()
ball['start_date'].unique()
ball['wides'].unique()
ball['wicket_type'].unique()
```

#### Filling null values in required Field

Here, we fill the wides, noballs, etc. "extras" as 0 in the DataFrame.

```python
ball[['wides','noballs','byes' ,'legbyes']]=ball[['wides','noballs','byes' ,'legbyes']].fillna(0)
```

#### Analyzing Different types of features

We explore the distribution of the different features by displaying the counts of its unique values.

```python
stadiums=ball['venue'].unique()
stadium_dict = { stadium : i + 1 for i, stadium in enumerate(stadiums)}
stadium_dict
```

#### New Dataframe for better Overview

We also made some changes and created a new DataFrame `results` which includes some important feautures like the total runs scored on a single delivery, the cumulative runs scored by the batting team in the current innings and finally dropped some columns that we didn't consider of much importance.

```python
result = ball.groupby(['match_id', 'innings'])[['runs_off_bat', 'extras', 'wides', 'noballs', 'byes', 'legbyes']].sum()

result = result.reset_index()
result.drop(['noballs' , 'byes' , 'legbyes','wides'], axis=1, inplace=True)
result['total'] = result['runs_off_bat'] + result['extras']
result.drop(['runs_off_bat' , 'extras'], axis=1, inplace=True)
```

Then we merge it with the main Df `ball` based on common features, 'match_id' and 'innings'.

```python
ball = ball.merge(result, on=['match_id', 'innings'])
ball
```

#### Furthermore

Finally, with some useful attributes being extracted, we finally complete the cleaning of the data with the following features.

```python
ball.drop(['noballs' , 'byes' , 'legbyes','wides'], axis=1, inplace=True)
ball['cumulative_runs'] = ball.groupby(['match_id', 'innings'])['runs_off_bat'].cumsum() + ball.groupby(['match_id', 'innings'])['extras'].cumsum()
ball['wickets'] = ball.groupby(['match_id', 'innings'])['wicket'].cumsum()
ball.drop(['striker' , 'non_striker' , 'bowler','match_id','runs_off_bat','extras','innings','wicket','start_date'], axis=1, inplace=True)
```

### Step 3: Data Transformation and Label Encoding

#### Mapping and Encoding Venues

Making some hashes so as to obtain the data in a useful format for the training of our model

```python
ball['venue'] = ball['venue'].map(stadium_dict)
```

#### Making dummies

We use the `pd.get_dummies()` function to get the data in format as of OneHot vectors which can be useful in case of categorical data which here is, `batting_team` and `bowling_team`.

```python
ball = pd.get_dummies(ball, columns=['batting_team', 'bowling_team'], dtype=int)
```

####

Summarizing, the following columns were handled in the the preprocessing.

- 'batting_team'
- 'bowling_team'
- 'venue'
- 'wicket'
- 'wicket_type'

## Model Training and Evaluation

### Training, Validation and Testing Sets

- **Creating Training Set (`X_train`):**

  - `X_train = ball.drop(labels='total', axis=1)[ball['ball'] < 30]`
    - Generates the training set `X_train` by excluding the 'total' column from `ball` DataFrame where the number of overs is less than 30.

- **Creating Validation Set (`X_val`):**

  - `X_val = ball.drop(labels='total', axis=1)[(ball['ball'] >= 30) & (ball['ball'] < 40)]`
    - Produces the validation set `X_val` by excluding the 'total' column from `ball` DataFrame where the number of overs is between 30 (inclusive) and 40 (exclusive).

- **Creating Test Set (`X_test`):**

  - `X_test = ball.drop(labels='total', axis=1)[ball['ball'] >= 40]`
    - Constructs the test set `X_test` by excluding the 'total' column from `ball` DataFrame where the number of overs is 40 or more.

- ```python
   y_train = ball[ball['ball']< 30]['total'].values
  y_val = ball[(ball['ball']>= 30) & (ball['ball'] < 40)]['total'].values
  y_test = ball[ball['ball']>= 40]['total'].values
  ```

### Classical Models

#### Linear Regression

The linear regression model is used for the initial prediction here, with using MSE and R² as the Evaluation metrics.

```python
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test).round(0).astype(int)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)
```

#### Ridge Regression

In this model, we used `GridSearchCV` with several values of the `alpha` hyperparameter so as to obtain the best params for our model.

```python
ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train,y_train)
y_pred = ridge_regressor.predict(X_test).round(0).astype(int)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

#### Lasso Regression

Similary for Lasso Regression, we tried `GridSearchCV`

```python
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X_train,y_train)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)
y_pred = lasso_regressor.predict(X_test).round(0).astype(int)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)
```

### Neural Network

Here, we have made a dense neural network with 4 hidden layers and the `adam` optimizer.

```python
src = keras.Sequential([
    keras.layers.Dense(16, input_shape=(X_train.shape[1],), activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(2, activation='relu'),
    keras.layers.Dense(1)
])

src.compile(optimizer='adam', loss='mean_squared_error')
src.fit(X_train, y_train, epochs=150, batch_size=32,validation_data=(X_val, y_val))
```

### API
In the API, we will take the inputs as:
`Batting Team: `
`Bowling Team: `
`Venue: `

and then send the output as the runs predicted.

## ODI Match Winner

For predicting the match winner, we use the Keras DNN model for the classification purpose. The steps in include general preprocessing, data transformation with the use of `ColumnTransformer` and final training. We also have made use of Pickle for making the main component of API.

#### Preprocessing
The preprocessing remains similar here, we use the files, `deliveries.csv` and `matches.csv` for the prediction of the winner of the match. We move throught various dataframes to get the final df for our model, `final_cricket`, which consists of columns:
- 'match_id'
- 'team1'
- 'team2'
- 'cumulative_runs'
- 'runs_left'
- 'crr'
- 'rrr'
- 'wicket_type'
- 'wickets_left'
- 'winner'
- 'result'

and more. We define the Training and Testing Sets afterwards.

#### Pipeline steps
```python
# ODI Match Winner
trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse=False, drop='first'), ['batting_team', 'bowling_team', 'venue'])
], remainder='passthrough')
def final():
    winner = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(33,)), 
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    winner.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return winner
model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn = final, epochs = 10, batch_size = 32)
pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', model)
])
```

#### Model Evaluations
Here are a few functions that prove useful to us in summarizing the model and also being helpful in creating useful insights of our model and thus for the next step of pickling for API.

```python
def match_summary(row):
    print("Batting Team-" + row['batting_team'] + " | Bowling Team-" + row['bowling_team'] + " | Target- " + str(row['total']))
def match_progression(x_df,match_id,pipe):
    match = x_df[x_df['match_id'] == match_id]
    match['fractional_part'] = match['ball'].apply(lambda x: x - int(x))
    tolerance = 1e-10
    match = match[np.isclose(match['fractional_part'], 0.6, atol=tolerance)]
    temp_df = match[['batting_team','bowling_team','venue','runs_left','balls_left','wickets_left','total','crr','rrr']].dropna()
    temp_df = temp_df[temp_df['balls_left'] != 0]
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0]*100,1)
    temp_df['win'] = np.round(result.T[1]*100,1)
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1)
    
    target = temp_df['total'].values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets = list(temp_df['wickets_left'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0,10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]
    
    print("Target-",target)
    temp_df = temp_df[['end_of_over','runs_after_over','wickets_in_over','lose','win']]
    return temp_df,target
temp_df,target = match_progression(final_cricket,9,pipe)
temp_df
```

#### API
In the api we will take the following inputs:
`Batting Team: `
`Bowling Team: `
`Venue: `
`Target: `
`Runs Left: `
`Wickets Left: `

and then send the output, the predicted Winner.

### Team Composition Prediction
For predicting Team Composition for a match, we used the Dataset from [Howstat.com]('http://howstat.com/cricket/Statistics/WorldCup/SeriesAnalysis.asp?SeriesCode=1117'). We got field like
- Batting Strike Rate
- Bowling Strike Rate
- Averages
- Economies of Bowlers
- Inning-wise distinctions

and more data from this dataset. 

#### Preprocessing

We converted the features to numeric features in the DataFrame, handles null values which mostly includes filling with zero which was the most suitable according to the null values.

#### Training
We declared the features as following:
``` python
features = ['NO', '50s', '100s', '0s', 'HS', 'Runs', 'Bat_S/R', 'Bat_Avg', 'Ca', 'St', '% Team Runs', 'O', 'M', 'R', 'W', '4w', 'Best', 'Bowl_Avg', 'Bowl_S/R', 'E/R']
```

and the target as `Playing_XI`.

#### Model

Here as well, we have used Dense NN with dropout layers for optimisation. 

```python
play_xi = tf.keras.Sequential([
        tf.keras.layers.Input(len(features),), 
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

play_xi.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

#### Prediction

We Predict the probabilities, for a player playing in team or not.

```python
train_probabilities = play_xi.predict(X_train_scaled)
test_probabilities = play_xi.predict(X_test_scaled)
player_stats.loc[X_train.index, 'PredictedProbabilities'] = train_probabilities
player_stats.loc[X_test.index, 'PredictedProbabilities'] = test_probabilities
for country in player_stats['Country'].unique():
    top_players = player_stats[player_stats['Country'] == country].nlargest(11, 'PredictedProbabilities')['Player']
    print(f"\nTop 11 players with the highest probability for {country}:\n{top_players}")
```

#### API
In the api we will take the following inputs:
`Batting Team: `
`Bowling Team: `
`Venue: `

and then send the output, the predicted Team Compositions.

### Sixes in the Tournament

Here, we have use the `deliveries.csv` file from the ICC WC 2023 Kaggle Dataset. We have also used another dataset from Kaggle for getting the number of sixes from the recent few matches. 

#### Preprocessing

This includes dropping the null values where needed, filling the null values with 0 in the extras field as done earlier. From this, we get a time series data that can be useful for prediction of our target.

#### Model

Our model is as follows: 

```python
model = Sequential()
model.add(LSTM(units=50, return_sequences = True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences = True))
model.add(LSTM(units=25))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
```

We have used RMS error as the Evaluation Metric here. We have plotted some graphs for the same in the notebooks.


#### API
Here, we don't take any input in the API, We just return the prediction of number of sixes in the tournament.