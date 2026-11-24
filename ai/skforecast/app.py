import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from skforecast.recursive import ForecasterRecursive

train_df = pd.read_csv('train.csv')
train_df['Date'] = pd.to_datetime(train_df['Date'])
sample_data = train_df[(train_df['store'] == 0) & (train_df['product'] == 0)]
sample_data = sample_data.set_index('Date')
sample_data = sample_data.sort_index()
sample_data = sample_data.asfreq('MS')

sns.lineplot(x='Date', y='number_sold', data=sample_data)
plt.show()
#print('sample_data', sample_data)
def add_lagged_variables(df, lag_count=1):
    new_df = df.copy()
    for i in range(1, lag_count + 1):
        new_df[f'lag_{i}'] = new_df['number_sold'].shift(i)
    #print('new_df', new_df)
    return new_df.dropna()

lagged_data = add_lagged_variables(sample_data, 6)
print('lagged_data', lagged_data)

steps = 5
data_train = lagged_data[:-steps]
data_test  = lagged_data[-steps:]
print('data_train', data_train)
print('data_test', data_test)


#set_dark_theme()
fig, ax = plt.subplots(figsize=(6, 2.5))
data_train['number_sold'].plot(ax=ax, label='train')
data_test['number_sold'].plot(ax=ax, label='test')
ax.legend()

forecaster = ForecasterRecursive(
                 estimator = RandomForestRegressor(random_state=123),
                 lags      = 6
             )
forecaster.fit(y=data_train['number_sold'])
forecaster