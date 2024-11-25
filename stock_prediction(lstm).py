!pip install yfinance

#import libraries
import math
import yfinance as yf # Changed import to yfinance
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Download data using yfinance
df = yf.download(tickers="FSL.NS", start="2022-01-01", end="2024-11-25")

# Display the first few rows of the dataframe
df

df.shape

plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price ', fontsize=18)
plt.show()

print(df.head())
print(df.columns)

# Instead of using df.filter, use square brackets to select the "Close" column
data = df[["Close"]]
dataset = data.values
training_data_len = math.ceil(len(dataset) * .8)
training_data_len

#scale the data
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
scaled_data

train_data=scaled_data[0:training_data_len , :]
x_train=[]
y_train=[]
for i in range(90,len(train_data)):
  x_train.append(train_data[i-90:i,0])
  y_train.append(train_data[i,0])
  if i<=90:
    print(x_train)
    print(y_train)
    print()

x_train,y_train=np.array(x_train),np.array(y_train)
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
print(x_train.shape)

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(90,1)))
model.add(LSTM(50,return_sequences=False))

model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mean_squared_error')

model.fit(x_train,y_train,batch_size=1,epochs=1)

#create test dataset
test_data=scaled_data[training_data_len-90: , :]
x_test=[]
y_test=dataset[training_data_len: , :]
for i in range(90,len(test_data)):
  x_test.append(test_data[i-90:i,0])

x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)

#RSME
rsme=np.sqrt(np.mean(((predictions-y_test)**2)))
rsme

train=data[:training_data_len]
valid=data[training_data_len:]
valid['Predictions']=predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price ', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'],loc='upper left')
plt.show()

#show the valid and predictied price
valid

# @title ('Actual_Price', 'FSL.NS') vs ('Predicted_Price', '')

from matplotlib import pyplot as plt
valid.plot(kind='scatter', x="('Actual_Price', 'FSL.NS')", y="('Predicted_Price', '')", s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)

# @title ('Predicted_Price', '')

from matplotlib import pyplot as plt
valid["('Predicted_Price', '')"].plot(kind='line', figsize=(8, 4), title="('Predicted_Price', '')")
plt.gca().spines[['top', 'right']].set_visible(False)

# @title Distribution of FSL.NS Closing Prices

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.hist(df['Close'], bins=20, edgecolor='black')
plt.title('Distribution of FSL.NS Closing Prices')
plt.xlabel('Closing Price')
_ = plt.ylabel('Frequency')

stock_quote= yf.download(tickers="FSL.NS", start="2022-01-01", end="2024-11-25")
# Check if stock_quote is empty
if stock_quote.empty:
    raise ValueError("The stock_quote DataFrame is empty. Please check the ticker symbol and date range.")

# Use 'Close' instead of 'close' for filtering, because 'Close' is a column name
#in the downloaded data
new_df = stock_quote[['Close']]
# Check if new_df is empty
if new_df.empty:
  raise ValueError("The new_df DataFrame is empty. Please check the stock_quote DataFrame.")

last_90_days = new_df[-90:].values
# Reshape last_90_days to have at least one feature
last_90_days = last_90_days.reshape(-1, 1)  # Reshape to (90, 1)

last_90_days_scaled = scaler.transform(last_90_days)
X_test = []
X_test.append(last_90_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)
