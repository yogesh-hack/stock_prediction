import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential,load_model
from keras.layers import LSTM,Dense,Dropout

import streamlit as st
from pandas_datareader import data
st.title("Stock Trend Prediction")

user_input=st.selectbox('Select which Company You see the stock data:',
('Microsoft','Apple','tesla','Tata Motors','Alphabet','Amazon','Visa Inc.','NVIDIA','Exxon Mobil','Walmart','Mastercard','Alibaba Group','Toyota Motor','PepsiCo','Oracle','Accenture','Uber Technologies')
)
dict={
    'Microsoft':'MSFT',
    'Apple':'AAPL',
    'Tesla':'TSLA',
    'Alphabet':'GOOGL',
    'Amazon':'AMZN',
    'Visa Inc.':'V',
    'NVIDIA':'NVDA',
    'Exxon Mobil':'XOM',
    'Walmart':'WMT',
    'Mastercard':'MA',
    'Alibaba Group':'BABA',
    'Toyota Motor':'TM',
    'PepsiCo':'PEP',
    'Oracle':'ORCL',
    'Accenture':'ACN',
    'Qualcomm':'QCOM',
    'Uber Technologies':'UBER',
    'Tata Motors':'TTM'
}

start_date=st.date_input("Start Date:",value=pd.to_datetime("1999-06-01",format="%Y-%m-%d"))
end_date=st.date_input("End date:",value=pd.to_datetime("2012-06-02",format="%Y-%m-%d"))

#Convert date into string
From=start_date.strftime("%Y-%m-%d")
To=end_date.strftime("%Y-%m-%d")


st.write('Stock data from :',start_date,'To :',end_date)

# show Microsoft stock data
st.subheader("Stock data Description are:")

inp=dict[user_input]

stock=data.DataReader(inp, 'yahoo', From, To)
st.write(stock.describe())

st.subheader("Stock Data list are :")
st.write(stock)

st.subheader("Close Price vs Open price")
fig,ax=plt.subplots(figsize=(16,8))
ax.set_facecolor('#000041')
ax.plot(stock['Open'],color='red',label='Open price')
plt.plot(stock['Close'],color='cyan',label='Close price')
plt.legend()
plt.title("Stock Trend Chart")
st.pyplot(fig)

stock=stock['Open'].values
stock=stock.reshape(-1,1)
st.write("Reshaping the Open price ")
st.write(stock)

train_data=np.array(stock[:int(stock.shape[0]*0.8)])
test_data=np.array(stock[int(stock.shape[0]*0.8):])

scaler=MinMaxScaler(feature_range=(0,1))
train_data=scaler.fit_transform(train_data)
test_data=scaler.transform(test_data)

def dataset(stock):
    x=[]
    y=[]
    for i in range(50,stock.shape[0]):
        x.append(stock[i-50:i,0])
        y.append(stock[i,0])
    x=np.array(x)
    y=np.array(y)
    return x,y

x_train,y_train=dataset(train_data)
x_test,y_test=dataset(test_data)

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

model=load_model('stock_prediction.h5')

prediction=model.predict(x_test)
prediction=scaler.inverse_transform(prediction)
y_test_scale=scaler.inverse_transform(y_test.reshape(-1,1))



st.subheader("Prediction price vs Original Price")
fig2,ax=plt.subplots(figsize=(16,8))
ax.set_facecolor('#000041')
ax.plot(y_test_scale,color='red',label='original price')
plt.plot(prediction,color='cyan',label='predict price')
plt.title('Predition stock price chart')
plt.legend()
st.pyplot(fig2)

stock=data.DataReader(inp, 'yahoo', From, To)
st.subheader("Prediction Price")
p=pd.DataFrame(prediction)
st.write(p)
st.subheader("Original(Test) price")
d=pd.DataFrame(y_test_scale)
st.write(d)

st.write("Our Prediction price is almost same as the Original(Test) Price")

st.write('Made by Yogesh Bahgel')

