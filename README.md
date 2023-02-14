---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.2
  nbformat: 4
  nbformat_minor: 5
---

::: {#30f815eb-8fef-4eb7-ae70-b5a1538aeba5 .cell .markdown id="30f815eb-8fef-4eb7-ae70-b5a1538aeba5"}
# Stock trend prediction
:::

::: {#7740dd09-65a4-453c-8b3a-6a2e5b80c0c8 .cell .markdown id="7740dd09-65a4-453c-8b3a-6a2e5b80c0c8"}
### *Import library*
:::

::: {#c9a6e626-cc01-48c2-b6e1-8536a51b1c09 .cell .code id="c9a6e626-cc01-48c2-b6e1-8536a51b1c09"}
``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential,load_model
from keras.layers import LSTM,Dense,Dropout
```
:::

::: {#653b2426-402e-440d-adba-f55c27c31656 .cell .markdown id="653b2426-402e-440d-adba-f55c27c31656"}
### Stock data from yahoo finance
:::

::: {#b34340f9-6d0f-4da1-9ec1-3e68d9ebda63 .cell .code id="b34340f9-6d0f-4da1-9ec1-3e68d9ebda63" outputId="a5ec2929-e1c7-47cb-85fb-4403d47174bd"}
``` python
from pandas_datareader import data

stock=data.DataReader('GOOG', 'yahoo', '1999-06-01', '2016-06-13')
stock.head()
```

::: {.output .execute_result execution_count="15"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2004-08-19</th>
      <td>51.835709</td>
      <td>47.800831</td>
      <td>49.813290</td>
      <td>49.982655</td>
      <td>44871361.0</td>
      <td>49.982655</td>
    </tr>
    <tr>
      <th>2004-08-20</th>
      <td>54.336334</td>
      <td>50.062355</td>
      <td>50.316402</td>
      <td>53.952770</td>
      <td>22942874.0</td>
      <td>53.952770</td>
    </tr>
    <tr>
      <th>2004-08-23</th>
      <td>56.528118</td>
      <td>54.321388</td>
      <td>55.168217</td>
      <td>54.495735</td>
      <td>18342897.0</td>
      <td>54.495735</td>
    </tr>
    <tr>
      <th>2004-08-24</th>
      <td>55.591629</td>
      <td>51.591621</td>
      <td>55.412300</td>
      <td>52.239197</td>
      <td>15319808.0</td>
      <td>52.239197</td>
    </tr>
    <tr>
      <th>2004-08-25</th>
      <td>53.798351</td>
      <td>51.746044</td>
      <td>52.284027</td>
      <td>52.802086</td>
      <td>9232276.0</td>
      <td>52.802086</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#3c3d86b0-df4e-4348-b2d6-93d235633749 .cell .code id="3c3d86b0-df4e-4348-b2d6-93d235633749" outputId="0a6e0733-81e1-4fac-99eb-7fcde9fed551"}
``` python
stock.shape
```

::: {.output .execute_result execution_count="16"}
    (2975, 6)
:::
:::

::: {#0502bb8c-487a-4e37-bde8-e61dc69cfffc .cell .code id="0502bb8c-487a-4e37-bde8-e61dc69cfffc" outputId="a2b0fa31-c884-4551-c3c1-f17eef5c1f6a"}
``` python
fig,ax=plt.subplots(figsize=(16,8))
ax.set_facecolor('#000041')
ax.plot(stock['Open'],color='red',label='Open price')
plt.plot(stock['Close'],color='cyan',label='Close price')
plt.legend()
plt.show()
```

::: {.output .display_data}
![](vertopal_084c6997c29445b4ac09b3d11a883aa1/94fcea4521cca03cd79b4f9eabe8327b96864c7b.png)
:::
:::

::: {#ce26826d-53bf-41d6-9e7a-94f93f20f8a3 .cell .code id="ce26826d-53bf-41d6-9e7a-94f93f20f8a3" outputId="814c5cfc-b78e-41a6-f1cc-8569039cd9c3"}
``` python
# stock=stock['Open']
stock=stock.reshape(-1,1)
stock
```

::: {.output .execute_result execution_count="23"}
    array([[ 49.81328964],
           [ 50.31640244],
           [ 55.16821671],
           ...,
           [722.86999512],
           [719.4699707 ],
           [716.51000977]])
:::
:::

::: {#9da7a8d3-b7bc-41a2-81c1-2f3bec6d3a97 .cell .markdown id="9da7a8d3-b7bc-41a2-81c1-2f3bec6d3a97"}
### Split training data and testing data
:::

::: {#96d884a6-04c9-48e1-a91a-ab74e3caf69f .cell .code id="96d884a6-04c9-48e1-a91a-ab74e3caf69f"}
``` python
train_data=np.array(stock[:int(stock.shape[0]*0.8)])
test_data=np.array(stock[int(stock.shape[0]*0.8):])
```
:::

::: {#ebee759f-7364-48db-8e21-d3c5aa982460 .cell .markdown id="ebee759f-7364-48db-8e21-d3c5aa982460"}
### Scale data 0 and 1
:::

::: {#a05aac73-4984-494b-8b69-f3ef1048cd05 .cell .code id="a05aac73-4984-494b-8b69-f3ef1048cd05"}
``` python
scaler=MinMaxScaler(feature_range=(0,1))
train_data=scaler.fit_transform(train_data)
test_data=scaler.transform(test_data)
```
:::

::: {#1795cda8-09f7-477d-a250-16b18002653a .cell .markdown id="1795cda8-09f7-477d-a250-16b18002653a"}
### create a function to create dataset
:::

::: {#9659e9fc-3ed0-4f41-89ac-a5a19943e88f .cell .code id="9659e9fc-3ed0-4f41-89ac-a5a19943e88f"}
``` python
def dataset(stock):
    x=[]
    y=[]
    for i in range(50,stock.shape[0]):
        x.append(stock[i-50:i,0])
        y.append(stock[i,0])
    x=np.array(x)
    y=np.array(y)
    return x,y
```
:::

::: {#1e0dba1f-7dbb-4419-ad8a-46241f213447 .cell .markdown id="1e0dba1f-7dbb-4419-ad8a-46241f213447"}
### create training and testing data by call our dataset function
:::

::: {#c2121625-f8b9-4299-b1c3-81a79667d395 .cell .code id="c2121625-f8b9-4299-b1c3-81a79667d395"}
``` python
x_train,y_train=dataset(train_data)
x_test,y_test=dataset(test_data)
```
:::

::: {#fef0f9d7-051d-400f-a60a-8d6f770cdd01 .cell .markdown id="fef0f9d7-051d-400f-a60a-8d6f770cdd01"}
#### we need to reshape our data using LSTM layers(Long short term model)
:::

::: {#ba5f6a16-572a-48aa-96a3-41df6e7ddea7 .cell .code id="ba5f6a16-572a-48aa-96a3-41df6e7ddea7"}
``` python
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
```
:::

::: {#6217c229-e8af-4888-9942-919bc09b00d7 .cell .markdown id="6217c229-e8af-4888-9942-919bc09b00d7"}
### Create the Model
:::

::: {#dd4519e9-370b-416e-bfc6-847f3a503630 .cell .code id="dd4519e9-370b-416e-bfc6-847f3a503630"}
``` python
model=Sequential()
model.add(LSTM(units=96,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=96,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96))
model.add(Dropout(0.2))
model.add(Dense(units=1))
```
:::

::: {#c62725c8-2d7b-4a47-a419-f4d0e8cece1b .cell .markdown id="c62725c8-2d7b-4a47-a419-f4d0e8cece1b"}
### Reshape feature for LSTM layers
:::

::: {#db296642-8b88-4a1b-bb46-7b3e177d7372 .cell .code id="db296642-8b88-4a1b-bb46-7b3e177d7372"}
``` python
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
```
:::

::: {#632e0d17-b06f-4b43-b0b8-02f109587d92 .cell .markdown id="632e0d17-b06f-4b43-b0b8-02f109587d92"}
### Compile our Model
:::

::: {#455d2f4b-c04d-44c0-940a-6c89f59dcbea .cell .code id="455d2f4b-c04d-44c0-940a-6c89f59dcbea"}
``` python
model.compile(loss='mean_squared_error',optimizer='adam')
```
:::

::: {#95fbe96e-074f-4053-9e0f-9581f54e7905 .cell .markdown id="95fbe96e-074f-4053-9e0f-9581f54e7905"}
#### save our Model and start the training
:::

::: {#2c93ac50-5a5e-4ed0-ac43-521b2b81d166 .cell .code id="2c93ac50-5a5e-4ed0-ac43-521b2b81d166" outputId="428c59bc-e3a0-4208-f5d4-55d0c5f833ca"}
``` python
model.fit(x_train,y_train,epochs=50,batch_size=32)
model.save('stock_prediction.h5')
```

::: {.output .stream .stdout}
    Epoch 1/50
    73/73 [==============================] - 24s 196ms/step - loss: 0.0115
    Epoch 2/50
    73/73 [==============================] - 15s 208ms/step - loss: 0.0024
    Epoch 3/50
    73/73 [==============================] - 16s 226ms/step - loss: 0.0020
    Epoch 4/50
    73/73 [==============================] - 17s 227ms/step - loss: 0.0023
    Epoch 5/50
    73/73 [==============================] - 17s 227ms/step - loss: 0.0019
    Epoch 6/50
    73/73 [==============================] - 17s 227ms/step - loss: 0.0017
    Epoch 7/50
    73/73 [==============================] - 15s 209ms/step - loss: 0.0015
    Epoch 8/50
    73/73 [==============================] - 14s 197ms/step - loss: 0.0015
    Epoch 9/50
    73/73 [==============================] - 14s 196ms/step - loss: 0.0015
    Epoch 10/50
    73/73 [==============================] - 14s 196ms/step - loss: 0.0015
    Epoch 11/50
    73/73 [==============================] - 16s 226ms/step - loss: 0.0013
    Epoch 12/50
    73/73 [==============================] - 17s 231ms/step - loss: 0.0014
    Epoch 13/50
    73/73 [==============================] - 17s 229ms/step - loss: 0.0013
    Epoch 14/50
    73/73 [==============================] - 17s 229ms/step - loss: 0.0011
    Epoch 15/50
    73/73 [==============================] - 15s 201ms/step - loss: 9.9512e-04
    Epoch 16/50
    73/73 [==============================] - 14s 197ms/step - loss: 9.5737e-04
    Epoch 17/50
    73/73 [==============================] - 14s 197ms/step - loss: 9.8520e-04
    Epoch 18/50
    73/73 [==============================] - 15s 205ms/step - loss: 0.0011
    Epoch 19/50
    73/73 [==============================] - 17s 230ms/step - loss: 8.8507e-04
    Epoch 20/50
    73/73 [==============================] - 17s 231ms/step - loss: 9.3658e-04
    Epoch 21/50
    73/73 [==============================] - 17s 231ms/step - loss: 9.7068e-04
    Epoch 22/50
    73/73 [==============================] - 16s 223ms/step - loss: 9.4676e-04
    Epoch 23/50
    73/73 [==============================] - 14s 197ms/step - loss: 8.7933e-04
    Epoch 24/50
    73/73 [==============================] - 14s 198ms/step - loss: 8.8784e-04
    Epoch 25/50
    73/73 [==============================] - 14s 197ms/step - loss: 9.1487e-04
    Epoch 26/50
    73/73 [==============================] - 15s 205ms/step - loss: 8.9545e-04
    Epoch 27/50
    73/73 [==============================] - 17s 233ms/step - loss: 7.9457e-04
    Epoch 28/50
    73/73 [==============================] - 17s 231ms/step - loss: 7.5912e-04
    Epoch 29/50
    73/73 [==============================] - 17s 233ms/step - loss: 7.2577e-04
    Epoch 30/50
    73/73 [==============================] - 16s 214ms/step - loss: 7.6448e-04
    Epoch 31/50
    73/73 [==============================] - 15s 199ms/step - loss: 6.7472e-04
    Epoch 32/50
    73/73 [==============================] - 14s 198ms/step - loss: 7.1527e-04
    Epoch 33/50
    73/73 [==============================] - 14s 198ms/step - loss: 7.2515e-04
    Epoch 34/50
    73/73 [==============================] - 15s 210ms/step - loss: 7.2840e-04
    Epoch 35/50
    73/73 [==============================] - 17s 234ms/step - loss: 7.3191e-04
    Epoch 36/50
    73/73 [==============================] - 17s 230ms/step - loss: 6.3490e-04
    Epoch 37/50
    73/73 [==============================] - 17s 232ms/step - loss: 6.1274e-04
    Epoch 38/50
    73/73 [==============================] - 15s 203ms/step - loss: 6.2752e-04
    Epoch 39/50
    73/73 [==============================] - 15s 200ms/step - loss: 6.2549e-04
    Epoch 40/50
    73/73 [==============================] - 15s 199ms/step - loss: 6.6006e-04
    Epoch 41/50
    73/73 [==============================] - 15s 200ms/step - loss: 6.1807e-04
    Epoch 42/50
    73/73 [==============================] - 16s 223ms/step - loss: 6.5746e-04
    Epoch 43/50
    73/73 [==============================] - 19s 261ms/step - loss: 5.7168e-04
    Epoch 44/50
    73/73 [==============================] - 17s 232ms/step - loss: 5.6556e-04
    Epoch 45/50
    73/73 [==============================] - 16s 226ms/step - loss: 6.5408e-04
    Epoch 46/50
    73/73 [==============================] - 15s 199ms/step - loss: 5.0303e-04
    Epoch 47/50
    73/73 [==============================] - 15s 199ms/step - loss: 5.5713e-04
    Epoch 48/50
    73/73 [==============================] - 14s 198ms/step - loss: 5.7554e-04
    Epoch 49/50
    73/73 [==============================] - 15s 199ms/step - loss: 5.1364e-04
    Epoch 50/50
    73/73 [==============================] - 16s 224ms/step - loss: 5.5195e-04
:::
:::

::: {#16a16b01-ffb9-4770-b9cf-2934200a907d .cell .markdown id="16a16b01-ffb9-4770-b9cf-2934200a907d"}
### now load the model
:::

::: {#fdc732de-5703-4a5d-aca5-3dc1753c2d1e .cell .code id="fdc732de-5703-4a5d-aca5-3dc1753c2d1e"}
``` python
model=load_model('stock_prediction.h5')
```
:::

::: {#9cf25191-2127-43f8-9eac-fd58774e4985 .cell .markdown id="9cf25191-2127-43f8-9eac-fd58774e4985"}
### visualize our predicted data
:::

::: {#6a3a5c22-4f40-4cef-ab8e-ab2194fde340 .cell .code id="6a3a5c22-4f40-4cef-ab8e-ab2194fde340" outputId="1c83506d-f190-49c7-842d-fba73eeee069"}
``` python
prediction=model.predict(x_test)
prediction=scaler.inverse_transform(prediction)
y_test_scale=scaler.inverse_transform(y_test.reshape(-1,1))
```

::: {.output .stream .stdout}
    18/18 [==============================] - 1s 72ms/step
:::
:::

::: {#e53cb1b9-75c4-483e-9bc3-3101012f8d5c .cell .markdown id="e53cb1b9-75c4-483e-9bc3-3101012f8d5c"}
### Result
:::

::: {#0dd42470-4e79-4fbb-ac5e-ab2655e03bd2 .cell .code id="0dd42470-4e79-4fbb-ac5e-ab2655e03bd2" outputId="6a3b8136-0cad-4885-c134-fab9f5e68751"}
``` python
fig,ax=plt.subplots(figsize=(16,8))
ax.set_facecolor('#000041')
ax.plot(y_test_scale,color='red',label='original price')
plt.plot(prediction,color='cyan',label='predict price')
plt.legend()
plt.show()
```

::: {.output .display_data}
![](vertopal_084c6997c29445b4ac09b3d11a883aa1/84501bb593661c9ddb082b2cd1cf671126acee1e.png)
:::
:::

::: {#23e13ff3-6446-44ba-91df-5bee87faece0 .cell .code id="23e13ff3-6446-44ba-91df-5bee87faece0"}
``` python
import finplot as fplt
import yfinance

symbol = 'GOOG'
df = yfinance.download(symbol)

ax = fplt.create_plot(symbol)

fplt.candlestick_ochl(df[['Open', 'Close', 'High', 'Low']], ax=ax)
fplt.plot(df['Close'].rolling(200).mean(), ax=ax, legend='SMA 200')
fplt.plot(df['Close'].rolling(50).mean(), ax=ax, legend='SMA 50')
fplt.plot(df['Close'].rolling(20).mean(), ax=ax, legend='SMA 20')

fplt.volume_ocv(df[['Open', 'Close', 'Volume']], ax=ax.overlay())
fplt.show()
```
:::

::: {#60c04192-6375-43eb-ada5-1c816eebe656 .cell .code id="60c04192-6375-43eb-ada5-1c816eebe656"}
``` python
```
:::

::: {#6c4dfe12-4337-4894-aa06-08a80b529a97 .cell .code id="6c4dfe12-4337-4894-aa06-08a80b529a97"}
``` python
```
:::
