
# coding: utf-8

# In[80]:


import pandas as pd
import numpy as np
import datetime
from dateutil.parser import parse
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error
import warnings 
warnings.filterwarnings('ignore')


# In[81]:


data = pd.read_csv('Item_1_training.csv')


# In[82]:


data_forecast = pd.read_csv('Item_1_forecast.csv')


# In[83]:


data_forecast['Date'] = pd.to_datetime(data_forecast['Date'],format='%d/%m/%Y') 
data_forecast = data_forecast.set_index('Date')


# In[84]:


Item = data["Product"][0]
Customer = data["Customer"][0]
Organisation = data["Organisation"][0]


# In[85]:


univarient_dataset = data[['Date','Shipments History']]


# In[86]:


univarient_dataset['Date'] = pd.to_datetime(univarient_dataset['Date'],format='%d/%m/%Y') 
univarient_dataset = univarient_dataset.set_index('Date')


# In[87]:


df = univarient_dataset["Shipments History"]


# In[88]:


rcParams['figure.figsize'] = 18, 8


# In[89]:


dftest = adfuller(df, autolag='AIC')


# In[90]:


d=0
if(dftest[1]<0.05):
    d=0
else:
    df_data = df
    while(dftest[1]>0.05 and d<2):
        d = d+1
        df_differenced = df_data.diff().dropna()
        dftest = adfuller(df_differenced, autolag='AIC')
        df_data = df_differenced
    df = df_data


# In[91]:


size = int(len(df) * 0.75)
train, test = df[0:size], univarient_dataset["Shipments History"][size:len(df)]


# In[92]:


len(train),len(test)


# In[93]:


df_train = train.astype('float32')


# In[94]:


import sys
val = sys.maxsize
p = 0
q = 0
for i in range(10):
    for j in range(10):
        try:
            model = ARIMA(df_train, order=(i,d,j))
            results_ARMA = model.fit(disp=-1)  
            print('Order = ', i,j)
            print('AIC: ', results_ARMA.aic)
            if(results_ARMA.aic<val):
                val = results_ARMA.aic
                p = i
                q = j
        except :
            print('Order = ', i,j)
            print('pass')
        
print('Optimum lag is', p,q)
print('AIC_value', val)


# In[95]:


test_model = ARIMA(df_train, order=(p, d, q))  
test_results_ARMA = test_model.fit(disp=-1)  
rcParams['figure.figsize'] = 18, 8
plt.plot(df_train)
plt.plot(test_results_ARMA.fittedvalues, color='red')
plt.title('ARMA Model')
plt.show()


# In[96]:


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


# In[97]:


mean_absolute_percentage_error(df_train[d:],test_results_ARMA.fittedvalues)


# In[98]:


pred_test = test_results_ARMA.predict(start = test.index[0], end= test.index[-1])


# In[99]:


if d==0:
    df_test_forecast = pd.DataFrame(pred_test, columns=univarient_dataset.columns )
elif d==1:
    df_test_forecast = pd.DataFrame(pred_test, columns=univarient_dataset.columns + '_1d')
else :
    df_test_forecast = pd.DataFrame(pred_test, columns=univarient_dataset.columns + '_2d')


# In[100]:


df_test_forecast.head()


# In[101]:


def invert_transformation(df_train, df_forecast, d):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        if d==2:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
            df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
        elif d==1:
            df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
        else :
            df_fc[str(col)+'_forecast'] = df_fc[str(col)]
    return df_fc


# In[102]:


df_test_results = invert_transformation(univarient_dataset[0:size], df_test_forecast, d)        

df_test_pred = pd.DataFrame(df_test_results[str(univarient_dataset.columns[0])+'_forecast'], index=df_test_results.index )
df_test_pred.head()


# In[103]:


test


# In[104]:


df_test_pred["Shipments History_forecast"]


# In[105]:


mean_absolute_percentage_error(test.values,df_test_pred["Shipments History_forecast"].values)


# In[106]:


plt.plot(test)
plt.plot(df_test_pred["Shipments History_forecast"], color='red')
plt.title('ARMA Model_Prediction_graph')
plt.show()

#test.values#df_test_pred["Shipment History_forecast"].values#Below part will be used if the main data won't converse for the above p,q valuesimport sys
val = sys.maxsize
p = 0
q = 0
for i in range(10):
    for j in range(10):
        try:
            model = ARIMA(df.astype('float32'), order=(i,d,j))
            results_ARMA = model.fit(disp=-1)  
            print('Order = ', i,j)
            print('AIC: ', results_ARMA.aic)
            if(results_ARMA.aic<val):
                val = results_ARMA.aic
                p = i
                q = j
        except :
            print('Order = ', i,j)
            print('pass')
        
print('Optimum lag is', p,q)
print('AIC_value', val)
# In[107]:


actual_model = ARIMA(df.astype('float32'), order=(p, d, q))  
final_results_ARMA = actual_model.fit(disp=-1)  
plt.plot(df.astype('float32'))
plt.plot(final_results_ARMA.fittedvalues, color='red')
plt.title('ARMA Model')
plt.show()


# In[108]:


predictted_value = final_results_ARMA.predict(start = data_forecast.index[0], end= data_forecast.index[-1])


# In[109]:


if d==0:
    df_forecast = pd.DataFrame(predictted_value, columns=univarient_dataset.columns )
elif d==1:
    df_forecast = pd.DataFrame(predictted_value, columns=univarient_dataset.columns + '_1d')
else :
    df_forecast = pd.DataFrame(predictted_value, columns=univarient_dataset.columns + '_2d')


# In[110]:


df_forecast.head()


# In[111]:


df_results = invert_transformation(univarient_dataset, df_forecast, d)        


# In[112]:


df_results.head()


# In[113]:


Item_col = pd.Series(Item, index=df_results.index )
Cust_col = pd.Series(Customer, index=df_results.index )
Org_col = pd.Series(Organisation, index=df_results.index )


# In[114]:


Model_col = pd.Series('ARIMA', index=df_results.index )


# In[115]:


df_pred = pd.DataFrame({'Shipments History Forecast': df_results[str(univarient_dataset.columns[0])+'_forecast'],'Item': Item_col,'Customer': Cust_col,'Organisation': Org_col,'Model': Model_col})


# In[116]:


df_pred = df_pred.rename_axis('Date').reset_index()


# In[117]:


df_pred.to_csv(Item+'_Arima.csv',index = False)


# In[118]:


plt.plot(univarient_dataset)
plt.plot(df_results['Shipments History_forecast'], color='red')
plt.title('ARMA Model_Prediction_graph')
plt.show()

