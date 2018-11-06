#!/usr/bin/env python
# coding: utf-8

# # Multivariate Multisteps Timeseries Forecast With LSTM
# #### This script use the time series of the data of France98 to predict the Number of Request in the servers in the future. The data are pre processed and stored as CSV files. In each files we have the information minute per minute of the number of the Request recoded in the servers.  We want to use this data to predict the traffic domand within the next 3 minutes. As we know by the "Time Series Analysis" we can use the data as timeseries and training a Recurrent neural network to forecast the information.
# #### The first part is to load the data. After that we "Prepare the data" to be analyzed as Time Series. 
# #### 1) Detrend the data (If s necessary). This procedure is not necessary with our data as we see with the ADF test the data are Stationary and we can go forward.
# #### 2)Scale the data. The data used in the neural Network is scaled. We use the MinMax scaler with value [0,1] to train the LSTM and predict the data
# #### 3)Trasform as supervised problem. This part is described below.
# 
# #### After load and prepare the data we split the data. The splitting consist in two phases. The first phases is the split data in Training and Test set.  The second phases consist to split the data in inputs and outputs of the LSTM. The input are the data used to training the LSTM for predict the outputs. 
# #### Buid the LSTM model and fit the model with the training data. After that we use the Test data input for prediction.
# #### The Resut of the prediction is an array with the values of the Timeseries, as we want to predict the next 3 minutes the results of the LSTM are 3 arrays, each for the forecast at the sequence t+1.t+2,t+3.
# #### After that we compute the RMSE error for next evaluations. In the script we first compute the error in the scale 0,1 and after we compute the error in the original scale of the data. 
# #### Compute the RMSE indicate the goodness of the model. We use the values of the RMSE for make several experiment and found the best selection of LAGs and EPOCH to train the LSTM. The evaluation of RMSE is made in the notebook Evaluation of RMSE.
# 
# ### Next Part
# #### The results of the forecast are stored and used by a Matematial model to check the NFV. We use a matematical model to find out how many Network Virtual Function store based the prediction of the traffic. This for solve the scalability problem.

# In[1]:


import pandas as pd
from pandas import read_csv
from pandas import DatetimeIndex
from pandas import datetime
from pandas import Series
from pandas import concat
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
from numpy import concatenate
from natsort import natsorted, ns
import seaborn as sns
import csv
import glob, os


# In[2]:


""" When using different dataset the part to change this code are :
    Retrive, Parser, The config variable(option), and the part to evaluate the RMSE"""


# ### This part of code present different user-function 
# 
# ### Retrive:
# #### is used for load each file csv that are the result of the previously PARSER process. We collect all parsed CSV and is stored in array, the function return the array with the name of the file.csv and the lenght of the array.
# ### Load dataset:
# #### The function take as input the array with the names of the file in whic is store the data. each rows of the file are stored in one Datafrane with index the Date Time format.
# 
# ### Series to supervised 
# #### This is the most important function. For training the data as TImeseries with LSTM we need to trasform the data as supervised. Each sample of our data is called Timestep. The time steps are the simple that describe the dimension of our simples. Most back in time we want to see and most high is timesteps. The same thing are in the forecast. Most forward we want to predict and more high are the dimension of te Timesteps. The result of this function is a Dataframe that describe have the dimension of the sample in which we have: LAGs and Sequence. 
# #### Lag is how the dimension of back in the past. Sequence are the dimension of the future.
# 

# In[21]:


def retrive():
    a=[]
    i=0
    for file in os.listdir('../data/data/'):
        if file.endswith(".csv"):
            filename = file
            a.append(filename)
            a = [w.replace('.csv', '') for w in a]
    return(a,len(a))

# Define a function. Use the parser function for take the data as datetime, is a function of pandas library.
def parser(x,y,z):
    x =x+':'+y+':'+z
    return datetime.strptime(x,' %d/%b/%Y:%H:%M')

# Load dataset
def load_data(vettore,leng):
    contr = False
    for x in range(leng):      
        day = vettore[x]
        #print(vettore[x])
        if contr == True:
            series_temp = read_csv('../data/data/%s.csv'%day, header=0,
                          parse_dates={'date_time' :['Day','Hour','Minute']}, index_col = 'date_time',
                          squeeze=True, date_parser=parser)
            series_temp = series_temp[series_temp['Byte_count'] != 0]
            series_temp = series_temp[:-1]
            series_temp.head(2)
            series = series.append(series_temp)
            #print("Row:",counter," + ",len(series_temp))
            counter = counter + len(series_temp)
            #print("Rows number (tot): ",counter)
        else:
            series = read_csv('../data/data/%s.csv'%day, header=0,
                          parse_dates={'date_time' :['Day','Hour','Minute']}, index_col = 'date_time',
                          squeeze=True, date_parser=parser)
            counter = len(series)
           # print("Row number (tot): ", counter)
            contr = True
    return series
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


# # Begin

# In[22]:

def runs_exper(exp,epoc,lag):
#Configuration
   
        exp_number = exp
        test_samples = 600
        #test_samples = 20000
        n_features = 2
        n_seq = 3
        n_lag = lag
        epoch = epoc
        n_obs = n_features*n_lag
        n_out = n_features*n_seq #Become the output of the lstm, the n. columns of the output
        path = "Experiments2/"
        filename = "RMSE_Res_N%d_Ep%d_Lg%d.csv" %(exp_number,epoch,n_lag)


        # In[23]:


        #Retrive each filename where the series is stored
        vet,leng = retrive()
        vet = natsorted(vet, key=lambda y: y.lower())
        #Load the dataset from different files, each file name is on vet
        series = load_data(vet,leng)
        series = series[['Byte_count','Request_count']]
        tk_last2h  = series.tail(120)
        tk_last2h = tk_last2h.index 


        # In[ ]:





        # In[24]:


        """ Prepare the time series data
        1)Detrend the data (If s necessary)
        2)Scale the data
        3)Trasform as supervised problem
            """


        # In[25]:


        #scale the data
        scaled_data = DataFrame()
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(series)
        #scaled_data
        #Trasform Data as supervised problem (Sliding Window)
        series_supervised = series_to_supervised(scaled_data, (n_lag-1), (n_seq+1))
        series_supervised.head(2)


        # In[26]:


        #split the series_supervised into TRAINING AND TEST
        series_values = series_supervised.values
        train_series = series_values[:-test_samples, :]
        test_series = series_values[-test_samples:, :]
        #Trasform data as Input and Output for the lstm
        train_input, train_output = train_series[:, :n_obs], train_series[:, -n_out:]
        test_input, test_output = test_series[:, :n_obs], test_series[:, -n_out:]

        print('Training Size: ',train_series.shape[0])
        print('Test Size: ',test_series.shape[0])
        print('Training input data (X): ',train_input.shape)
        print('Training output data (y): ',train_output.shape)
        print('Test input data (X): ',test_input.shape)
        print('Test output data (y): ',test_output.shape)


        # ## Creation and fit the model

        # In[ ]:


        #here we need: train_input, test_input, nlag, nfeatures
        #Reshape for the LSTM, smaple,lag,and feature
        train_input = train_input.reshape(train_input.shape[0], n_lag, n_features)
        test_input = test_input.reshape(test_input.shape[0], n_lag, n_features)
        #train_input.shape
        #Definition of the network
        model = Sequential()
        model.add(LSTM(50, input_shape=(train_input.shape[1], train_input.shape[2])))

        model.add(Dense(n_out))
        model.compile(loss='mean_squared_error', optimizer='adam')
        #Fit the model
        #print("Fit LSTM with %d lags and %d sequences" %n_lag %n_seq)
        history = model.fit(train_input, train_output, epochs=epoch, batch_size=72, verbose=1, shuffle=False)


        # ### Make the predicion.
        # #### Each columns of the y_predict is a prediction of each feature of my dataset, In this case LSTM predicth both Number of Requests and the response in Byte of of the all requests in the minutes. The forecast are made for the next three minutes.

        # In[ ]:


        #prediction of the model,need test input
        y_predict = model.predict(test_input)
        #print('y_predict_head \n',y_predict[:3])
        print('y_predict_tail \n',y_predict[-3:])
        #print('test_predict_head \n',test_output[:3])
        print('test_predict_tail \n',test_output[-3:])


        # ### Evaluation of RMSE
        # #### First evaluation of the RMSE with the scaled data and after the RMSE compute on the Original Scale

        # In[ ]:


        """ The rmse here is an error valuated on [0 1] """
        # Compute the error ONLY in the "Requests Count" columns. This code change if the feaures of the data change.
        for i in range(1,n_out,2):
	            rmse = sqrt(mean_squared_error(y_predict[:,i], test_output[:,i]))
	            print('Test RMSE t+%d: %.3f'% ((i+1)-i , rmse))


        # In[ ]:


        """Reshape for the evaluation"""
        #Using the test data concatenate it with the forecast and with the train.
        test_input = test_input.reshape(test_input.shape[0],n_obs)
        #test_input.shape  TIP: this result should be equal to the print beafore "Test input data (X):"


        # In[ ]:


        #Rescale the output predicted: 
        #this is an iteration that depend by the initial scaler. The scaler fit the data with
        #a shape with n column. Here this code depens on : n_seq, scaler, y_predict. This return a prediction with the original scale
        test1 = scaler.inverse_transform(y_predict[:,0:2])
        test1
        test2 = scaler.inverse_transform(y_predict[:,2:4])
        test2
        test3 = scaler.inverse_transform(y_predict[:,4:6])
        test3
        inv_output_pred = concatenate((test1,test2,test3), axis=1)
        #inv_output_pred

        #rescale the output of the ground truth: Is the same things for the previous.
        #But here we rescale the data of the ground truth
        test_1 = scaler.inverse_transform(test_output[:,0:2])
        test_1
        test_2 = scaler.inverse_transform(test_output[:,2:4])
        test_2
        test_3 = scaler.inverse_transform(test_output[:,4:6])
        test_3
        inv_output_groun = concatenate((test_1,test_2,test_3), axis=1)
        #inv_output_groun


        # In[ ]:


        # Valutate the rmse using the indices, but i can do with the step two using the n_output as previously.
        rows = ['t+1','t+2','t+3']
        rmse_df = pd.DataFrame(columns=['RMSE'], index=rows)

        for i in range(0,n_seq,1):
            rmse = sqrt(mean_squared_error(inv_output_groun[:,(i+(i+1)):(i+(i+2))],inv_output_pred[:,(i+(i+1)):(i+(i+2))]))
            rmse_df.RMSE[i] = rmse
            print('t+%d RMSE: %f' % ((i+1), rmse))
        rmse_df.to_csv(filename, sep='\t', encoding='utf-8')    


        # In[ ]:





        # ## Save the LSTM and Ground Truth Forecast
        # 
        # ### This part save for each minutes the value of the prediction. So the output that we stored is a timeseries for each timesteps, Example:
        # #### 1998-07-11 22:00 Request_Count(t+1)'  Request_Count(t+2)''  Request_Count(t+3)'''
        # #### 1998-07-11 22:01 Request_Count(t+1)' Request_Count(t+2)''  Request_Count(t+3)'''
        # #### 1998-07-11 22:02 Request_Count(t+1)'  Request_Count(t+2)''  Request_Count(t+3)'''

        # In[ ]:


        idx = series.tail(600) # indexs = to testesampel
        idx = idx.index
        truth_prediction = pd.DataFrame(index=idx)
        lstm_prediction = pd.DataFrame( index=idx)

        for i in range (1,n_out,2):
                truth_prediction['t+%d'%(i)]=inv_output_groun[:,i]
                lstm_prediction['t+%d'%(i)]=inv_output_pred[:,i]
                
        lstm_prediction.to_csv('../Result/lstm_prediction_600sample.csv', sep='\t', encoding='utf-8')
        truth_prediction.to_csv('../Result/test_prediction.csv', sep='\t', encoding='utf-8')


        # #### In the real scenarios the real information that we need for scale our NFV is a time series that present the sum of the values of each rows taken by the exact timesteps for example:
        # #### 1998-07-11 22:03 ( Request_Count(t+1)' + Request_Count(t+2)'' + Request_Count(t+3)''' )
        # #### 1998-07-11 22:06 ( Request_Count(t+1)' + Request_Count(t+2)'' + Request_Count(t+3)''' )
        # #### 1998-07-11 22:09 ( Request_Count(t+1)' + Request_Count(t+2)'' + Request_Count(t+3)''' )
        # 
        # 
        # #### (This is made in the "Cost evaluation" notebook)

        # ## Forecast Graphs
        # ### Fist Graph:  This graph show the last two hours and the last 3 minutes are the prediction, the ground vs LSTM 

        # In[ ]:


        # Buidl a structure from the pred, and the ground that extract only the targhet columns for each timestep (RequestCount)
        Req_count_ground = concatenate((test_1[:,1:],test_2[:,1:],test_3[:,1:]), axis=1)
        Req_count_LSTM = concatenate((test1[:,1:],test2[:,1:],test3[:,1:]), axis=1)

        #Extract the predicted value at the each timestep from each column of the output.
        value_predict = []
        for i in range(n_seq):
            value_predict = numpy.append(value_predict,Req_count_LSTM[-1:,i])
        Rc_series_until_t = list
        Rc_series_until_t = Req_count_ground[:-n_seq,2]
        #Rc_series_until_t
        #rename this
        Rc_ground_forecast  = list
        Rc_ground_forecast = Req_count_ground[-n_seq:,2]
        #Rc_ground_forecast

        #Build a series with data until t and the two forecast(ground, prediction)
        series_predicted=[]
        series_ground=[]

        series_predicted = numpy.append(Rc_series_until_t, value_predict)
        series_ground = numpy.append(Rc_series_until_t, Rc_ground_forecast)

        OX_ticks_str = [dateRef.strftime('%Y-%m-%d %H:%M') for dateRef in tk_last2h]
        OX_ticks_pos = range(len(Req_count_LSTM[-120:,0]))

        pyplot.plot(range(len(series_predicted[-120:])), series_predicted[-120:], color='g', label='Forecast')
        pyplot.plot(range(len(series_ground[-120:])), series_ground[-120:], color='b', label='Ground Truth')

        pyplot.title('Request Count Ground and Forecast', weight='bold',fontsize=20)
        pyplot.xticks(OX_ticks_pos, OX_ticks_str, rotation=90, horizontalalignment='right', fontsize=7)
        pyplot.ylabel("Number of Requests",fontsize=14)
        pyplot.xlabel("Minutes",fontsize=14)
        pyplot.legend(loc='upper left', fancybox=True, fontsize='large', framealpha=0.5)
        pyplot.rcParams['figure.figsize'] = (12,9)
        pyplot.tight_layout()

        pyplot.savefig('plots/forecast_for_paper/forecast_ExpN_%d_Ep%d_Lg%d.png' %(exp_number,epoch,n_lag))
        #pyplot.show()


        # ### Second Graph: The Ground Truth Prediction and the LSTM's Time Steps

        # In[ ]:



        #Load_Series
        Req_count_LSTM = read_csv('../Result/lstm_prediction_600sample.csv', sep='\t',index_col = 'date_time', encoding='utf-8')
        Req_count_ground = read_csv('../Result/test_prediction.csv', sep='\t',index_col = 'date_time', encoding='utf-8')
        Req_count_LSTM.columns=['t_1','t_2','t_3']
        Req_count_ground.columns=['t_1','t_2','t_3']


        # In[ ]:



        tk_last2h #vedi sopra
        OX_ticks_str = [dateRef.strftime('%Y-%m-%d %H:%M') for dateRef in tk_last2h]
        OX_ticks_pos = range(len(Req_count_LSTM.t_1[-120:]))
        pyplot.plot(range(len(Req_count_LSTM.t_1[-120:])), Req_count_LSTM.t_1[-120:], color='g', alpha=1.0,  ms=3, mfc='b', mec='b', label='Forecast_t+1')
        pyplot.plot(range(len(Req_count_LSTM.t_2[-120:])), Req_count_LSTM.t_2[-120:], color='y', alpha=1.0,  ms=3, mfc='b', mec='b', label='Forecast_t+2')
        pyplot.plot(range(len(Req_count_LSTM.t_3[-120:])), Req_count_LSTM.t_3[-120:], color='r', alpha=1.0,  ms=3, mfc='b', mec='b', label='Forecast_t+3')
        pyplot.plot(range(len(Req_count_ground.t_3[-120:])), Req_count_ground.t_3[-120:], color='b', alpha=1.0, mec='b', ls='-', lw=3.0 ,label='Ground Truth Series')

        pyplot.rcParams['figure.figsize'] = (12,9)
        pyplot.title('Request Count Forecast', weight='bold',fontsize=20)
        pyplot.xticks(OX_ticks_pos, OX_ticks_str, rotation=40, horizontalalignment='right', fontsize=7)
        pyplot.ylabel("Number of Requests",fontsize=14)
        pyplot.xlabel("Minutes",fontsize=14)
        pyplot.legend(loc='upper left', fancybox=True, fontsize='large', framealpha=0.5)
        pyplot.tight_layout()

        pyplot.savefig('plots/forecast_for_paper/forecast_all__ExpN_%d_Ep%d_Lg%d.png' %(exp_number,epoch,n_lag))
        #pyplot.show()


        # ## THIS Graphs SHOW HOW THE MODEL FIT THE GROUD TRUTH FOR THE TIME STEPS
        # ### This graphs show how the forecast for single sequence  (t+1,t+2,t+3)  fit the Groud Truth

        # In[ ]:


        tk_last2h #vedi sopra
        OX_ticks_str = [dateRef.strftime('%Y-%m-%d %H:%M') for dateRef in tk_last2h]
        OX_ticks_pos = range(len(Req_count_LSTM.t_1[-120:]))
        pyplot.plot(range(len(Req_count_LSTM.t_1[-120:])), Req_count_LSTM.t_1[-120:], color='g', alpha=1.0,  ms=3, mfc='b', mec='b', label='Forecast_t+1')
        pyplot.plot(range(len(Req_count_ground.t_3[-120:])), Req_count_ground.t_3[-120:], color='b', alpha=1.0, mec='b', ls='-', lw=3.0 ,label='Ground Truth Series')

        pyplot.rcParams['figure.figsize'] = (12,9)
        pyplot.title('Forecast Ground vs t+1', weight='bold',fontsize=20)
        pyplot.xticks(OX_ticks_pos, OX_ticks_str, rotation=40, horizontalalignment='right', fontsize=7)
        pyplot.ylabel("Number of Requests",fontsize=14)
        pyplot.xlabel("Minutes",fontsize=14)
        pyplot.legend(loc='upper left', fancybox=True, fontsize='large', framealpha=0.5)
        pyplot.tight_layout()
        pyplot.savefig('plots/forecast_for_paper/forecastt+1_ExpN_%d_Ep%d_Lg%d.png' %(exp_number,epoch,n_lag))

        #pyplot.show()


        # In[ ]:



        tk_last2h #vedi sopra
        OX_ticks_str = [dateRef.strftime('%Y-%m-%d %H:%M') for dateRef in tk_last2h]
        OX_ticks_pos = range(len(Req_count_LSTM.t_1[-120:]))

        pyplot.plot(range(len(Req_count_LSTM.t_2[-120:])), Req_count_LSTM.t_2[-120:], color='y', alpha=1.0,  ms=3, mfc='b', mec='b', label='Forecast_t+2')

        pyplot.plot(range(len(Req_count_ground.t_3[-120:])), Req_count_ground.t_3[-120:], color='b', alpha=1.0, mec='b', ls='-', lw=3.0 ,label='Ground Truth Series')

        pyplot.rcParams['figure.figsize'] = (12,9)
        pyplot.title('Request Count Forecast', weight='bold',fontsize=20)
        pyplot.xticks(OX_ticks_pos, OX_ticks_str, rotation=40, horizontalalignment='right', fontsize=7)
        pyplot.ylabel("Number of Requests",fontsize=14)
        pyplot.xlabel("Minutes",fontsize=14)
        pyplot.legend(loc='upper left', fancybox=True, fontsize='large', framealpha=0.5)
        pyplot.tight_layout()
        pyplot.savefig('plots/forecast_for_paper/forecast_t+2__ExpN_%d_Ep%d_Lg%d.png' %(exp_number,epoch,n_lag))
        #pyplot.show()


        # In[ ]:



        tk_last2h #vedi sopra
        OX_ticks_str = [dateRef.strftime('%Y-%m-%d %H:%M') for dateRef in tk_last2h]
        OX_ticks_pos = range(len(Req_count_LSTM.t_1[-120:]))
        pyplot.plot(range(len(Req_count_LSTM.t_3[-120:])), Req_count_LSTM.t_3[-120:], color='r', alpha=1.0,  ms=3, mfc='b', mec='b', label='Forecast_t+3')
        pyplot.plot(range(len(Req_count_ground.t_3[-120:])), Req_count_ground.t_3[-120:], color='b', alpha=1.0, mec='b', ls='-', lw=3.0 ,label='Ground Truth Series')

        pyplot.rcParams['figure.figsize'] = (12,9)
        pyplot.title('Request Count Forecast', weight='bold',fontsize=20)
        pyplot.xticks(OX_ticks_pos, OX_ticks_str, rotation=40, horizontalalignment='right', fontsize=7)
        pyplot.ylabel("Number of Requests",fontsize=14)
        pyplot.xlabel("Minutes",fontsize=14)
        pyplot.legend(loc='upper left', fancybox=True, fontsize='large', framealpha=0.5)
        pyplot.tight_layout()
        pyplot.savefig('plots/forecast_for_paper/forecast_t+3_ExpN_%d_Ep%d_Lg%d.png' %(exp_number,epoch,n_lag))
        #pyplot.show()


# # End

# In[ ]:

def main():
#	for l in (3,15,30,60):	
	#                for x in (3,4):
		                exp=4
		                epoc=20
		                lag =2000
		                print("Run Experiment N %d, lag%d, epoch %d" %(exp,lag,epoc))
		                runs_exper(exp,epoc,lag)
		
main()




