import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import TensorBoard
import sklearn.metrics as sk
import pandas_datareader.data as web
import sklearn.model_selection as ms
import sklearn.preprocessing as pr
from functools import reduce
from collections import deque
import matplotlib.dates as mdates


#References: https://github.com/XianhaiC/Recession-Prediction-LSTM
#https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
#https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
#https://towardsdatascience.com/understanding-rnn-and-lstm-f7cdf6dfc14e
#https://towardsdatascience.com/recurrent-neural-networks-for-recession-forecast-f435a2a4f3ef
#https://blog.usejournal.com/stock-market-prediction-by-recurrent-neural-network-on-lstm-model-56de700bff68
#https://medium.com/analytics-vidhya/stock-market-prediction-using-python-article-4-the-next-recession-923185a2736f

class LSTM_neural_network:

    def __init__(self, d_timesteps, d_horizon, n_dropout, n_features):
        self.d_timesteps = d_timesteps
        self.d_horizon = d_horizon
        self.n_dropout = n_dropout
        self.n_features = n_features


    def preprocessing_coincident_indicators(self):
        print('Processing...')

        # Retrieve Data
        start = '1967-01-01'
        end = '2020-06-01'

        # Retrieve the NBER recession indicators from FRED
        self.rec = web.DataReader('USREC', 'fred', start=start, end=end)

        # Get the leading indciators from FRED
        self.data = web.DataReader(['INDPRO', 'W875RX1', 'CMRMTSPL', 'PAYEMS'], 'fred', start=start, end=end).astype(float)
        self.data.columns = ['IP', 'PILTP', 'MTS', 'ENAP']

        self.main_df = pd.concat([self.data, self.rec], axis=1)
        self.main_df['HORIZON'] = self.main_df['USREC'].shift(-self.d_horizon)
        self.main_df = self.main_df.drop('USREC', 1)


        for col in self.main_df.columns:
            if col != 'HORIZON':
                self.main_df[col] = np.log(self.main_df[col]).diff()
                self.main_df.dropna(inplace=True)
                self.main_df[col] = pr.scale(self.main_df[col].values)
        self.main_df.dropna(inplace=True)

        self.data_x = self.main_df[['IP', 'PILTP', 'MTS', 'ENAP']]
        self.data_y = self.main_df[['HORIZON']]

        # modifies the x dataset into the shape (sample, timestep, features)
        self.data_x_mod = np.array(
            [[self.data_x.values[timestep] for timestep in np.arange(case, case + self.d_timesteps)] for case in
             np.arange(len(self.data_x.values) - self.d_timesteps + 1)])

        # modifies the y dataset into the shape (case, output value)
        self.data_y_mod = np.array(
            self.data_y.values[self.d_timesteps - 1: len(self.data_x.values)])

        # the row indices for the recession state dataframe split for training and testing sets
        self.rng_y = self.data_y.index[self.d_timesteps - 1: len(self.data_x.values)]

        return self.data_x_mod, self.data_y_mod, self.rng_y

    def preprocessing_yield_curve(self):
        print('Processing...')

        # Retrieve Data
        start = '1982-01-01'
        end = '2020-06-01'

        # Retrieve the NBER recession indicators from FRED
        self.rec = web.DataReader('USREC', 'fred', start=start, end=end)

        # Get the leading indciators from FRED
        self.data = web.DataReader(['DGS10', 'DGS3MO'], 'fred', start=start, end=end).astype(float).fillna(0)
        self.data = self.data.resample('MS').mean()

        self.main_df = pd.concat([self.data, self.rec], axis=1)
        self.main_df['HORIZON'] = self.main_df['USREC'].shift(-self.d_horizon)
        self.main_df['SLOPE_YC'] = (self.main_df['DGS10'] - self.main_df['DGS3MO'])
        self.main_df = self.main_df.drop(['USREC', 'DGS10', 'DGS3MO'], 1)

        for col in self.main_df.columns:
            if col != 'HORIZON':
                self.main_df[col] = np.log(self.main_df[col]).diff()
                self.main_df.dropna(inplace=True)
                self.main_df[col] = pr.scale(self.main_df[col].values)
        self.main_df.dropna(inplace=True)

        self.data_x = self.main_df[['SLOPE_YC']]
        self.data_y = self.main_df[['HORIZON']]

        # modifies the x dataset into the shape (sample, timestep, features)
        self.data_x_mod = np.array(
            [[self.data_x.values[timestep] for timestep in np.arange(case, case + self.d_timesteps)] for case in
             np.arange(len(self.data_x.values) - self.d_timesteps + 1)])

        # modifies the y dataset into the shape (case, output value)
        self.data_y_mod = np.array(
            self.data_y.values[self.d_timesteps - 1: len(self.data_x.values)])

        # the row indices for the recession state dataframe split for training and testing sets
        self.rng_y = self.data_y.index[self.d_timesteps - 1: len(self.data_x.values)]

        return self.data_x_mod, self.data_y_mod, self.rng_y

    def train_LSTM(self):
        print('Training...')

        self.cv = ms.TimeSeriesSplit(n_splits=2)

        for train, test in self.cv.split(self.data_x_mod, self.data_y_mod):
            x_train, x_test = self.data_x_mod[train], self.data_x_mod[test]
            y_train, y_test = self.data_y_mod[train], self.data_y_mod[test]

        # initiate LSTM
        # two hidden layers are used
        self.model = Sequential()
        self.model.add(LSTM(16, input_shape=(self.d_timesteps, self.n_features), return_sequences=True, activation='relu'))
        self.model.add(Dropout(self.n_dropout))
        self.model.add(LSTM(16, return_sequences=False, activation='relu'))
        self.model.add(Dropout(self.n_dropout))
        self.model.add(Dense(1, activation='sigmoid'))
        #Compile the model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #Fit the model
        self.history = self.model.fit(x_train, y_train, epochs=33, batch_size=64,validation_data=(x_test, y_test))
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def evaluate_LSTM(self):
        print('Evaluating...')

        self.rng_y_train = self.rng_y[:int(self.y_train.shape[0])]
        self.rng_y_test = self.rng_y[int(self.y_train.shape[0]):]

        self.prediction_full_sample = self.model.predict(self.data_x_mod)
        self.prediction_in_sample = self.model.predict(self.x_train)
        self.prediction_out_of_sample = self.model.predict(self.x_test)

        self.df_full_sample_pred = pd.DataFrame(self.prediction_full_sample, index=pd.date_range(end='2020-06-01', periods=self.data_x_mod.shape[0], freq='MS'))
        self.df_out_of_sample_pred = pd.DataFrame(self.prediction_out_of_sample,index=pd.date_range(end='2020-06-01', periods=self.x_test.shape[0],freq='MS'))
        self.df_in_sample_pred = pd.DataFrame(self.prediction_in_sample, index=pd.date_range(end=self.df_out_of_sample_pred.index[0], periods=self.x_train.shape[0], freq='MS'))


        self.df_out_of_sample_pred['TARGET'] = np.where(self.df_out_of_sample_pred[0] > 0.5, 1, 0)

        self.y_hat = self.df_out_of_sample_pred['TARGET'].tolist()

        self.cm = sk.confusion_matrix(self.y_test, self.y_hat)

        self.TN = self.cm[0][0]
        self.FN = self.cm[1][0]
        self.TP = self.cm[1][1]
        self.FP = self.cm[0][1]

        print('TN:',self.TN,'FN:',self.FN,'TP:', self.TP,'FP:', self.FP)
        print('AR: (TP+TN)/(TP+TN+FP+FN) = ', (self.TP+self.TN)/(self.TP+self.TN+self.FN+self.FP))
        print('PR: TP/(TP+FP) = ', self.TP/(self.TP+self.FP))

        self.rmse = sk.mean_squared_error(self.y_test, self.y_hat)
        print('rmse: ',self.rmse)


    def cross_validation_scores(self):
        # Import and initialize the cross-validation iterator
        self.cv = ms.TimeSeriesSplit(n_splits=3)

        self.cvscores = []
        for train, test in self.cv.split(self.data_x_mod, self.data_y_mod):
            x_train, x_test = self.data_x_mod[train], self.data_x_mod[test]
            y_train, y_test = self.data_y_mod[train], self.data_y_mod[test]
            # initiate LSTM
            # two hidden layers are used
            self.model = Sequential()
            self.model.add(
                LSTM(16, input_shape=(self.d_timesteps, self.n_features), return_sequences=True, activation='relu'))
            self.model.add(Dropout(self.n_dropout))
            self.model.add(LSTM(16, return_sequences=False, activation='relu'))
            self.model.add(Dropout(self.n_dropout))
            self.model.add(Dense(1, activation='sigmoid'))
            # Compile the model
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            # Fit the model
            self.history = self.model.fit(x_train, y_train, epochs=33, batch_size=64, verbose=1,
                                          validation_data=(x_test, y_test))
            # Evaluate model
            self.scores = self.model.evaluate(self.data_x_mod[test], self.data_y_mod[test], verbose=0)
            print("%s: %.2f%%" % (self.model.metrics_names[1], self.scores[1] * 100))
            self.cvscores.append(self.scores[1] * 100)

        print("%.2f%% (+/- %.2f%%)" % (np.mean(self.cvscores), np.std(self.cvscores)))
        print('cvscores: ',self.cvscores)


    def visualize_results(self):
        print('Visualizing...')

        self.rec_full_sample = web.DataReader('USREC', 'fred', start=self.df_full_sample_pred.index[0],end='2020-06-01')
        self.rec_out_of_sample = web.DataReader('USREC', 'fred', start=self.df_out_of_sample_pred.index[0],end='2020-06-01')
        self.rec_in_sample = web.DataReader('USREC', 'fred', start=self.df_in_sample_pred.index[0],end=self.df_in_sample_pred.index[-1])

        #Displays the evaluations on the test data
        self.fig, self.ax = plt.subplots(figsize=(8,2.5))
        self.ax.plot(self.df_full_sample_pred, label='Recession probabilities (Full-sample)',c='royalblue', linewidth=1.0)
        self.ax.fill_between(self.rec_full_sample.index, 0, 1, where=self.rec_full_sample['USREC'].values, color='k', alpha=0.1, linewidth=1.0)
        self.ax.set( title='Recession probabilities (Full-sample)', xlabel='data index', ylabel='Probability')

        self.fig, self.ax = plt.subplots(figsize=(8,2.5))
        self.ax.plot(self.df_out_of_sample_pred[0], label='Recession probabilities (Out-of-sample)', c='green', linewidth=1.0)
        self.ax.fill_between(self.rec_out_of_sample.index, 0, 1, where=self.rec_out_of_sample['USREC'].values, color='k', alpha=0.1, linewidth=1.0)
        self.ax.set(title='Recession probabilities (Out-of-sample)', xlabel='data index', ylabel='Probability')

        self.fig, self.ax = plt.subplots(figsize=(8,2.5))
        self.ax.plot(self.df_in_sample_pred, label='Recession probabilities (In-sample)', c='royalblue', linewidth=1.0)
        self.ax.fill_between(self.rec_in_sample.index, 0, 1, where=self.rec_in_sample['USREC'].values, color='k', alpha=0.1, linewidth=1.0)
        self.ax.set(title='Recession probabilities (In-sample)', xlabel='data index', ylabel='Probability')

        self.fig, self.ax = plt.subplots()
        self.ax.plot(self.history.history['loss'])
        self.ax.plot(self.history.history['val_loss'])
        self.ax.plot(self.history.history['accuracy'])
        self.ax.plot(self.history.history['val_accuracy'])

        #Iterate through CV splits
        self.cv = ms.TimeSeriesSplit(n_splits=10)
        self.fig, self.ax = plt.subplots()
        for i, (train, test) in enumerate(self.cv.split(self.data_x_mod, self.data_y_mod)):
        #Plot the training data on each iteration, to see the behavior of the CV
            self.l1 = self.ax.scatter(train, [i] * len(train), c=[plt.cm.coolwarm(.1)], marker='_', lw=6)
            self.l2 = self.ax.scatter(test, [i] * len(test), c=[plt.cm.coolwarm(.9)], marker='_', lw=6)
            self.ax.set(ylim=[10, -1], title='TimeSeriesSplit', xlabel='Years', ylabel='CV iteration')

        plt.show()

    def save_model(self):
        print('Saving!')
        self.model.save('InsertNameHere.h5')

if __name__ == "__main__":
    #LSTM_neural_networks(d_timesteps, d_horizon, n_dropout, n_features)
    #Used the following inputs for coincident indicators:(12 timesteps, forecast horizon(1, 3, 12), 0.2 dropout, 4 features)
    #Used 33 epochs for 1 month horizon to achieve 17.34% validation loss and 16.24% loss,
    #Used 33 epochs for 3 month horizon to achieve 25.46% validation loss and 23.95% loss,
    #Used 22 epochs for 12 month horizon to achieve 32.22% validation loss and 39.24% loss
    #Used the following inputs for yield curve: (24 timesteps, forecast horizon (1, 3, 12), 0.2 dropout, 1 feature)
    #Used 25 epochs for 1 month horizon to achieve 9.9% validation loss and 12.4% loss
    #Used 22 epochs for 3 month horizon to achieve 16.56% validation loss nad 19.23% loss
    #Used 29 eochs for 12 month horizon to achieve 17.22% validation loss and 18.44% loss


    lstm = LSTM_neural_network(12, 1, 0.2, 4)
    lstm.preprocessing_coincident_indicators() #In order to access the yield curve function change "lstm.preprocessing_
    lstm.train_LSTM()                           #coincident_indicators()" to "lstm.preprocessing_yield_curve" and adjust
    lstm.evaluate_LSTM()                        #the __init__ variables as described above.
    lstm.visualize_results()
    lstm.cross_validation_scores()
    #lstm.save_model()

