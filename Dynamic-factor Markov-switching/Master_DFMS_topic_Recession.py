import numpy as np
from Modified_statsmodels import KimYoo1995Model
from Modified_statsmodels import sw_ms
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from functools import reduce
import sklearn.metrics as sk


class DynamicFactorMarkovSwitching:
#This is a dynamic-factor Markov-swichting model as implemented by Chauvet (1989) and Kim and Yoo (1995)
#This code imports functions and tools from the provided "Modified_statsmodels" file within the folder.
#This separate file was provided because there is currently no application for a multivariate version of a
#dynamic factor model with regime swichting available in statsmodels. The code within the "Modified_statsmodels"
#file was originally developed by Valera Likhosherstov as part of a Google competition. However, his result was
#not merged into the statsmodels library.

    def __init__(self, k_regimes, k_factors, factor_order, error_order, mle_optimized_params):
        self.k_regimes = k_regimes
        self.k_factors = k_factors
        self.factor_order = factor_order
        self.error_order = error_order
        self.mle_optimized_params = mle_optimized_params

    def preprocessing_coincident_indicators(self):
        print('Processing...')

        self.true = sw_ms
        self.dtype = np.dtype('float')

        # Retrieve in-sample data
        self.start = '1976-01-01'
        self.end = '2006-04-01'
        self.data = web.DataReader(['INDPRO', 'W875RX1', 'CMRMTSPL', 'PAYEMS'], 'fred', start=self.start,
                                               end=self.end).astype(float)
        self.data.columns = ['IP', 'PILTP', 'MTS', 'ENAP']
        self.data = self.data.shift(3).dropna()

        self.IP = np.array(self.data['IP'], dtype=self.dtype)
        self.PILTP = np.array(self.data['PILTP'], dtype=self.dtype)
        self.MTS = np.array(self.data['MTS'], dtype=self.dtype)
        self.ENAP = np.array(self.data['ENAP'], dtype=self.dtype)

        self.yy = np.zeros((360, 4), dtype=self.dtype)
        self.yy[:, 0] = (np.log(self.IP[1:]) - np.log(self.IP[:-1])) * 100
        self.yy[:, 1] = (np.log(self.PILTP[1:]) - np.log(self.PILTP[:-1])) * 100
        self.yy[:, 2] = (np.log(self.MTS[1:]) - np.log(self.MTS[:-1])) * 100
        self.yy[:, 3] = (np.log(self.ENAP[1:]) - np.log(self.ENAP[:-1])) * 100

        self.yy -= self.yy.mean(axis=0)
        self.yy /= self.yy.std(axis=0)
        self.obs = self.yy

    def preprocessing_yield_curve(self):

        self.true = sw_ms
        self.dtype = np.dtype('float')

        # Retrieve Data
        self.start = '1976-01-01'
        self.end = '2006-04-01'

        # Get the economic variable from FRED
        self.econ_data = web.DataReader(['INDPRO'], 'fred', start=self.start,end=self.end).astype(float)
        self.econ_data = self.econ_data.shift(3).dropna()

        #The data for the yield curve parameter was downloaded from the public database of FRED, as those data is not
        #available using Pandas DataReader for the respective time horizon
        self.data_x = pd.read_csv('DGS10.csv', index_col='DATE', header=0, parse_dates=True)
        self.data_y = pd.read_csv('WGS2YR.csv', index_col='DATE', header=0, parse_dates=True)
        self.data_z = pd.read_csv('TB3MS.csv', index_col='DATE', header=0, parse_dates=True)
        self.Data = [self.data_x, self.data_y, self.data_z]
        self.data = reduce(lambda left, right: pd.merge(left, right, on='DATE'), self.Data).iloc[:361]

        self.data['slope'] = ((self.data['DGS10'] - self.data['TB3MS'])).dropna()
        self.data['level'] = ((self.data['DGS10'] + self.data['TB3MS'] + self.data['WGS2YR']) / 3).dropna()
        self.data['curvature'] = (((2 * self.data['WGS2YR']) - (self.data['TB3MS'] + self.data['DGS10']))).dropna()

        def difference(data, interval=1):
            diff = list()
            for i in range(interval, len(data)):
                value = data[i] - data[i - interval]
                diff.append(value)
            return pd.DataFrame(diff)

        self.slope = difference(self.data['slope']).dropna()
        self.curvature = difference(self.data['curvature']).dropna()
        self.level = difference(self.data['level']).dropna()
        self.IP = difference(self.econ_data['INDPRO']).dropna()

        self.yy = np.zeros((360, 4), dtype=self.dtype)
        self.yy[:, 0] = self.slope[0]
        self.yy[:, 1] = self.curvature[0]
        self.yy[:, 2] = self.level[0]
        self.yy[:, 3] = self.IP[0]

        self.yy -= self.yy.mean(axis=0)
        self.yy /= self.yy.std(axis=0)
        self.obs = self.yy

    def preprocessing_prediction(self):
        #This function refers to the out-of-sample prediction. As the standard model.predict() function in statsmodels
        #is not implemented into Markov-swichting models yet, I needed to write my own version of a prediction code.
        self.true = sw_ms
        self.dtype = np.dtype('float')

        # Retrieve out-of-sample data
        self.start_pred = '2006-01-01'
        self.end_pred = '2020-04-01'
        self.data_predict = web.DataReader(['INDPRO', 'W875RX1', 'CMRMTSPL', 'PAYEMS'], 'fred', start=self.start_pred,
                                   end=self.end_pred).astype(float)
        self.data_predict.columns = ['IP', 'PILTP', 'MTS', 'ENAP']
        self.data_predict = self.data_predict.shift(3).dropna()

        self.IP = np.array(self.data_predict['IP'], dtype=self.dtype)
        self.PILTP = np.array(self.data_predict['PILTP'], dtype=self.dtype)
        self.MTS = np.array(self.data_predict['MTS'], dtype=self.dtype)
        self.ENAP = np.array(self.data_predict['ENAP'], dtype=self.dtype)

        self.yy = np.zeros((168, 4), dtype=self.dtype)
        self.yy[:, 0] = (np.log(self.IP[1:]) - np.log(self.IP[:-1])) * 100
        self.yy[:, 1] = (np.log(self.PILTP[1:]) - np.log(self.PILTP[:-1])) * 100
        self.yy[:, 2] = (np.log(self.MTS[1:]) - np.log(self.MTS[:-1])) * 100
        self.yy[:, 3] = (np.log(self.ENAP[1:]) - np.log(self.ENAP[:-1])) * 100

        self.yy -= self.yy.mean(axis=0)
        self.yy /= self.yy.std(axis=0)
        self.obs_predict = self.yy

    def DFMS_model_setup(self):
        #This function imports the KimYoo1995 model from the "Modified statsmodels" file.
        self.model = KimYoo1995Model(self.k_regimes, self.obs, self.k_factors, self.factor_order,
                                     error_order=self.error_order,
                                     loglikelihood_burn=self.true['start'],
                                     enforce_stationarity=False)

        return self.model

    def maximum_likelihood_estimation(self):
        #This function executed the MLE procedure
        print('Estimating...')

        #Set start parameter for MLE
        self.start_params = np.array(self.true['start_params_std_data'], dtype=self.dtype)

        #Initialize MLE
        self.results = self.model.fit(start_params=self.start_params, maxiter=10)

        #Save results to csv for input to estimation_results
        self.temp = np.array(self.results.params)
        self.optimized_parameters = pd.DataFrame(self.temp)
        self.optimized_parameters.to_csv('InsertNameHere.csv')

    def prediction(self):
        self.model_predict = KimYoo1995Model(self.k_regimes, self.obs_predict, self.k_factors, self.factor_order,
                                     error_order=self.error_order,
                                     loglikelihood_burn=self.true['start'],
                                     enforce_stationarity=False)

        return self.model_predict

    def estimation_results(self):

        self.df = pd.read_csv(self.mle_optimized_params)
        self.df.columns = ['Index', 'Estimates']
        self.final = np.array(self.df['Estimates'], dtype=self.dtype)

        self.results = self.model.smooth(self.final)

        self.results_predict = self.model_predict.smooth(self.final)

        #In-sample prediction of smoothed and filtered recession probabiltiies
        self.df_smoothed_regime_probs_in_sample = pd.DataFrame(self.results.smoothed_regime_probs[0].T,
                                                     index=self.data.index[:-1])
        self.df_filtered_regime_probs_in_sample = pd.DataFrame(self.results.filtered_regime_probs[0].T,
                                                     index=self.data.index[:-1])

        # Out-of-sample prediction of smoothed and filtered recession probabiltiies
        self.df_smoothed_regime_probs_out_of_sample = pd.DataFrame(self.results_predict.smoothed_regime_probs[0].T,
                                                     index=self.data_predict.index[:-1])
        self.df_filtered_regime_probs_out_of_sample = pd.DataFrame(self.results_predict.filtered_regime_probs[0].T,
                                                     index=self.data_predict.index[:-1])

    def evaluate_DFMS(self):
        print('Evaluating...')

        #Evaluation out-of-sample data
        self.df_smoothed_regime_probs_out_of_sample['TARGET'] = np.where(self.df_smoothed_regime_probs_out_of_sample[0] > 0.5, 1, 0)
        self.y_hat = self.df_smoothed_regime_probs_out_of_sample['TARGET'].tolist()
        self.rec_out_of_sample = web.DataReader('USREC', 'fred', start=self.start_pred, end=self.end_pred)

        self.cm = sk.confusion_matrix(self.rec_out_of_sample.iloc[4:], self.y_hat)

        self.TN = self.cm[0][0]
        self.FN = self.cm[1][0]
        self.TP = self.cm[1][1]
        self.FP = self.cm[0][1]

        print('TN:',self.TN,'FN:',self.FN,'TP:', self.TP,'FP:', self.FP)
        print('AR: (TP+TN)/(TP+TN+FP+FN) = ', (self.TP+self.TN)/(self.TP+self.TN+self.FN+self.FP))
        print('PR: TP/(TP+FP) = ', self.TP/(self.TP+self.FP))

        self.rmse = sk.mean_squared_error(self.rec_out_of_sample[4:], self.y_hat)
        print('rmse: ',self.rmse)

        #Evaluation in-sample data

        self.df_smoothed_regime_probs_in_sample['TARGET'] = np.where(
            self.df_smoothed_regime_probs_in_sample[0] > 0.5, 1, 0)
        self.y_hat = self.df_smoothed_regime_probs_in_sample['TARGET'].tolist()
        self.rec_in_sample = web.DataReader('USREC', 'fred', start=self.start, end=self.end)

        self.cm = sk.confusion_matrix(self.rec_in_sample.iloc[4:], self.y_hat)

        self.TN = self.cm[0][0]
        self.FN = self.cm[1][0]
        self.TP = self.cm[1][1]
        self.FP = self.cm[0][1]

        print('TN:', self.TN, 'FN:', self.FN, 'TP:', self.TP, 'FP:', self.FP)
        print('AR: (TP+TN)/(TP+TN+FP+FN) = ', (self.TP + self.TN) / (self.TP + self.TN + self.FN + self.FP))
        print('PR: TP/(TP+FP) = ', self.TP / (self.TP + self.FP))

        self.rmse = sk.mean_squared_error(self.rec_in_sample[4:], self.y_hat)
        print('rmse: ', self.rmse)

    def visualize_results(self):
        print('Visualizing...')

        # Retrieve and also plot the NBER recession indicators
        self.rec_in_sample = web.DataReader('USREC', 'fred', start=self.start, end=self.end)
        self.rec_full_sample = web.DataReader('USREC', 'fred', start=self.start, end=self.end_pred)

        #Plot smoothed recession probabilities in-sample
        self.fig, self.ax = plt.subplots(figsize=(8,2.5))
        self.ax.plot(self.df_smoothed_regime_probs_in_sample[0], label='Smoothed recession probabilities',linewidth=1.0)
        self.ax.fill_between(self.rec_in_sample.index, 0, 1, where=self.rec_in_sample['USREC'].values, color='k', alpha=0.1, linewidth=1.0)
        self.ax.set(title='Smoothed recession probabilities (In-sample)', xlabel='Years', ylabel='Probability')

        #Plot filtered recession probabilities in-sample
        self.fig, self.ax = plt.subplots(figsize=(8,2.5))
        self.ax.plot(self.df_filtered_regime_probs_in_sample[0], label='Filtered recession probabilities',linewidth=1.0)
        self.ax.fill_between(self.rec_in_sample.index, 0, 1, where=self.rec_in_sample['USREC'].values, color='k', alpha=0.1)
        self.ax.set(title='Filtered recession probabilities (In-sample)', xlabel='Years', ylabel='Probability')

        # Plot smoothed recession probabilities out-of-sample
        self.fig, self.ax = plt.subplots(figsize=(8,2.5))
        self.ax.plot(self.df_smoothed_regime_probs_out_of_sample[0], label='Smoothed recession probabilities', c='green', linewidth=1.0)
        self.ax.fill_between(self.rec_out_of_sample.index, 0, 1, where=self.rec_out_of_sample['USREC'].values, color='k', alpha=0.1)
        self.ax.set(title='Smoothed recession probabilities (Out-of-sample)', xlabel='Years', ylabel='Probability')

        # Plot filtered recession probabilities out-of-sample
        self.fig, self.ax = plt.subplots(figsize=(8,2.5))
        self.ax.plot(self.df_filtered_regime_probs_out_of_sample[0], label='Filtered recession probabilities', c='green', linewidth=1.0)
        self.ax.fill_between(self.rec_out_of_sample.index, 0, 1, where=self.rec_out_of_sample['USREC'].values, color='k', alpha=0.1)
        self.ax.set(title='Filtered recession probabilities (Out-of-sample)' , xlabel='Years', ylabel='Probability')

        # Plot smoothed recession probabilities full-sample
        self.fig, self.ax = plt.subplots(figsize=(8,2.5))
        self.ax.plot(self.df_smoothed_regime_probs_in_sample[0], label='Smoothed recession probabilities', c='royalblue', linewidth=1.0)
        self.ax.plot(self.df_smoothed_regime_probs_out_of_sample[0], label='Smoothed recession probabilities', c='royalblue', linewidth=1.0)
        self.ax.fill_between(self.rec_full_sample.index, 0, 1, where=self.rec_full_sample['USREC'].values,
                             color='k', alpha=0.1)
        self.ax.set(title='Smoothed recession probabilities (Full-sample)', xlabel='Years', ylabel='Probability')

        # Plot filtered recession probabilities full-sample
        self.fig, self.ax = plt.subplots(figsize=(8,2.5))
        self.ax.plot(self.df_filtered_regime_probs_out_of_sample[0], label='Filtered recession probabilities', c='royalblue', linewidth=1.0)
        self.ax.plot(self.df_filtered_regime_probs_in_sample[0], label='Filtered recession probabilities', c='royalblue', linewidth=1.0)
        self.ax.fill_between(self.rec_full_sample.index, 0, 1, where=self.rec_full_sample['USREC'].values,
                             color='k', alpha=0.1)
        self.ax.set(title='Filtered recession probabilities (Full-sample)', xlabel='Years', ylabel='Probability')

        plt.show()

mle_optimized_params = ['MLE_optimized_params_coincident_indicators.csv',
                        'MLE_optimized_params_yield_curve.csv']


if __name__ == "__main__":
    #This function activates the main code. Please note, that in a first step the model parameters
    #are estimated using the maximum_likelihood_estimation is executed. In order to activate this
    #function, my recommendation is to first comment out all functions that are not necessary,
    #as the iteration process takes some time. The only function that need to be activated are
    #"dfmsm.preprocessing_yield_curve", "dfmsm.DFMS_model_setup" and "dfmsm.maximum_likelihood_estimation".
    #In order to save time, I provided the estimated parameters within the folder. To access them
    #you have to choose between 0 and 1 inside the brackets within "mle_optimized_params" for coincident_indicators
    #and yield curve, respectively. In this current setup the smoothed and filtered recession probabilities using
    #coincident economic variables as input data are calculated.
    dfmsm = DynamicFactorMarkovSwitching(2, 3, 1, 2, mle_optimized_params[0])
    dfmsm.preprocessing_coincident_indicators()
    dfmsm.preprocessing_prediction()
    dfmsm.DFMS_model_setup()
    #dfmsm.maximum_likelihood_estimation()
    dfmsm.prediction()
    dfmsm.estimation_results()
    dfmsm.evaluate_DFMS()
    dfmsm.visualize_results()








