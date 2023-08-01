# AnomalyDetection-MultivariateTimeSeries
This is a package that detects anomalies in a multivariate time series. It has 2 models included that is 1. A CNN model and 2. A RNN model using LSTM.
The configur.ini file should be in the folder just outside the folder containing the __main__.py file. 
Installing the dependencies using Poetry causes the tensorflow package to not be installed properly 
then you will have to uninstall tensorflow from the virtual environment created by poetry using "poetry uninstall tensorflow" 
and then reinstall tensorflow by activating the virtual environment and running the command "python -m pip install tensorflow=="2.12.0".
