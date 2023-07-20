import numpy as np
import pandas as pd
import json
import yaml
import matplotlib.pyplot as plt
from configparser import ConfigParser
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *

class Data:
    config = ConfigParser()
    config.read("configur.ini")
    path = config['paths']['inputlocation']

    def __init__(self) -> None:
        pass
        
    def Dataset(self, data_file, file_type):
        if file_type == 'csv' or file_type == 'txt':
            return pd.read_csv(data_file, sep='\t', header=None)
        elif file_type == 'json':
            with open(data_file) as f:
                data = json.load(f)
            return pd.DataFrame(data)
        elif file_type == 'yaml':
            with open(data_file) as f:
                data = yaml.safe_load(f)
            return pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
    def InputFrame(self):
        sensors = self.config['file_names']['sensors'].split(',')
        in_df = pd.DataFrame()
        for txt in sensors:
            read_df = self.Dataset(self.path+txt, self.config['file_type']['file_type'])
            in_df = in_df.append(read_df)
        in_df = in_df.sort_index().values.reshape(-1,len(sensors),len(in_df.columns)).transpose(0,2,1)
        return in_df
    
    def OutFrame(self):
        out_df = self.Dataset(self.path+self.config['file_names']['output'], self.config['file_type']['file_type'])
        out_df.columns = self.config['out_names']['out_names'].split(',')
        return out_df

        
    def DataInsights(self, InputDataframe):
        sensors = self.config['file_names']['sensors'].split(',')
        plt.figure(figsize=(8,5))
        plt.plot(InputDataframe[0])
        plt.title('Original Data')
        plt.ylabel('Value')
        plt.xlabel('Time')
        np.set_printoptions(False)
        plt.legend(sensors)
        plt.show()
        temp = pd.DataFrame(InputDataframe[0][:][:])
        print(temp.describe())

    #def DataPreProcess(self, InputDataframe):
         #Change dimensions according to model
        #sensors = 4 # TODO: get this data from config file
        #df = InputDataframe
        #df = df.sort_index().values.reshape(-1,sensors,len(df.columns))
        #df = df.mean(axis=2).reshape(-1,sensors,1) # TODO: we have to change this to work based on "CentralTendancy"
        #df = df.transpose(0,2,1)
        #ANy other preprocessing
        #return df
    def DataPreProcess(self, OutDataFrame):
        # TODO: Any preprocessing if required for "InputDataFrame"
        ####################
        # LABEL DISTRIBUTION
        component = self.config['req_out']['req_out']
        label = OutDataFrame
        label = label[component] #considering only one of the component
        # MAPPING LABEL
        d_label, d_reverse_label = {}, {}
        for i,lab in enumerate(label.unique()):
            d_label[lab] = i
            d_reverse_label[i] = lab

        label = label.map(d_label)
        OutDataFrame = to_categorical(label)

        return (d_reverse_label,OutDataFrame)

    def timeSeriesToTrainingData(self, InputDataFrame, OutDataFrame):
        # Convert the InputDataFrame to Training Data with the output in the last column
        df = InputDataFrame
        y = OutDataFrame

        X_train, X_test, y_train, y_test = train_test_split(df, y, random_state = 42, test_size=0.2)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        #TODO: return TrainingDataFrame
        return (X_train, X_test, y_train, y_test)
          
    # def timeSeriesToTrainingData(self, InputDataFrame, OutDataFrame):
    #     # Convert the InputDataFrame to Training Data with the output in the last column
    #     label = OutDataFrame
    #     label = label.Cooler #considering only cooler component
    #     mapping = {3: 0, 20: 1, 100: 2}
    #     reverse_map = {0:3, 1:20, 2:100}
    #     label = label.map(mapping)

    #     df = InputDataFrame
    #     y = to_categorical(label)

    #     X_train, X_test, y_train, y_test = train_test_split(df, y, random_state = 42, test_size=0.2)
    #     scaler = StandardScaler()
    #     X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    #     X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    #     #TODO: return TrainingDataFrame
    #     return (X_train, X_test, y_train, y_test, reverse_map)
    