import numpy as np
import pandas as pd
import json
import yaml
import matplotlib.pyplot as plt
from configparser import ConfigParser
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Data:

    def __init__(self) -> None:
        self.config = ConfigParser()
        self.config.read("configur.ini")
        self.path = self.config['paths']['inputlocation']
        self.inputDataframe = pd.DataFrame()
        self.outDataframe = pd.DataFrame()
        self.rev_label_dict = {}
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        
    def __dataset_read(self, data_file, file_type):
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
        
    def __read_source_data(self):
        sensors = self.config['file_names']['sensors'].split(',')
        in_df = pd.DataFrame()
        for txt in sensors:
            read_df = self.__dataset_read(self.path+txt, self.config['file_type']['file_type'])
            in_df = pd.concat([in_df,read_df])
        in_df = in_df.sort_index().values.reshape(-1,len(sensors),len(in_df.columns)).transpose(0,2,1)
        self.inputDataframe = in_df
    
    def __read_labels(self):
        out_df = self.__dataset_read(self.path+self.config['file_names']['output'], self.config['file_type']['file_type'])
        out_df.columns = self.config['out_names']['out_names'].split(',')
        self.outDataframe = out_df
        component = self.config['req_out']['req_out']
        self.outDataframe = self.outDataframe[component]
        #print(self.outDataframe)

    def __convert_input_data_2D(self):
        sensors = self.config['file_names']['sensors'].split(',')
        self.inputDataframe = self.inputDataframe.reshape(-1,len(sensors)*(self.inputDataframe.shape[1]))
        self.outDataframe = np.array(self.outDataframe)
        # print(type(self.outDataframe))
        # print(type(self.inputDataframe))
        # print(self.inputDataframe.shape)
        
        
    def __transform_train_test_rnn(self):
        scalertr=MinMaxScaler(feature_range=(0,1))
        self.train_x=scalertr.fit_transform(self.train_x)
        scalerte=MinMaxScaler(feature_range=(0,1))
        self.test_x=scalerte.fit_transform(self.test_x)
        self.train_x = self.train_x.reshape(-1,1,self.train_x.shape[1])
        self.test_x = self.test_x.reshape(-1,1,self.test_x.shape[1])
        self.test_y = self.__make_categorical(self.test_y)
        self.train_y = self.__make_categorical(self.train_y)
        # print(self.train_x.shape)
        # print(self.test_x.shape)
        # print(type(self.inputDataframe))
        # with open("output.txt", "a") as f:
        #     with np.printoptions(threshold=np.inf):
        #         print("Hello stackoverflow!", file=f)
        #         print(self.train_x, file = f)
        #         print("----------------------------------------------------------", file = f)
        #         print(self.train_y, file=f)

    def __make_trainable_data(self,xarray,yarray,seq): #xarray is x_train or x_test and yarray is y_train or y_test
        nb_samples = xarray.shape[0] - seq

        xarray_reshaped = np.zeros((nb_samples,xarray.shape[1] ))
        yarray_reshaped = np.zeros((nb_samples))
        for i in range(nb_samples):
            y_position = i + seq
            xarray_reshaped[i] = xarray[i:y_position,:]
            yarray_reshaped[i] = yarray[y_position]
        self.inputDataframe = xarray_reshaped
        self.outDataframe = yarray_reshaped
        #print('sample size', nb_samples)
    
    
    def __make_categorical(self, ytrain):
        # LABEL DISTRIBUTION
        label = pd.Series(ytrain)
        main_label = pd.Series(self.outDataframe)
        # MAPPING LABEL
        d_label, d_reverse_label = {}, {}
        for i,lab in enumerate(pd.Series(main_label.unique()).sort_values(ascending = True)):
            d_label[lab] = i
            d_reverse_label[i] = lab
        self.rev_label_dict = d_reverse_label
        label = label.map(d_label)
        ans = to_categorical(label)
        #print(d_label)
        #print(d_reverse_label)
        #print(label)
        return ans

    def __split_train_test(self):
        # Splits dataframes to training and test data
        df = self.inputDataframe
        y = self.outDataframe

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(df, y, random_state = 42, test_size=0.2)
    
    def __data_insights(self):
        sensors = self.config['file_names']['sensors'].split(',')
        plt.figure(figsize=(8,5))
        plt.plot(self.inputDataframe[0])
        plt.title('Original Data')
        plt.ylabel('Value')
        plt.xlabel('Time')
        np.set_printoptions(False)
        plt.legend(sensors)
        plt.show()
        temp = pd.DataFrame(self.inputDataframe[0][:][:])
        print(temp.describe())
     
    def __transform_train_test_cnn(self):
        scaler = StandardScaler()        
        self.train_x = scaler.fit_transform(self.train_x.reshape(-1, self.train_x.shape[-1])).reshape(self.train_x.shape)
        self.test_x = scaler.transform(self.test_x.reshape(-1, self.test_x.shape[-1])).reshape(self.test_x.shape)
        self.train_y = self.__make_categorical(self.train_y)
        self.test_y = self.__make_categorical(self.test_y)
        #print(self.train_x.shape)
        #print(self.train_y.shape)
    
    def ready_data_cnn(self):
        self.__read_labels()
        self.__read_source_data()
        self.__data_insights()
        self.__split_train_test()
        self.__transform_train_test_cnn()

    def ready_data_rnn(self):
        self.__read_labels()
        self.__read_source_data()
        self.__data_insights()
        self.__convert_input_data_2D()
        self.__make_trainable_data(self.inputDataframe, self.outDataframe, 1)
        self.__split_train_test()
        self.__transform_train_test_rnn()