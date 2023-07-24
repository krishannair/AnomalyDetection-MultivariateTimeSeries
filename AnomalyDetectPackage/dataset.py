import numpy as np
import pandas as pd
import json
import yaml
import matplotlib.pyplot as plt
from configparser import ConfigParser
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

class Data:
    config = ConfigParser()
    config.read("configur.ini")
    path = config['paths']['inputlocation']
    inputDataframe = pd.DataFrame()
    outDataframe = pd.DataFrame()
    rev_label_dict = {}
    train_x = None
    train_y = None
    test_x = None
    test_y = None

    def __init__(self) -> None:
        pass
        
    def dataset_read(self, data_file, file_type):
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
        
    def input_frame(self):
        sensors = self.config['file_names']['sensors'].split(',')
        in_df = pd.DataFrame()
        for txt in sensors:
            read_df = self.dataset_read(self.path+txt, self.config['file_type']['file_type'])
            in_df = in_df.append(read_df)
        in_df = in_df.sort_index().values.reshape(-1,len(sensors),len(in_df.columns)).transpose(0,2,1)
        self.inputDataframe = in_df
    
    def out_frame(self):
        out_df = self.dataset_read(self.path+self.config['file_names']['output'], self.config['file_type']['file_type'])
        out_df.columns = self.config['out_names']['out_names'].split(',')
        self.outDataframe = out_df

    def input_data_2D(self):
        sensors = self.config['file_names']['sensors'].split(',')
        self.inputDataframe = self.inputDataframe.reshape(-1,1,len(sensors)*(self.inputDataframe.shape[1]))
    
    def data_insights(self):
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

    def data_preprocess(self):
        # TODO: Any preprocessing if required for "InputDataFrame"
        ####################
        # LABEL DISTRIBUTION
        component = self.config['req_out']['req_out']
        label = self.outDataframe
        label = label[component] #considering only one of the component
        #print(label)
        # MAPPING LABEL
        d_label, d_reverse_label = {}, {}
        for i,lab in enumerate(label.unique()):
            d_label[lab] = i
            d_reverse_label[i] = lab
        #print(d_label)
        #print(d_reverse_label)
        label = label.map(d_label)
        #print(label)
        self.outDataframe = to_categorical(label)
        #OutDataFrame = label.values
        #print(OutDataFrame)
        self.rev_label_dict = d_reverse_label

    def timeseries_to_trainingdata(self):
        # Convert the InputDataFrame to Training Data with the output in the last column
        df = self.inputDataframe
        y = self.outDataframe

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(df, y, random_state = 42, test_size=0.2)
        scaler = StandardScaler()
        self.train_x = scaler.fit_transform(self.train_x.reshape(-1, self.train_x.shape[-1])).reshape(self.train_x.shape)
        self.test_x = scaler.transform(self.test_x.reshape(-1, self.test_x.shape[-1])).reshape(self.test_x.shape)
    