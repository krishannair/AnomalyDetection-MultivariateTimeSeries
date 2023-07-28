from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import GlobalAveragePooling1D
from configparser import ConfigParser

class Model:
    def __init__(self) -> None:
        self.config = ConfigParser()
        self.config.read("configur.ini")
        self.loss_function = self.config['model_config']['loss_fun']
        self.activation_function = self.config['model_config']['act_fun']
        self.dense_activation = self.config['model_config']['dense_act_fun']
        self.time_periods = int(self.config['model_config']['time_periods'])
        self.num_sensors = int(len(self.config['file_names']['sensors'].split(',')))
        self.optimizer = self.config['model_config']['optimizer']
        self.dropout_rate = float(self.config['model_config']['dropout_rate'])
        self.out_dimension = int(self.config['model_config']['out_dimension'])
        
        #output dimensionality (no of output filters)
        self.units_rnn = 100
        self.f1 = 100
        self.f2 = 100
        self.f3 = 160
        self.f4 = 160
        self.metrics = self.config['model_config_cnn']['metrics']
        self.kernel_size = int(self.config['model_config_cnn']['kernel_size']) #length of window
        self.pool_size = int(self.config['model_config_cnn']['pool_size'])

    def architecture2(self):
        model = Sequential()
        model.add(LSTM(self.units_rnn, activation = self.activation_function,input_shape=(1, self.num_sensors*self.time_periods)))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.out_dimension, activation=self.dense_activation))
        model.compile(loss=self.loss_function, optimizer=self.optimizer)
        print(model.summary())
        return model

    def architecture(self):
        #model defining
        model_m = Sequential()
        model_m.add(Conv1D(self.f1, self.kernel_size, activation=self.activation_function, input_shape=(self.time_periods, self.num_sensors)))
        model_m.add(Conv1D(self.f2, self.kernel_size, activation=self.activation_function))
        model_m.add(MaxPooling1D(self.pool_size))
        model_m.add(Conv1D(self.f3, self.kernel_size, activation=self.activation_function))
        model_m.add(Conv1D(self.f4, self.kernel_size, activation=self.activation_function))
        model_m.add(GlobalAveragePooling1D(name='G_A_P_1D'))
        model_m.add(Dropout(self.dropout_rate))
        model_m.add(Dense(self.out_dimension, activation=self.dense_activation))
        print(model_m.summary())
        model_m.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=[self.metrics])

        return model_m

        
    def fit(self,model_m, x_train, y_train):
        #batch size and no of epochs are variable (to abstract)
        BATCH_SIZE = int(self.config['model_config']['batch_size'])
        EPOCHS = int(self.config['model_config']['episodes'])

        #training (fitting)
        history = model_m.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, verbose=2)

        #plotting loss functions
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('loss plot')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend(['loss', 'val_loss'])
        plt.show()
    
    def predict(self, x_test, model):
        pred = model.predict(x_test)
        return pred

    def validate(self, x_test, y_test, label_dict, model):
        pred = self.predict(x_test, model)
        pred_test = np.argmax(pred, axis = 1)
        print(classification_report([label_dict[np.argmax(label)] for label in y_test], 
                                    [label_dict[label] for label in pred_test]))

        cnf_matrix = confusion_matrix([label_dict[np.argmax(label)] for label in y_test], 
                                      [label_dict[label] for label in pred_test])
        plt.figure(figsize=(7,7))       
        self.__plot_confusion_matrix(cnf_matrix, list(label_dict.values()))
        plt.show()
        
    def __plot_confusion_matrix(self,cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
        
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize=25)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90, fontsize=15)
        plt.yticks(tick_marks, classes, fontsize=15)

        fmt = '.2f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize = 14)

        plt.ylabel('True label', fontsize=20)
        plt.xlabel('Predicted label', fontsize=20)

