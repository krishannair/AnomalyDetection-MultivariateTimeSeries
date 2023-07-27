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

    def architecture2(self, train_x):
        model = Sequential()
        model.add(LSTM(100, input_shape=(train_x.shape[1], train_x.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss=self.config['model_config']['loss_fun'], optimizer=self.config['model_config']['optimizer'])
        print(model.summary())
        return model

    def architecture(self):
        #TO BE abstracted 
        #start
        LOSS_FUNCTION = self.config['model_config']['loss_fun']
        ACTIVATION_FUNCTION = self.config['model_config']['act_fun']
        DENSE_ACTIVATION = self.config['model_config']['dense_act_fun']
        TIME_PERIODS = int(self.config['model_config']['time_periods'])
        num_sensors = int(self.config['model_config']['num_sensors'])
        OPTIMIZER = self.config['model_config']['optimizer']
        METRICS = self.config['model_config']['metrics']
        
        #output dimensionality (no of output filters)
        f1 = 100
        f2 = 100
        f3 = 160
        f4 = 160
        kernel_size = 6 #length of window
        pool_size = 3
        dropout_rate = 0.5
        out_dimension = 3
        #end

        #model defining
        model_m = Sequential()
        model_m.add(Conv1D(f1, kernel_size, activation=ACTIVATION_FUNCTION, input_shape=(TIME_PERIODS, num_sensors)))
        model_m.add(Conv1D(f2, kernel_size, activation=ACTIVATION_FUNCTION))
        model_m.add(MaxPooling1D(pool_size))
        model_m.add(Conv1D(f3, kernel_size, activation=ACTIVATION_FUNCTION))
        model_m.add(Conv1D(f4, kernel_size, activation=ACTIVATION_FUNCTION))
        model_m.add(GlobalAveragePooling1D(name='G_A_P_1D'))
        model_m.add(Dropout(dropout_rate))
        model_m.add(Dense(out_dimension, activation=DENSE_ACTIVATION))
        print(model_m.summary())
        model_m.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=[METRICS])

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
    
    def prediction(self, x_test, model):
        pred = model.predict(x_test)
        return pred

    def validation(self, x_test, y_test, label_dict, model):
        pred = self.prediction(x_test, model)
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
        #plt.colorbar()
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

