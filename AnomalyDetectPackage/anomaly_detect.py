from . import datasets as datasets #Not able to import * here... Data() is becoming unknown
from . import models as models
import sys
class AnomalyDetect:
    def __init__(self) -> None:
        self.data = datasets.Data()
        self.model_obj = models.Model()
        self.model1 = None

    def model_defining(self, choice):
        if choice==1:
            self.data.ready_data_cnn()
            self.model1 = self.model_obj.architecture()
        elif choice==2:
            self.data.ready_data_rnn()
            self.model1 = self.model_obj.architecture2()
        else:
            print("Invalid Choice.")
            sys.exit()

    def model_training(self):
        self.model_obj.fit(self.model1,self.data.train_x,self.data.train_y)


    def model_validaton(self):
        self.model_obj.validate(self.data.test_x, self.data.test_y,self.data.rev_label_dict,self.model1)
