from . import dataset
from . import model
class Wrapper:
    data = dataset.Data()
    model_obj = model.Model()
    model1 = None
    def __init__(self) -> None:
        pass
    def model_defining(self):
        self.data.out_frame()
        self.data.input_frame()
        self.data.data_insights()
        self.model1 = self.model_obj.defining()
        self.data.timeseries_to_trainingdata()

    def model_training(self):
        self.model_obj.training(self.model1,self.data.train_x,self.data.train_y)


    def model_validaton(self):
        self.model_obj.validation(self.data.test_x, self.data.test_y,self.data.rev_label_dict,self.model1)
