import AnomalyDetectPackage.datasets as datasets #Not able to import * here... Data() is becoming unknown
import AnomalyDetectPackage.models as models
class AnomalyDetect:
    def __init__(self) -> None:
        self.data = datasets.Data()
        self.model_obj = models.Model()
        self.model1 = None

    def model_defining(self):
        self.data.read_labels()
        self.data.read_source_data()
        self.data.data_insights()
        self.data.data_preprocess()
        self.model1 = self.model_obj.architecture()
        self.data.timeseries_to_trainingdata()

    def model_training(self):
        self.model_obj.fit(self.model1,self.data.train_x,self.data.train_y)


    def model_validaton(self):
        self.model_obj.validation(self.data.test_x, self.data.test_y,self.data.rev_label_dict,self.model1)
