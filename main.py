import pandas as pd
import dataset
import numpy as np
import model

def main():
    data = dataset.Data()
    data.out_frame()
    data.input_frame()
    data.data_insights()
    #in_df = data.input_data_2D(in_df)
    model_obj = model.Model()
    model1 = model_obj.defining()
    data.data_preprocess()
    #print(out_df.shape)
    train_x, test_x, train_y, test_y = data.timeseries_to_trainingdata()
    #model1 = model_obj.defining2(train_x)
    model_obj.training(model1,train_x,train_y)
    model_obj.validation(test_x, test_y,data.rev_label_dict,model1)
    return 0
if __name__ == "__main__":
    main()  