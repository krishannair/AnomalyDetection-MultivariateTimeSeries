import pandas as pd
import dataset
import numpy as np
import model

def main():
    data = dataset.Data()
    out_df = data.OutFrame()
    print(out_df.columns)
    in_df = data.InputFrame()
    data.DataInsights(in_df)
    #in_df = data.DataPreProcess(in_df)
    model_obj = model.Model()
    model1 = model_obj.Defining()
    label_dict,out_df = data.DataPreProcess(out_df)
    train_x, test_x, train_y, test_y = data.timeSeriesToTrainingData(in_df, out_df)
    model_obj.Training(model1,train_x,train_y)
    model_obj.validation(test_x, test_y,label_dict,model1)
    #data.plot_confusion_matrix(Confusion_matrix, list(revmap.values()))
    return 0

if __name__ == "__main__":
    main()