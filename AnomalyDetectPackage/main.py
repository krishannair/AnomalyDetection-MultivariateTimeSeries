import dataset
import model

def main():
    data = dataset.Data()
    data.read_labels()
    data.read_source_data()
    data.data_insights()
    #in_df = data.input_data_2D(in_df)
    model_obj = model.Model()
    model1 = model_obj.architecture()
    data.data_preprocess()
    #print(out_df.shape)
    data.timeseries_to_trainingdata()
    #model1 = model_obj.defining2(train_x)
    model_obj.fit(model1,data.train_x,data.train_y)
    model_obj.validation(data.test_x, data.test_y,data.rev_label_dict,model1)
    return 0
if __name__ == "__main__":
    main()  