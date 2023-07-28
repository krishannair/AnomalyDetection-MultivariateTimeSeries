from configparser import ConfigParser

config = ConfigParser()

config["paths"] = {
    "inputLocation": "C:\\Users\\ASUS\\Downloads\\condition+monitoring+of+hydraulic+systems\\"
}

config["file_names"] = {
    "sensors": "TS1.txt,TS2.txt,TS3.txt,TS4.txt",
    "output": "profile.txt"
}

config["file_type"] = {
    "file_type": "csv"
}


config["out_names"] = {
    "out_names" : "Cooler,Valve,Pump,Accumulator,Flag"
}

config["model_config"] = {
    "loss_fun" : "categorical_crossentropy",
    "act_fun" : "relu",
    "dense_act_fun" : "softmax",
    "time_periods" : "60",
    "num_sensors" : "4",
    "optimizer" : "adam",
    "metrics" : "accuracy",
    "episodes" : "10",
    "batch_size" : "16"
}
config["req_out"] = {
    "req_out" : "Cooler"
}
with open("configur.ini", "w") as f:
    config.write(f)