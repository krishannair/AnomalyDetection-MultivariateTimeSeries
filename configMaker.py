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

with open("configur.ini", "w") as f:
    config.write(f)