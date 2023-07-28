import AnomalyDetectPackage as ad
def main():
    detect_obj = ad.AnomalyDetect()
    print("Which model do you want to use:")
    choice = int(input("Enter \"1\" for CNN Model \nEnter \"2\" for RNN Model\n"))
    detect_obj.model_defining(choice)
    detect_obj.model_training()
    detect_obj.model_validaton()

if __name__ == "__main__":
    raise SystemExit(main())  