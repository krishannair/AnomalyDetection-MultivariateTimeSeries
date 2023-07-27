from AnomalyDetectPackage.anomaly_detect import *
def main():
    detect_obj = AnomalyDetect()
    detect_obj.model_defining()
    detect_obj.model_training()
    detect_obj.model_validaton()

if __name__ == "__main__":
    raise SystemExit(main())  