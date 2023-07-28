import AnomalyDetectPackage as ad
def main():
    detect_obj = ad.AnomalyDetect()
    detect_obj.model_defining()
    detect_obj.model_training()
    detect_obj.model_validaton()

if __name__ == "__main__":
    raise SystemExit(main())  