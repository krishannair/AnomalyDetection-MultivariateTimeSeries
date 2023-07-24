import AnomalyDetectPackage.wrapper as wp
def main():
    wrapper_obj = wp.Wrapper()
    wrapper_obj.model_defining()
    wrapper_obj.model_training()
    wrapper_obj.model_validaton()

if __name__ == "__main__":
    main()  