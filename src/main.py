from xgb_processor import XGBAnalysis


def main():
    print("Start with bookie_xgb_analysis ms")
    pp = XGBAnalysis()
    pp.xgb_fit_and_predict()
    return print("Done with bookie_xgb_analysis ms")


if __name__ == "__main__":
    main()
