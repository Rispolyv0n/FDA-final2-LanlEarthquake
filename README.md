# FDA-final2-LanlEarthquake

## Original dataset
https://www.kaggle.com/c/LANL-Earthquake-Prediction/data

## File Description
- `main.py` : run stacking model
- `main_single.py` : run single models
- `singleGridSearch.py` : script for grid searching
- `feature_extraction.py` : script for feature extraction
- `func.py` : script for helper functions
- /model
  - `stacking.py` : class definition of the stacking model
- /data
  - `features-4190.csv` : features of 4190 data
  - `features-test-withId.csv` : features of testing data
  - `features-to-be-deleted.txt` : names of unnecessary features
  - `train.csv` : * original training dataset (not provided here)
  - `/test` : * folder contains all the original testing data (not provided here)
    - `seg_XXXXX.csv`: * all the original testing data (not provided here)
    - ...
  - `/output` : all the submission data
- /plot : all the plots

## Run
In `main.py` and `main_single.py`, there are some parameters:
- `plot_feature_importance` : Set to `True` then it will plot feature importance calculated by each single model.
- `plot_model_correlation` : Set to `True` then it will plot correlations between single models, output images will be in `/plot` directory.
- `read_feature_from_file` : Set to `True` then it won't have to calculate the features from the original dataset again, it only read the pre-calculated features from the file `features-*.csv` in `/data` directory. If `False`, then original dataset is required. Original dataset should be put in the `/data` directory.
- `remove_bad_feature` : Set to `True` then it would drop features with low importance(not recommended).

After setting the parameters above, you can run the scripts by following commands:
- `python3 main.py` : run stacking model
- `python3 main_single.py` : run 12 single models, it will output 12 csv files in the `/data/output` directory. The 12 output files are predictions of testing data by each model respectively.
