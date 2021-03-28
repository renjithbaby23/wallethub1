"""
Created by: Renjith Baby
            renjithbaby23@gmail.com

Evaluation script for walletHub assignment.

Usage:
    $ python evaluate.py --input <path_to_input_file.csv>
    Prerequisites:
        1. Use python 3.8.3 and install the requirements given in the requirements.txt
        2. The input csv file is expected to have the exact same format as that of the
            validation.csv file given in the same repo
"""

import argparse
import logging
import pathlib
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

np.warnings.filterwarnings("ignore")


class Validator(object):
    """
    Validator class to handle the entire evaluation process
    """

    def __init__(self, validation_file_path: str, model_path: str = "model.pkl"):
        self.file = pathlib.Path(validation_file_path)
        assert (
            self.file.is_file()
        ), f"{self.file} is not a valid csv file! Please provide a valid csv path."
        self.model_path = pathlib.Path(model_path)
        self.target = None
        self.predictions = None
        self._prepare_data()
        self._load_model()
        self._predict()

    def _prepare_data(self):
        self._data = pd.read_csv(self.file)
        self.target = self._data["y"].values.ravel()
        self._data.drop(columns=["y"], inplace=True)

    def _load_model(self):
        try:
            self._model = joblib.load(self.model_path)
            logging.info("Loaded the model...")
        except Exception as err:
            logging.error("Failed to load the model %s", self.model_path)
            logging.error("Error: %s", err)
            raise err

    def _predict(self):
        start = time.time()
        self.predictions = self._model.predict(self._data.values).astype(int)
        print(f"Model prediction time: {time.time() - start:.2f} seconds")
        logging.info("Model predictions completed...")

    def _accuracy(
        self, target: np.array, predictions: np.array, thresh: float
    ) -> (float, float):
        """
        Calculates accuracy using the following criteria:
            if the predicted value is within +-thresh, then the prediction is
            considered as correct and else incorrect
        """
        rmse = mean_squared_error(target, predictions, squared=False)
        diff = np.abs(target - predictions) <= thresh
        acc = np.round((diff.sum() / diff.shape[0]) * 100, 2)
        return rmse, acc

    def get_accuracy(self, thresh: float = 3.0) -> (float, float):
        rmse, accuracy = self._accuracy(self.target, self.predictions, thresh)
        return rmse, accuracy


class CustomTransform1(BaseEstimator, TransformerMixin):
    """
    1. Remove x067, x094, x095, x096 - constant columns
    2. Remove highly correlated binary variables - x246, x247, x261, x262, x270, x271, x282, x284, x300
    Since using hard coded index, this transformation must be used at first
    3. Mapping of categorical varibles as mentioned above
    """

    def __init__(self):
        pass

    def func_map(x, thresh):
        "Mapping function for custom transformation"
        if x < thresh:
            return x
        else:
            return thresh

    vfunc_map = np.vectorize(func_map)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X_ = X.copy()
        list_1 = [67, 76, 77, 78, 155, 151]
        list_2 = [36, 48, 49, 50, 51, 52, 106]
        list_3 = [
            22,
            154,
            251,
            37,
            38,
            45,
            46,
            47,
            53,
            54,
            60,
            68,
            79,
            99,
            100,
            101,
            107,
            111,
            121,
            122,
            148,
            162,
            168,
            173,
            174,
            175,
            176,
            177,
            178,
            181,
            182,
            196,
            227,
            228,
            229,
            240,
            250,
            253,
        ]
        list_4 = [21, 147, 161, 286, 301]
        list_5 = [18]
        list_8 = [17]

        X_[:, list_1] = self.vfunc_map(X_[:, list_1], 1)
        X_[:, list_2] = self.vfunc_map(X_[:, list_2], 2)
        X_[:, list_3] = self.vfunc_map(X_[:, list_3], 3)
        X_[:, list_4] = self.vfunc_map(X_[:, list_4], 4)
        X_[:, list_5] = self.vfunc_map(X_[:, list_5], 5)
        X_[:, list_8] = self.vfunc_map(X_[:, list_8], 8)

        X_ = np.delete(
            X_, [66, 93, 94, 95, 245, 246, 260, 261, 269, 270, 281, 283, 299], 1
        )
        return X_


class CustomTransform2(BaseEstimator, TransformerMixin):
    """
    Custom transformation for imputing
    """

    def __init__(self):
        self.imputer = SimpleImputer(verbose=1, strategy="most_frequent")
        # self.imputer = IterativeImputer(max_iter=10, random_state=42, verbose=1)

    def fit(self, X, y=None):
        X_ = X.copy()
        self.imputer.fit(X_)
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        imputed = self.imputer.transform(X_)
        return imputed


if __name__ == "__main__":
    logging.basicConfig(filename="logs.log", level=logging.DEBUG)
    parser = argparse.ArgumentParser(
        description="Evaluation of the walletHub model on holdout validation set"
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="path to the dataset",
        default="./validation.csv",
    )
    args = parser.parse_args()
    try:
        validator = Validator(args.input)
    except Exception as err:
        logging.error("Failed to create validator class...")
        logging.error("Error %s", err)
        raise err
    rmse, accuracy = validator.get_accuracy(3)
    print("*" * 30)
    print(f"RMSE\t\t: {np.round(rmse,2)}")
    print(f"Accuracy\t: {accuracy}%")
    print("*" * 30)
