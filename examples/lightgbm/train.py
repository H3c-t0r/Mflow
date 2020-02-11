import argparse

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import lightgbm as lgb
import matplotlib as mpl

import mlflow
import mlflow.lightgbm

mpl.use('Agg')


def parse_args():
    parser = argparse.ArgumentParser(description='LightGBM example')
    parser.add_argument('--colsample-bytree', type=float, default=1.0,
                        help='subsample ratio of columns when constructing each tree (default: 1.0)')
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='subsample ratio of the training instances (default: 1.0)')
    parser.add_argument('--conda-env', type=str, default=None,
                        help='the path to a conda environment yaml file (default: None)')
    return parser.parse_args()


def main():
    # parse command-line arguments
    args = parse_args()

    # prepare train and test data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_set = lgb.Dataset(X_train, label=y_train)

    # enable auto logging
    mlflow.lightgbm.autolog()

    with mlflow.start_run():

        # train model
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'colsample_bytree': args.colsample_bytree,
            'subsample': args.subsample,
            'seed': 42,
        }
        model = lgb.train(params, train_set, valid_sets=[train_set], valid_names=['train'])

        # evaluate model
        y_proba = model.predict(X_test)
        y_pred = y_proba.argmax(axis=1)
        loss = log_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        # log metrics
        mlflow.log_metrics({'log_loss': loss, 'accuracy': acc})


if __name__ == '__main__':
    main()
