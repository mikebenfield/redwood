import argparse
import pathlib
import sys
from time import perf_counter

import numpy as np
import sklearn.datasets as datasets
from sklearn.metrics import log_loss, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier

def arg_generate(args):
    directory = args.directory

    (X, y) = datasets.make_classification(
        n_samples=50000,
        n_features=args.n_features,
        n_informative=int(args.n_features * 0.8),
        n_redundant=int(args.n_features * 0.1),
        n_classes=args.n_classes,
    )
    X_test = X[:10000]
    X_train = X[10000:]
    y_test = y[:10000]
    y_train = y[10000:]
    y_train = y_train[:, np.newaxis]
    train = np.concatenate([X_train, y_train], axis=1)

    test_path = pathlib.Path(directory, 'data.test')
    train_path = pathlib.Path(directory, 'data.train')
    result_path = pathlib.Path(directory, 'data.target')

    fmt = ['%f']*args.n_features + ['%d']
    np.savetxt(test_path, X_test, delimiter=' ', fmt='%f')
    np.savetxt(result_path, y_test, delimiter=' ', fmt='%d')
    np.savetxt(train_path, train, delimiter=' ', fmt=fmt)

def arg_train_predict(args):
    directory = args.directory
    prediction_file = args.prediction_file

    test_path = pathlib.Path(directory, 'data.test')
    train_path = pathlib.Path(directory, 'data.train')
    result_path = pathlib.Path(directory, 'data.target')

    time_1 = perf_counter()
    train = np.loadtxt(train_path)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    time_2 = perf_counter()
    print('{} seconds to parse training data'.format(time_2 - time_1))

    classifier = ExtraTreesClassifier(
        n_estimators=args.tree_count,
        max_features=args.split_tries,
        min_samples_split=args.min_samples_split,
        n_jobs=args.thread_count
    )
    classifier.fit(X_train, y_train)
    time_3 = perf_counter()
    print('{} seconds to train'.format(time_3 - time_2))

    X_test = np.loadtxt(test_path)
    time_4 = perf_counter()
    print('{} seconds to parse testing data'.format(time_4 - time_3))

    y_pred = classifier.predict_proba(X_test)
    time_5 = perf_counter()
    print('{} seconds to predict'.format(time_5 - time_4))
    np.savetxt(prediction_file, y_pred, delimiter=' ', fmt='%f')


def arg_evaluate(args):
    target_file = args.target
    pred_file = args.prediction

    y_target = np.loadtxt(target_file)
    y_pred = np.loadtxt(pred_file)
    ll = log_loss(y_target, y_pred)
    print('log loss: {}'.format(ll))
    y_pred_a = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_target, y_pred_a)
    print('accuracy: {}'.format(accuracy))

    
def print_usage_and_quit():
    usage = """
"""
    print(usage)
    sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    generate_parser = subparsers.add_parser('generate')
    generate_parser.add_argument('--directory', required=True)
    generate_parser.add_argument('--n_classes', type=int, required=True)
    generate_parser.add_argument('--n_features', type=int, required=True)
    generate_parser.set_defaults(func=arg_generate)

    train_predict_parser = subparsers.add_parser('train_predict')
    train_predict_parser.add_argument('--directory', required=True)
    train_predict_parser.add_argument('--prediction_file', required=True)
    train_predict_parser.add_argument('--tree_count', type=int, required=True)
    train_predict_parser.add_argument('--thread_count', type=int, required=True)
    train_predict_parser.add_argument('--min_samples_split', type=int, required=True)
    train_predict_parser.add_argument('--split_tries', type=int, required=True)
    train_predict_parser.set_defaults(func=arg_train_predict)

    evaluate_parser = subparsers.add_parser('evaluate')
    evaluate_parser.add_argument('--target', required=True)
    evaluate_parser.add_argument('--prediction', required=True)
    evaluate_parser.set_defaults(func=arg_evaluate)


    args = parser.parse_args()
    args.func(args)
