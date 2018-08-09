import argparse
import pathlib
import sys
from time import perf_counter

import numpy as np
import sklearn.datasets as datasets
from sklearn.metrics import log_loss, accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor


def arg_prob_generate(args):
    directory = args.directory

    (X, y) = datasets.make_classification(
        n_samples=args.train_size + args.test_size,
        n_features=args.n_features,
        n_informative=int(args.n_features * 0.8),
        n_redundant=int(args.n_features * 0.1),
        n_classes=args.n_classes, )
    X_test = X[:args.test_size]
    X_train = X[args.test_size:]
    y_test = y[:args.test_size]
    y_train = y[args.test_size:]
    y_train = y_train[:, np.newaxis]
    train = np.concatenate([X_train, y_train], axis=1)

    test_path = pathlib.Path(directory, 'data.test')
    train_path = pathlib.Path(directory, 'data.train')
    result_path = pathlib.Path(directory, 'data.target')

    fmt = ['%f'] * args.n_features + ['%d']
    np.savetxt(test_path, X_test, delimiter=' ', fmt='%f')
    np.savetxt(result_path, y_test, delimiter=' ', fmt='%d')
    np.savetxt(train_path, train, delimiter=' ', fmt=fmt)


def arg_regress_generate(args):
    directory = args.directory

    (X, y) = datasets.make_regression(
        n_samples=args.train_size + args.test_size,
        n_features=args.n_features,
        n_informative=int(args.n_features * 0.8))
    X_test = X[:args.test_size]
    X_train = X[args.test_size:]
    y_test = y[:args.test_size]
    y_train = y[args.test_size:]
    y_train = y_train[:, np.newaxis]
    train = np.concatenate([X_train, y_train], axis=1)

    test_path = pathlib.Path(directory, 'data.test')
    train_path = pathlib.Path(directory, 'data.train')
    result_path = pathlib.Path(directory, 'data.target')

    fmt = ['%f'] * (args.n_features + 1)
    np.savetxt(test_path, X_test, delimiter=' ', fmt='%f')
    np.savetxt(result_path, y_test, delimiter=' ', fmt='%f')
    np.savetxt(train_path, train, delimiter=' ', fmt=fmt)


def train_predict(args, model, prediction_f):
    directory = args.directory
    prediction_file = args.prediction_file

    test_path = pathlib.Path(directory, 'data.test')
    train_path = pathlib.Path(directory, 'data.train')

    time_1 = perf_counter()
    train = np.loadtxt(train_path)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    time_2 = perf_counter()
    print('{} seconds to parse training data'.format(time_2 - time_1))

    model.fit(X_train, y_train)
    time_3 = perf_counter()
    print('{} seconds to train'.format(time_3 - time_2))

    X_test = np.loadtxt(test_path)
    time_4 = perf_counter()
    print('{} seconds to parse testing data'.format(time_4 - time_3))

    y_pred = prediction_f(model, X_test)
    time_5 = perf_counter()
    print('{} seconds to predict'.format(time_5 - time_4))
    np.savetxt(prediction_file, y_pred, delimiter=' ', fmt='%f')


def arg_prob_train_predict(args):
    classifier = ExtraTreesClassifier(
        n_estimators=args.tree_count,
        max_features=args.split_tries,
        min_samples_split=args.min_samples_split,
        n_jobs=args.thread_count)
    train_predict(args, classifier, lambda mod, x: mod.predict_proba(x))


def arg_regress_train_predict(args):
    classifier = ExtraTreesRegressor(
        n_estimators=args.tree_count,
        max_features=args.split_tries,
        min_samples_split=args.min_samples_split,
        n_jobs=args.thread_count)
    train_predict(args, classifier, lambda mod, x: mod.predict(x))


def arg_prob_evaluate(args):
    target_file = args.target
    pred_file = args.prediction

    y_target = np.loadtxt(target_file)
    y_pred = np.loadtxt(pred_file)
    ll = log_loss(y_target, y_pred)
    print('log loss: {}'.format(ll))
    y_pred_a = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_target, y_pred_a)
    print('accuracy: {}'.format(accuracy))


def arg_regress_evaluate(args):
    target_file = args.target
    pred_file = args.prediction

    y_target = np.loadtxt(target_file)
    y_pred = np.loadtxt(pred_file)
    mse = mean_squared_error(y_target, y_pred)
    print('mean squared error : {}'.format(mse))
    mae = mean_absolute_error(y_target, y_pred)
    print('mean absolute error : {}'.format(mae))


def add_train_predict_args(parser):
    parser.add_argument('--directory', required=True)
    parser.add_argument('--prediction_file', required=True)
    parser.add_argument('--tree_count', type=int, required=True)
    parser.add_argument('--thread_count', type=int, required=True)
    parser.add_argument('--min_samples_split', type=int, required=True)
    parser.add_argument('--split_tries', type=int, required=True)


def add_generate_args(parser):
    parser.add_argument('--directory', required=True)
    parser.add_argument(
        '--train_size',
        type=int,
        default=40000,
        help='Number of samples in the training set')
    parser.add_argument(
        '--test_size',
        type=int,
        default=10000,
        help='Number of samples in the test set')
    parser.add_argument('--n_features', type=int, required=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    prob_generate_parser = subparsers.add_parser('prob_generate')
    add_generate_args(prob_generate_parser)
    prob_generate_parser.add_argument('--n_classes', type=int, required=True)
    prob_generate_parser.set_defaults(func=arg_prob_generate)

    regress_generate_parser = subparsers.add_parser('regress_generate')
    add_generate_args(regress_generate_parser)
    regress_generate_parser.set_defaults(func=arg_regress_generate)

    prob_train_predict_parser = subparsers.add_parser('prob_train_predict')
    add_train_predict_args(prob_train_predict_parser)
    prob_train_predict_parser.set_defaults(func=arg_prob_train_predict)

    regress_train_predict_parser = subparsers.add_parser(
        'regress_train_predict')
    add_train_predict_args(regress_train_predict_parser)
    regress_train_predict_parser.set_defaults(func=arg_regress_train_predict)

    prob_evaluate_parser = subparsers.add_parser('prob_evaluate')
    prob_evaluate_parser.add_argument('--target', required=True)
    prob_evaluate_parser.add_argument('--prediction', required=True)
    prob_evaluate_parser.set_defaults(func=arg_prob_evaluate)

    regress_evaluate_parser = subparsers.add_parser('regress_evaluate')
    regress_evaluate_parser.add_argument('--target', required=True)
    regress_evaluate_parser.add_argument('--prediction', required=True)
    regress_evaluate_parser.set_defaults(func=arg_regress_evaluate)

    args = parser.parse_args()
    args.func(args)
