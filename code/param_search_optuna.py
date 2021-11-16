import tensorflow as tf
import optuna
import numpy as np
import functools

from tfsdec_main import (arg_parser, load_pickles, prepare_data, get_fold_num, get_fold_data, 
                        WeightAverager, train_classifier, train_regression, load_trained_models,
                        evaluate_regression, evaluate_classifier, save_results)

def objective(trial, df, stitch_index, signals, args):

    args.batch_size = trial.suggest_int("batch_size",50,600)
    args.lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    args.dropout= trial.suggest_uniform("dropout", 0, 0.8)
    args.reg = trial.suggest_float("reg", 1e-5, 1e-1, log=True)
    # print(args.batch_size, args.lr, args.dropout, args.reg)
    # fold_results = []
    roc_aucs = []

    k = get_fold_num(df)
    for i in range(k):
        args.fold_index = i
        print(f'Running fold {i}')
        # tf.keras.backend.clear_session()

        # Extract data from just this fold
        data, w2i, meta = get_fold_data(i, df, stitch_index, signals, args)
        x_train, x_dev, x_test = data[0:3]  # signals
        w_train, w_dev, w_test = data[3:6]  # words
        y_train, y_dev, y_test = data[6:9]  # labels (indices)
        z_train, z_dev, z_test = data[9:12]  # embeddings
        args.n_classes = len(w2i)

        models = []
        # results = {}
        # results['fold'] = i
        # results.update(meta)

        # Train
        if not args.classify and not args.ensemble:
            model, res = train_regression(x_train, z_train, x_dev, z_dev, args)
            # results.update(res)
            models = [model]
        elif args.classify and not args.ensemble:
            model, res = train_classifier(x_train, y_train, x_dev, y_dev, args)
            models = [model]
        elif args.ensemble:
            models = load_trained_models(i, args)
            # results['n_models'] = len(models)

        # Evaluate
        if args.classify:
            res = evaluate_classifier(models, w_train, x_test, y_test, w2i,
                                      args)
            # results.update(res)
        else:
            w_all = np.concatenate((w_train, w_dev, w_test), axis=0)
            y_all = np.concatenate((y_train, y_dev, y_test), axis=0)
            z_all = np.concatenate((z_train, z_dev, z_test), axis=0)
            all_data = (w_all, y_all, z_all)

            res = evaluate_regression(models, w_train, y_train, x_test, y_test, z_test,
                                      all_data, w2i, args)
            # results.update(res)

        print('ROCAUC', res['test_nn_rocauc_test_w_avg'])
        roc_aucs.append(res['test_nn_rocauc_test_w_avg'])
    # save_results(fold_results, args)
    # print(args.save_dir)
    roc_auc = sum(roc_aucs) / len(roc_aucs)
    print(roc_auc)
    return (roc_auc)


def search_param(df, stitch_index, signals, args):

    optimize_wrapper  = functools.partial(objective, df = df, stitch_index = stitch_index, 
                                        signals = signals, args = args)

    study = optuna.create_study(direction = "maximize")
    study.optimize(optimize_wrapper, n_trials = 100)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def main():
    args = arg_parser()
    
    # Load data
    signals, stitch_index, label_folds = load_pickles(args)
    # df = pd.DataFrame(label_folds)
    df = prepare_data(label_folds, args)  # prune

    search_param(df, stitch_index, signals, args)


if __name__ == '__main__':
    main()