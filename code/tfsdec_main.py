import argparse
import glob
import json
import operator
import os
import pickle
import random as python_random
import string
import uuid
from collections import Counter
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from sklearn.decomposition import PCA

from evaluate import (evaluate_embeddings, evaluate_roc, evaluate_topk,
                      get_class_predictions, get_class_predictions_kd)

def set_seed(seed=42):
    """Currently unused."""
    np.random.seed(seed)
    python_random.seed(seed)
    tf.random.set_seed(seed)


def arg_parser():
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument('--lag', type=int, default=None, help='')
    parser.add_argument('--lags', type=int, nargs='+', default=None, help='')
    parser.add_argument('--sig-elec-file', default=None)
    parser.add_argument('--mode',
                        type=str,
                        default='prod',
                        help='[prod]uction or [comp]rehension')
    parser.add_argument('--signal-pickle', type=str, required=True, help='')
    parser.add_argument('--label-pickle', type=str, required=True, help='')
    parser.add_argument('--embedding-pickle', type=str, required=True, help='')
    parser.add_argument('--half-window', type=float, default=312.5, help='')
    parser.add_argument('--pca', type=int, default=None, help='')
    parser.add_argument('--min-word-freq', type=int, default=0)
    parser.add_argument('--align-with', nargs='*', type=str, default=[])
    parser.add_argument('--min-dev-freq', type=int, default=-1)
    parser.add_argument('--min-test-freq', type=int, default=-1)

    # Training args
    parser.add_argument('--classify', action='store_true',
                        help='If true, run classification and not regression')
    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        help='Optimizer learning rate.')
    parser.add_argument('--batch-size',
                        type=int,
                        default=512,
                        help='Integer or None. Number of samples per '
                        'gradient update.')
    parser.add_argument('--epochs',
                        type=int,
                        default=1000,
                        help='Integer. Number of epochs to train the model. '
                        'An epoch is an iteration over the entire x and '
                        'y data provided.')
    parser.add_argument('--patience',
                        type=int,
                        default=150,
                        help='Number of epochs with no improvement after '
                        'which training will be stopped.')
    parser.add_argument('--lm-head',
                        action='store_true',
                        help='NotImplementedError')
    parser.add_argument('--ensemble',
                        action='store_true',
                        help='Use the trained models to create an ensemble. '
                        'No training is performed.')
    parser.add_argument('--n-weight-avg', type=int, default=0)

    # Model definition
    parser.add_argument('--conv-filters',
                        type=int,
                        default=128,
                        help='Number of convolutional filters in the model.')
    parser.add_argument('--reg',
                        type=float,
                        default=0.35,
                        help='Float. L2 regularization factor for '
                        'convolutional layers.')
    parser.add_argument('--reg-head',
                        type=float,
                        default=0,
                        help='Float. L2 regularization factor for dense head.')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.2,
                        help='Float between 0 and 1. Fraction of the input '
                        'units to drop.')

    # Other args
    parser.add_argument('--model',
                        type=str,
                        default='default-out',
                        help='Name of output directory.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--verbose',
                        type=int,
                        default=2,
                        help='0, 1, or 2. Verbosity mode. 0 = silent, '
                        '1 = progress bar, 2 = one line per epoch.')

    args = parser.parse_args()

    if args.lag is None:
        if os.environ.get('SLURM_ARRAY_TASK_ID') is not None:
            idx = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
            assert len(args.lags) > 0
            assert idx <= len(args.lags)

            args.lag = args.lags[idx - 1]
            print(f'Using slurm array lag: {args.lag}')
        elif len(args.lags) == 1:
            args.lag = args.lags[0]
        elif len(args.lags) > 1:
            print('Cannot run more than one lag when not in a job array.')
            exit(1)
        else:
            print('[INFO] - using lag 0')
            args.lag = 0  # default
    elif args.lags is not None:
        args.lag = args.lags[args.lag - 1]
        print(f'Using lag from lags: {args.lag}')

    # Set up save directory
    taskID = os.environ.get('SLURM_ARRAY_TASK_ID')
    jobID = os.environ.get('SLURM_ARRAY_JOB_ID')
    nonce = f'{jobID}-' if jobID is not None else ''
    nonce += f'{taskID}-' if taskID is not None else ''
    nonce += uuid.uuid4().hex[:8]
    nonce = 'ensemble' if args.ensemble else nonce

    save_dir = os.path.join('results', args.model, str(args.lag), nonce)
    os.makedirs(save_dir, exist_ok=True)

    args.save_dir = save_dir
    args.task_id = taskID
    args.job_id = jobID
    print(args)

    with open(os.path.join(save_dir, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)

    return args


def combine_label_embeddings(label_folds, embeddings, **kwargs):

    label_folds = pd.DataFrame([*label_folds.values()][0])

    combined = label_folds.merge(embeddings, how = 'left', on = ['word','adjusted_onset'])

    combined['lemmatized_word'] = combined['word']
    combined['in_glove'] = combined['in_gpt2'] = 1
    combined['word_freq_overall'] = kwargs['min_word_freq'] + 1
    combined['onset'] = combined['adjusted_onset']

    return combined


def load_pickles(args):

    with open(args.label_pickle, 'rb') as fh:
        label_folds = pickle.load(fh)
    
    with open(args.signal_pickle, 'rb') as fh:
        signal_d = pickle.load(fh)
    
    if args.embedding_pickle: # embedding pickle separate from label
        with open(args.embedding_pickle, 'rb') as fh:
            embeddings = pickle.load(fh)
        label_folds = combine_label_embeddings(label_folds, embeddings, min_word_freq = args.min_word_freq)

    print('Signals pickle info')
    for key in signal_d.keys():
        print(f'key: {key}, \t '
              f'type: {type(signal_d[key])}, \t '
              f'len: {len(signal_d[key])}')

    # sigkey = 'full_signal'
    # stitchkey = 'full_stitch_index'
    sigkey = 'binned_signal'
    stitchkey = 'bin_stitch_index'

    assert signal_d[sigkey].shape[0] == signal_d[stitchkey][-1], \
        'Error: Incorrect Stitching'
    assert signal_d[sigkey].shape[1] == len(
        signal_d['electrode_names']), 'Error: Incorrect number of electrodes'

    signals = signal_d[sigkey]
    stitch_index = signal_d[stitchkey]
    stitch_index.insert(0, 0)

    if args.sig_elec_file:
        elecs = pd.DataFrame(zip(signal_d['subject'],
                                 signal_d['electrode_names']),
                             columns=['subject', 'electrode'])
        elecs['id'] = elecs.index.values
        elecs.set_index(['subject', 'electrode'], inplace=True)

        sigelecs = pd.read_csv(args.sig_elec_file)
        sigelecs.set_index(['subject', 'electrode'], inplace=True)

        electrodes = sigelecs.join(elecs).id.values
        electrodes = electrodes[~np.isnan(electrodes)].astype(int)
        signals = signals[..., electrodes]
        print(f'Using subset of electrodes: {electrodes.size}')

    return signals, stitch_index, label_folds


def pitom(args, input_shapes, n_classes):
    '''Define the decoding model.
    input_shapes = (input_shape_cnn, input_shape_emb)
    '''

    desc = [(args.conv_filters, 3), ('max', 2), (args.conv_filters, 2)]

    input_cnn = tf.keras.Input(shape=input_shapes[0])

    prev_layer = input_cnn
    for filters, kernel_size in desc:
        if filters == 'max':
            prev_layer = tf.keras.layers.MaxPooling1D(
                pool_size=kernel_size, strides=None,
                padding='same')(prev_layer)
        else:
            # Add a convolution block
            prev_layer = tf.keras.layers.Conv1D(
                filters,
                kernel_size,
                strides=1,
                padding='valid',
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(args.reg),
                kernel_initializer='glorot_normal')(prev_layer)
            prev_layer = tf.keras.layers.Activation('relu')(prev_layer)
            prev_layer = tf.keras.layers.BatchNormalization()(prev_layer)
            prev_layer = tf.keras.layers.Dropout(args.dropout)(prev_layer)

    # Add final conv block
    prev_layer = tf.keras.layers.LocallyConnected1D(
        filters=args.conv_filters,
        kernel_size=2,
        strides=1,
        padding='valid',
        kernel_regularizer=tf.keras.regularizers.l2(args.reg),
        kernel_initializer='glorot_normal')(prev_layer)
    prev_layer = tf.keras.layers.BatchNormalization()(prev_layer)
    prev_layer = tf.keras.layers.Activation('relu')(prev_layer)

    cnn_features = tf.keras.layers.GlobalMaxPooling1D()(prev_layer)

    output = cnn_features
    if n_classes is not None:
        output = tf.keras.layers.LayerNormalization()(
                tf.keras.layers.Dense(
                    units=n_classes,
                    kernel_regularizer=tf.keras.regularizers.l2(args.reg_head),
                    activation='tanh')(cnn_features))

    model = tf.keras.Model(inputs=input_cnn, outputs=output)
    return model


class WeightAverager(tf.keras.callbacks.Callback):
    """Averages model weights across training trajectory, starting at
    designated epoch."""

    def __init__(self, epoch_count, patience):
        super(WeightAverager, self).__init__()
        self.epoch_count = min(epoch_count, 2 * patience)
        self.weights = []
        self.patience = patience

    def on_train_begin(self, logs=None):
        print('Weight averager over last {} epochs.'.format(self.epoch_count))

    def on_epoch_end(self, epoch, logs=None):
        if len(self.weights) and len(
                self.weights) == self.patience + self.epoch_count / 2:
            self.weights.pop(0)
        self.weights.append(self.model.get_weights())

    def on_train_end(self, logs=None):
        if self.weights:
            # self.best_weights = np.asarray(self.model.get_weights())
            w = 0
            p = 0
            for p, nw in enumerate(self.weights):
                w = (w * p + np.asarray(nw)) / (p + 1)
                if p >= self.epoch_count:
                    break
            self.model.set_weights(w)
            print('Averaged {} weights.'.format(p + 1))

def language_decoder(args):
    """Define language model decoder. Currently unusued."""
    lang_model = transformers.TFBertForMaskedLM.from_pretrained(
        args.model_name, cache_dir='/scratch/gpfs/zzada/cache-tf')
    d_size = lang_model.config.hidden_size
    v_size = lang_model.config.vocab_size

    lang_decoder = lang_model.mlm
    lang_decoder.trainable = False

    inputs = tf.keras.Input((d_size, ))
    x = tf.keras.layers.Reshape((1, d_size))(inputs)
    x = lang_decoder(x)
    x = tf.keras.layers.Reshape((v_size, ))(x)
    # x = Lambda(lambda z: tf.gather(z, vocab_indices, axis=-1))(x)
    x = tf.keras.layers.Activation('softmax')(x)
    lm_decoder = tf.keras.Model(inputs=inputs, outputs=x)
    lm_decoder.summary()
    return lm_decoder


def get_decoder(args):
    if args.lm_head:
        return language_decoder()
    else:
        return tf.keras.layers.Dense(
            args.n_classes,
            kernel_regularizer=tf.keras.regularizers.l2(args.reg_head))


def extract_signal_from_fold(examples, stitch_index, signals, args):

    fs = 16  # 16 for binned, otherwise 512
    shift_fs = int(args.lag / 1000 * fs)
    half_window = int(args.half_window / 1000 * fs)

    skipped = 0
    stitches = np.array(stitch_index)
    x, w, z = [], [], []
    for label in examples:

        bin_index = int(label['onset'] // 32)  # divide by 32 for binned signal
        bin_rank = (stitches <= bin_index).nonzero()[0][-1]
        bin_start, bin_stop = stitch_index[bin_rank], stitch_index[bin_rank + 1]

        left_edge = bin_index - half_window + shift_fs
        right_edge = bin_index + half_window + shift_fs

        if (left_edge < bin_start) or (right_edge > bin_stop):
            skipped += 1
            continue
        else:
            x.append(signals[left_edge:right_edge, :])  # binned
            w.append(label['label'])
            z.append(label['embeddings'])

    if skipped > 0:
        print(f'Skipped {skipped} examples due to boundary conditions')

    x = np.stack(x, axis=0)
    w = np.array(w)
    z = np.array(z)

    return x, w, z


def get_fold_data(k, df, stitch, X, args):

    labels = df.to_dict(orient='records')

    # Get masks
    f_train = [ex for ex in labels if ex[f'fold{k}'] == 'train']
    f_dev = [ex for ex in labels if ex[f'fold{k}'] == 'dev']
    f_test = [ex for ex in labels if ex[f'fold{k}'] == 'test']

    # Get signal
    x_train, w_train, z_train = extract_signal_from_fold(
        f_train, stitch, X, args)
    x_dev, w_dev, z_dev = extract_signal_from_fold(f_dev, stitch, X, args)
    x_test, w_test, z_test = extract_signal_from_fold(f_test, stitch, X, args)

    # Standarize signals based on training
    train_means = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    train_stds = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    x_train = (x_train - train_means) / train_stds
    x_dev = (x_dev - train_means) / train_stds
    x_test = (x_test - train_means) / train_stds

    # filter based on freq
    counter_train = Counter(w_train)
    if args.min_dev_freq > 0:
        class_list = set(
            map(
                lambda x: x[0],
                filter(lambda x: x[1] >= args.min_dev_freq,
                       counter_train.items())))
        mask = np.array([cls in class_list for cls in w_dev],
                        dtype=np.bool)
        x_dev, w_dev, z_dev = x_dev[mask], w_dev[mask], z_dev[mask]

    if args.min_test_freq > 0:
        class_list = set(
            map(
                lambda x: x[0],
                filter(lambda x: x[1] >= args.min_test_freq,
                       counter_train.items())))
        mask = np.array([cls in class_list for cls in w_test],
                        dtype=np.bool)
        x_test, w_test, z_test = x_test[mask], w_test[mask], z_test[mask]

    # Determine indexing
    # word2index = {w: j for j, w in enumerate(sorted(set(w_train.tolist())))}

    allwords = w_train.tolist() + w_dev.tolist() + w_test.tolist()
    word2index = {w: j for j, w in enumerate(sorted(set(allwords)))}
    y_train = np.array([word2index[w] for w in w_train])
    y_dev = np.array([word2index[w] for w in w_dev])
    y_test = np.array([word2index[w] for w in w_test])

    assert x_train.shape[0] == y_train.shape[0] == w_train.shape[
        0] == z_train.shape[0]
    assert x_dev.shape[0] == y_dev.shape[0] == w_dev.shape[0] == z_dev.shape[0]
    assert x_test.shape[0] == y_test.shape[0] == w_test.shape[
        0] == z_test.shape[0]

    results = {}
    results['n_train'] = x_train.shape[0]
    results['n_dev'] = x_dev.shape[0]
    results['n_test'] = x_test.shape[0]
    results['n_classes'] = np.unique(y_train).size
    results['n_classes_dev'] = np.unique(y_dev).size
    results['n_classes_test'] = np.unique(y_test).size
    print(json.dumps(results, indent=2))

    return ((x_train, x_dev, x_test, w_train, w_dev, w_test, y_train, y_dev,
             y_test, z_train, z_dev, z_test), word2index, results)


def load_trained_models(k, args):
    models = []
    prev_dir = os.path.dirname(args.save_dir)
    for fn in glob.glob(f'{prev_dir}/*/model-fold{k}.h5'):
        if os.path.isfile(fn):
            try:
                models.append(tf.keras.models.load_model(fn))
                print(f'Loaded {fn}')
            except Exception as e:
                print(f'Problem loading model: {e}')
    assert len(models) > 0, f'No trained models found: {prev_dir}'
    return models


def train_regression(x_train, y_train, x_dev, y_dev, args):
    '''Train a regression model'''
    model = pitom(args, [x_train.shape[1:]], n_classes=y_train.shape[1])
    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.Adam(lr=args.lr),
                  metrics=[tf.keras.metrics.CosineSimilarity()])

    args.stop_monitor = 'cosine_similarity'
    return train_model(model, x_train, y_train, x_dev, y_dev, args)

def train_classifier(x_train, y_train, x_dev, y_dev, args):
    '''Train a classifier model'''
    model = pitom(args, [x_train.shape[1:]], n_classes=None)
    model = tf.keras.Model(inputs=model.input,
                           outputs=get_decoder(args)(model.output))
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(lr=args.lr),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        ])

    n_classes = args.n_classes
    y_train = tf.keras.utils.to_categorical(y_train, n_classes)
    y_dev = tf.keras.utils.to_categorical(y_dev, n_classes)

    args.stop_monitor = 'val_accuracy'
    return train_model(model, x_train, y_train, x_dev, y_dev, args)


def train_model(model, x_train, y_train, x_dev, y_dev, args):
    '''Train a model'''

    # Save model info
    with open(os.path.join(args.save_dir, 'model-summary.txt'), 'w') as fp:
        with redirect_stdout(fp):
            model.summary()

    # Set up training parameters
    callbacks = []
    if args.patience > 0:
        stopper = tf.keras.callbacks.EarlyStopping(monitor=args.stop_monitor,
                                                   mode='max',
                                                   patience=args.patience,
                                                   restore_best_weights=True,
                                                   verbose=args.verbose)
        callbacks.append(stopper)

    if args.n_weight_avg > 0:
        averager = WeightAverager(args.n_weight_avg, args.patience)
        callbacks.append(averager)

    # Train model and save it
    history = model.fit(x=x_train,
                        y=y_train,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        validation_data=[x_dev, y_dev],
                        callbacks=callbacks,
                        verbose=args.verbose)
    model.save(os.path.join(args.save_dir, f'model-fold{args.fold_index}.h5'))

    # Store final value of each metric
    results = {k: float(v[-1]) for k, v in history.history.items()}
    return model, results


def evaluate_regression(models, w_train, y_train, x_test, y_test, z_test, all_data, w2i,
                        args):
    '''
    z_test are embeddings.
    y_all are all embeddings
    all_data = (w_all, y_all, z_all)
    '''
    n_classes = args.n_classes

    # Evaluate using tensorflow metrics
    results = {}
    if len(models) == 1:
        eval_test = models[0].evaluate(x_test, z_test)
        results.update({
            f'test_{metric}': float(value)
            for metric, value in zip(models[0].metrics_names, eval_test)
        })

    # Compute nearest neighbor performance
    w_all, y_all, z_all = all_data

    embs = []
    preds = np.zeros((len(models), len(x_test), n_classes), dtype=np.float64)
    for j, model in enumerate(models):
        model_preds = model.predict(x_test)
        embs.append(model_preds)
        preds[j] = get_class_predictions(model_preds, z_all, y_all, n_classes)

    # Average all preds
    if len(models) > 1:
        preds = np.average(preds, axis=0)
        embs = np.average(embs, axis=0)
    else:
        preds = preds.squeeze()
        embs = np.array(embs).squeeze()

    # Evaluate embeddding corr
    results.update(
        evaluate_embeddings(z_test,
                            embs,
                            prefix='test_',
                            save_dir=args.save_dir,
                            suffix=f'-fold_{args.fold_index}'))

    # Evaluate ROC-AUC
    index2word = {j: word for word, j in w2i.items()}
    res = evaluate_roc(preds,
                       tf.keras.utils.to_categorical(y_test, n_classes),
                       index2word,
                       Counter(w_train),
                       args.save_dir,
                       prefix='test_nn_',
                       suffix=f'ds_test-fold_{args.fold_index}')
    results.update(res)

    # Evaluate top1
    preds = np.zeros((len(models), len(x_test), n_classes), dtype=np.float64)
    for j, model in enumerate(models):
        model_preds = model.predict(x_test)
        preds[j] = get_class_predictions_kd(model_preds, z_all, y_all,
                                            n_classes)
    preds = np.average(preds, axis=0)  # average all predictions

    res = evaluate_topk(preds,
                        tf.keras.utils.to_categorical(y_test, n_classes),
                        index2word,
                        Counter(y_train),
                        args.save_dir,
                        prefix='test_nn_',
                        suffix=f'ds_test-fold_{args.fold_index}')
    results.update(res)

    return results


def evaluate_classifier(models, w_train, x_test, y_test, w2i, args):

    n_classes = args.n_classes
    w_train_freq = Counter(w_train)
    y_test_1hot = tf.keras.utils.to_categorical(y_test, n_classes)

    # Evaluate using tensorflow metrics
    results = {}
    if len(models) == 1:
        eval_test = models[0].evaluate(x_test, y_test_1hot)
        results.update({
            metric: float(value)
            for metric, value in zip(models[0].metrics_names, eval_test)
        })

    # Get model or ensemble predictions
    if len(models) == 1:
        predictions = models[0].predict(x_test)
    elif len(models) > 1:
        # ensemble the predictions
        predictions = np.zeros((len(models), len(x_test), n_classes))
        for j, model in enumerate(models):
            predictions[j] = model.predict(x_test)
        predictions = np.average(predictions, axis=0)

    assert n_classes == predictions.shape[1]

    index2word = {j: word for word, j in w2i.items()}
    res = evaluate_topk(predictions,
                        y_test_1hot,
                        index2word,
                        w_train_freq,
                        args.save_dir,
                        prefix='test_',
                        suffix=f'ds_test-fold_{args.fold_index}')
    results.update(res)

    res2 = evaluate_roc(predictions,
                        y_test_1hot,
                        index2word,
                        w_train_freq,
                        args.save_dir,
                        prefix='test_',
                        suffix=f'ds_test-fold_{args.fold_index}',
                        title=args.model)
    results.update(res2)

    rr = {k: v for k, v in results.items() if not isinstance(v, pd.DataFrame)}
    print(json.dumps(rr, indent=2))
    return results


def run_pca(df, k=50):
    pca = PCA(n_components=k, svd_solver='auto')

    df_emb = df['embeddings']
    pca_output = pca.fit_transform(df_emb.values.tolist())
    df['embeddings'] = pca_output.tolist()

    return df


def create_folds(df, num_folds, split_str=None):
    """create new columns in the df with the folds labeled
    Args:
        args (namespace): namespace object with input arguments
        df (DataFrame): labels
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    def stratify_split(df, num_folds, split_str=None):
        # Extract only test folds
        if split_str is None:
            skf = KFold(n_splits=num_folds, shuffle=False)  # random_state=0)
        elif split_str == 'stratify':
            skf = StratifiedKFold(n_splits=num_folds, shuffle=False)
        else:
            raise Exception('wrong string')

        folds = [t[1] for t in skf.split(df, df.label)]
        return folds

    folds = stratify_split(df, num_folds, split_str=split_str)

    # Go through each fold, and split
    fold_column_names = [f'fold{i}' for i in range(num_folds)]
    for i, fold_col in enumerate(fold_column_names):
        # Shift the number of folds for this iteration
        # [0 1 2 3 4] -> [1 2 3 4 0] -> [2 3 4 0 1]
        #                       ^ dev fold
        #                         ^ test fold
        #                 | - | <- train folds

        folds_ixs = np.roll(folds, i, axis=0)
        *_, dev_ixs, test_ixs = folds_ixs

        df[fold_col] = 'train'
        df.loc[dev_ixs, fold_col] = 'dev'
        df.loc[test_ixs, fold_col] = 'test'

    return df


def prepare_data(df, args):

    df['label'] = df.lemmatized_word.str.lower()

    # Clean up data
    # df = df[df.onset > 0]
    df = df[~df.label.isin(list(string.punctuation))]

    # Remove nans
    df.dropna(axis=0, subset=['embeddings'], inplace=True)

    # Run PCA
    if args.pca is not None:
        df = run_pca(df, k=args.pca)

    # Filter out criteria
    NONWORDS = {'hm', 'huh', 'mhm', 'mm', 'oh', 'uh', 'uhuh', 'um'}
    common = df.in_glove
    for model in args.align_with:
        common = common & df[f'in_{model}']
    nonword_mask = df.word.str.lower().apply(lambda x: x in NONWORDS)
    freq_mask = df.word_freq_overall >= args.min_word_freq
    df = df[common & freq_mask & ~nonword_mask]
    # df = df[common & freq_mask]  # & ~nonword_mask]

    # Keep production or comprehension
    op = operator.eq if 'prod' in args.mode.lower() else operator.ne
    df = df[op(df.speaker, 'Speaker1')]

    assert df.size > 0, 'No data left after processing criteria'

    # Create folds based on filtered dataset
    # df = create_folds(df.reset_index(drop=True), 5)

    return df


def save_results(fold_results, args):
    # Save all metrics
    results = {}
    for metric in fold_results[0]:
        values = [tr[metric] for tr in fold_results]
        agg = pd.concat if isinstance(values[0], pd.DataFrame) else np.mean
        results[f'avg_{metric}'] = agg(values)

    # Save dataframes
    dfs = {k: df for k, df in results.items() if isinstance(df, pd.DataFrame)}
    merged = pd.concat((dfs['avg_test_nn_rocauc_df'], dfs['avg_test_nn_topk_df']), axis=1)
    merged.to_csv(os.path.join(args.save_dir, 'avg_test_topk_rocaauc_df.csv'))

    # Remove all non-serializable objects
    bads = [k for k, v in results.items() if isinstance(v, pd.DataFrame)]
    for key in bads:
        del results[key]
    results = {k: float(v) for k, v in results.items()}
    print(json.dumps(results, indent=2))

    # Write out everything
    res = fold_results[0]
    bads = [k for k, v in res.items() if isinstance(v, pd.DataFrame)]
    for result in fold_results:
        for key in bads:
            if key in result and result[key] is not None:
                del result[key]
        for k, v in result.items():
            if type(v) != int or type(v) != float:
                result[k] = float(v)

    results['runs'] = fold_results
    results['args'] = vars(args)

    with open(os.path.join(args.save_dir, 'results.json'), 'w') as fp:
        json.dump(results, fp, indent=4)


def get_fold_num(df):
    folds_columns = [column for column in df.columns if 'fold' in column]
    return len(folds_columns)


if __name__ == '__main__':
    args = arg_parser()
    
    # Load data
    signals, stitch_index, label_folds = load_pickles(args)
    # df = pd.DataFrame(label_folds)
    df = prepare_data(label_folds, args)  # prune

    # Run
    histories = []
    fold_results = []

    k = get_fold_num(df)
    for i in range(k):
        print(f'Running fold {i}')
        # tf.keras.backend.clear_session()
        args.fold_index = i

        # Extract data from just this fold
        data, w2i, meta = get_fold_data(i, df, stitch_index, signals, args)
        x_train, x_dev, x_test = data[0:3]  # signals
        w_train, w_dev, w_test = data[3:6]  # words
        y_train, y_dev, y_test = data[6:9]  # labels (indices)
        z_train, z_dev, z_test = data[9:12]  # embeddings
        index2word = {j: word for word, j in w2i.items()}
        args.n_classes = len(w2i)

        models = []
        results = {}
        results['fold'] = i
        results.update(meta)

        # Train
        if not args.classify and not args.ensemble:
            model, res = train_regression(x_train, z_train, x_dev, z_dev, args)
            results.update(res)
            models = [model]
        elif args.classify and not args.ensemble:
            model, res = train_classifier(x_train, y_train, x_dev, y_dev, args)
            models = [model]
        elif args.ensemble:
            models = load_trained_models(i, args)
            results['n_models'] = len(models)

        # Evaluate
        if args.classify:
            res = evaluate_classifier(models, w_train, y_train, x_test, y_test, w2i,
                                      args)
            results.update(res)
        else:
            w_all = np.concatenate((w_train, w_dev, w_test), axis=0)
            y_all = np.concatenate((y_train, y_dev, y_test), axis=0)
            z_all = np.concatenate((z_train, z_dev, z_test), axis=0)
            all_data = (w_all, y_all, z_all)

            res = evaluate_regression(models, w_train, y_train, x_test, y_test, z_test,
                                      all_data, w2i, args)
            results.update(res)

        fold_results.append(results)
        print('ROCAUC', res['test_nn_rocauc_test_w_avg'])

    save_results(fold_results, args)
    print(args.save_dir)
