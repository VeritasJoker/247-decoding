# for hyperparameter search
import tensorflow as tf
import functools
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

import numpy as np


from tfsdec_main import arg_parser, load_pickles, prepare_data, get_fold_num, get_fold_data, WeightAverager

from scipy.stats import uniform, randint, loguniform


def pitom(input_shapes = (10,45), n_classes = 50, conv_filters = 160, 
        learn_rate = 0.00025, dropout = 21, reg = 0.003, reg_head = 0.0005):
    '''Define the decoding model.
    '''

    desc = [(conv_filters, 3), ('max', 2), (conv_filters, 2)]

    input_cnn = tf.keras.Input(shape=input_shapes)

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
                kernel_regularizer=tf.keras.regularizers.l2(reg),
                kernel_initializer='glorot_normal')(prev_layer)
            prev_layer = tf.keras.layers.Activation('relu')(prev_layer)
            prev_layer = tf.keras.layers.BatchNormalization()(prev_layer)
            prev_layer = tf.keras.layers.Dropout(dropout)(prev_layer)

    # Add final conv block
    prev_layer = tf.keras.layers.LocallyConnected1D(
        filters=conv_filters,
        kernel_size=2,
        strides=1,
        padding='valid',
        kernel_regularizer=tf.keras.regularizers.l2(reg),
        kernel_initializer='glorot_normal')(prev_layer)
    prev_layer = tf.keras.layers.BatchNormalization()(prev_layer)
    prev_layer = tf.keras.layers.Activation('relu')(prev_layer)

    cnn_features = tf.keras.layers.GlobalMaxPooling1D()(prev_layer)

    output = cnn_features
    if n_classes is not None:
        output = tf.keras.layers.LayerNormalization()(
                tf.keras.layers.Dense(
                    units=n_classes,
                    kernel_regularizer=tf.keras.regularizers.l2(reg_head),
                    activation='tanh')(cnn_features))

    model = tf.keras.Model(inputs=input_cnn, outputs=output)
    model.compile(loss='mse',
            optimizer=tf.keras.optimizers.Adam(lr=learn_rate),
            metrics=[tf.keras.metrics.CosineSimilarity()])

    return model


def fit_model(x_train, y_train, x_dev, y_dev, args):

    x_train = np.concatenate((x_train, x_dev), axis=0) # add dev to train
    y_train = np.concatenate((y_train, y_dev), axis=0) # add dev to train

    # Callbacks
    args.stop_monitor = 'cosine_similarity'
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

    # model_for_call = functools.partial(model_for_fit, x_train = x_train, y_train = y_train, args = args)
    model = KerasRegressor(build_fn=pitom, verbose=0, 
                            epochs = args.epochs)

    # Tuning Parameters
    batch_size = [100, 256, 400]
    learn_rate = [0.00025, 0.001, 0.01]
    dropout = [0.1, 0.21, 0.3]
    reg = [0.003, 0.1]
    
    batch_size = randint(100,600)
    dropout = uniform()
    learn_rate = loguniform(1e-5,1e-2)
    reg = loguniform(1e-4,1e-1)

    # Fixed parameters
    reg_head = [0.0005]
    x_shape = [x_train.shape[1:]]
    y_shape = [y_train.shape[1]]
    conv_filters = [args.conv_filters]

    param_grid = dict(batch_size=batch_size, learn_rate = learn_rate, dropout = dropout, reg = reg,
            reg_head = reg_head, input_shapes = x_shape, n_classes = y_shape, conv_filters = conv_filters)

    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs=1, 
                        cv=5, scoring = 'neg_mean_squared_error')
    grid_result = grid.fit(x_train, y_train, callbacks = callbacks)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def main():

    # Need to downgrade tensorflow estimator and scikit-learn
    # conda install tensorflow-estimator=2.1.0
    # conda install scikit-learn==0.21.2

    args = arg_parser()
    
    # Load data
    signals, stitch_index, label_folds = load_pickles(args)
    # df = pd.DataFrame(label_folds)
    df = prepare_data(label_folds, args)  # prune

    k = get_fold_num(df)
    for i in range(k):
        print(f'Running fold {i}')
        # tf.keras.backend.clear_session()

        # Extract data from just this fold
        data, w2i, meta = get_fold_data(i, df, stitch_index, signals, args)
        x_train, x_dev, x_test = data[0:3]  # signals
        w_train, w_dev, w_test = data[3:6]  # words
        y_train, y_dev, y_test = data[6:9]  # labels (indices)
        z_train, z_dev, z_test = data[9:12]  # embeddings
        index2word = {j: word for word, j in w2i.items()}
        args.n_classes = len(w2i)

        # Train
        fit_model(x_train, z_train, x_dev, z_dev, args)

if __name__ == '__main__':
    main()