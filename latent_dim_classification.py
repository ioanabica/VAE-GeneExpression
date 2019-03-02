import pickle

import numpy as np
from keras.utils import np_utils
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from autoencoder_models.SimpleAutoEncoder import SimpleAutoEncoder
from autoencoder_models.VAE_models import DiffVAE, DisentangledDiffVAE
from dimensionality_reduction import PCA_dim_reduction
from neural_network_models.neural_network import NeuralNetwork


def write_results_to_file(filename, data):
    with open(filename, 'a+b') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def SVM_classification(data, labels):
    SVM = svm.SVC(kernel='rbf')
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)

    # Identify best hyperparameters
    g_range = 2. ** np.arange(-10, -5, step=1)
    C_range = 2. ** np.arange(-5, 10, step=1)

    parameters = {'gamma': g_range, 'C': C_range}

    grid = GridSearchCV(SVM, parameters, n_jobs=4, cv=kfold)
    grid.fit(data, labels)
    C_best = grid.best_params_['C']
    gamma_best = grid.best_params_['gamma']

    # Perform classification with best hyperparamers
    SVM = svm.SVC(kernel='rbf', C=C_best, gamma=gamma_best)
    cv_results = cross_val_score(SVM, data, labels, scoring='accuracy', cv=kfold)

    return cv_results


def NN_classification(data, labels, latent_dim):
    neural_network = NeuralNetwork(input_size=latent_dim, num_classes=5,
                                   hidden_layers_dim=[512, 1024, 512], batch_size=128,
                                   epochs=300, learning_rate=0.001, dropout_probability=0.0)
    results = []
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    for train_index, test_index in kfold.split(data, labels):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        neural_network.train_nn(X_train, y_train)

        # Hot encoding
        y_test = y_test.astype(int) - 1
        y_test = np_utils.to_categorical(y_test, 5)

        accuracy = neural_network.nn_model.evaluate(X_test, y_test, verbose=1)
        results.append(accuracy)

    return np.array(results)


def evaluate_SVM_classification(gene_data, labels):
    latent_dims = [2, 3, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300]
    results_latent_dim = dict()
    for latent_dim in latent_dims:
        results_latent_dim[latent_dim] = dict()

        DiffVAE_model = DiffVAE(original_dim=gene_data.shape[1], latent_dim=latent_dim, hidden_layers_dim=[512, 256],
                                batch_size=256, epochs=50, learning_rate=0.001)

        DisentangledDiffVAE_model = DisentangledDiffVAE(original_dim=gene_data.shape[1], latent_dim=latent_dim,
                                                        hidden_layers_dim=[512, 256],
                                                        batch_size=256, epochs=50, learning_rate=0.001)

        AE_model = SimpleAutoEncoder(original_dim=gene_data.shape[1], latent_dim=latent_dim,
                                     hidden_layers_dim=[512, 256],
                                     batch_size=256, epochs=50, learning_rate=0.001)

        DiffVAE_model.train_vae(gene_data, 'Saved-Models/Encoders/vae_encoder_zebrafish.h5',
                                'Saved-Models/Decoders/vae_decoder_zebrafish50batchnorm.h5')

        DisentangledDiffVAE_model.train_vae(gene_data, 'Saved-Models/Encoders/info_vae_encoder_zebrafish.h5',
                                            'Saved-Models/Decoders/info_vae_decoder_zebrafish.h5')

        AE_model.train_ae(gene_data, 'Saved-Models/Encoders/ae_encoder_zebrafish.h5',
                          'Saved-Models/Decoders/ae_decoder_zebrafish.h5')

        InfoVAE_model_dim_reduction = DisentangledDiffVAE_model.dimensionality_reduction(gene_data,
                                                                                         'Saved-Models/Encoders/info_vae_encoder_zebrafish.h5')
        VAE_model_dim_reduction = DiffVAE_model.dimensionality_reduction(gene_data,
                                                                         'Saved-Models/Encoders/vae_encoder_zebrafish.h5')
        AE_dim_reduction = AE_model.dimensionality_reduction(gene_data,
                                                             'Saved-Models/Encoders/ae_encoder_zebrafish.h5')
        PCA_result = PCA_dim_reduction(gene_data, gene_data, n_components=latent_dim)

        results_latent_dim[latent_dim]['DiffVAE'] = SVM_classification(VAE_model_dim_reduction, labels)
        results_latent_dim[latent_dim]['DisentangledDiffVAE'] = SVM_classification(InfoVAE_model_dim_reduction, labels)
        results_latent_dim[latent_dim]['SimpleAE'] = SVM_classification(AE_dim_reduction, labels)
        results_latent_dim[latent_dim]['PCA'] = SVM_classification(PCA_result, labels)
        print results_latent_dim

        write_results_to_file('svm-classification.txt', results_latent_dim)


def evaluate_NN_classification(gene_data, labels):
    latent_dims = [2, 3, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300]
    results_latent_dim = dict()
    for latent_dim in latent_dims:
        results_latent_dim[latent_dim] = dict()

        DiffVAE_model = DiffVAE(original_dim=gene_data.shape[1], latent_dim=latent_dim, hidden_layers_dim=[512, 256],
                                batch_size=128, epochs=50, learning_rate=0.001)

        DisentangledDiffVAE_model = DisentangledDiffVAE(original_dim=gene_data.shape[1], latent_dim=latent_dim,
                                                        hidden_layers_dim=[512, 256],
                                                        batch_size=128, epochs=50, learning_rate=0.001)

        AE_model = SimpleAutoEncoder(original_dim=gene_data.shape[1], latent_dim=latent_dim,
                                     hidden_layers_dim=[512, 256],
                                     batch_size=128, epochs=50, learning_rate=0.001)

        DisentangledDiffVAE_model.train_vae(gene_data, 'Saved-Models/Encoders/info_vae_encoder_zebrafish.h5',
                                            'Saved-Models/Decoders/info_vae_decoder_zebrafish.h5')

        DiffVAE_model.train_vae(gene_data, 'Saved-Models/Encoders/vae_encoder_zebrafish.h5',
                                'Saved-Models/Decoders/vae_decoder_zebrafish50batchnorm.h5')

        AE_model.train_ae(gene_data, 'Saved-Models/Encoders/ae_encoder_zebrafish.h5',
                          'Saved-Models/Decoders/ae_decoder_zebrafish.h5')

        InfoVAE_model_dim_reduction = DisentangledDiffVAE_model.dimensionality_reduction(gene_data,
                                                                                         'Saved-Models/Encoders/info_vae_encoder_zebrafish.h5')
        VAE_model_dim_reduction = DiffVAE_model.dimensionality_reduction(gene_data,
                                                                         'Saved-Models/Encoders/vae_encoder_zebrafish.h5')
        AE_dim_reduction = AE_model.dimensionality_reduction(gene_data,
                                                             'Saved-Models/Encoders/ae_encoder_zebrafish.h5')
        PCA_result = PCA_dim_reduction(gene_data, gene_data, n_components=latent_dim)

        results_latent_dim[latent_dim]['DiffVAE'] = NN_classification(VAE_model_dim_reduction, labels, latent_dim)
        results_latent_dim[latent_dim]['DisentangledDiffVAE'] = NN_classification(InfoVAE_model_dim_reduction, labels,
                                                                                  latent_dim)
        results_latent_dim[latent_dim]['SimpleAE'] = NN_classification(AE_dim_reduction, labels, latent_dim)
        results_latent_dim[latent_dim]['PCA'] = NN_classification(PCA_result, labels, latent_dim)
        print (results_latent_dim)

        write_results_to_file('nn-classification.txt', results_latent_dim)
