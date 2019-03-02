import numpy as np

from autoencoder_models.VAE_models import DiffVAE, DisentangledDiffVAE
from data.data_processing import get_zebrafish_hspc, get_zebrafish_diff_data, \
    get_zebrafish_cell_type
from neural_network_models.neural_network import NeuralNetwork

gene_data, labels = get_zebrafish_diff_data()
latent_dim = 50

DiffVAE = DiffVAE(original_dim=gene_data.shape[1], latent_dim=latent_dim, hidden_layers_dim=[512, 256],
                  batch_size=128, epochs=100, learning_rate=0.001)

DisentangledDiffVAE = DisentangledDiffVAE(original_dim=gene_data.shape[1], latent_dim=latent_dim,
                                          hidden_layers_dim=[512, 256],
                                          batch_size=128, epochs=100, learning_rate=0.001)

NN_for_cells = NeuralNetwork(input_size=gene_data.shape[1], num_classes=5,
                             hidden_layers_dim=[256, 512], batch_size=128,
                             epochs=70, learning_rate=0.001, dropout_probability=0.2)

NN_for_cells.train_nn(gene_data, labels)
NN_for_cells.restore_neural_network('Saved-Models/NeuralNetworks/simple_nn.h5')


def shift_latent_dimensions(data, num_std, latent_dimensions):
    for latent_dim in latent_dimensions:
        std = np.std([data[:, latent_dim]])
        data[:, latent_dim] = data[:, latent_dim] + float(num_std) * std
    return data


def change_cell_type(cell_data, num_std, latent_dimensions):
    cell_dim_reduction = DisentangledDiffVAE.dimensionality_reduction(cell_data,
                                                                      'Saved-Models/Encoders/disentangleddisentangled_vae_encoder_zebrafish50.h5')
    cell_shifted = shift_latent_dimensions(cell_dim_reduction, num_std, latent_dimensions)
    cell_reconstructed = DisentangledDiffVAE.reconstruction(cell_shifted,
                                                            'Saved-Models/Decoders/disentangled_vae_decoder_zebrafish50.h5')
    NN_predictions = np.argmax(NN_for_cells.nn_model.predict(cell_reconstructed), axis=1)

    num_cells = NN_predictions.shape[0]
    transformed_cells = np.where(NN_predictions == 1)[0].shape[0]
    percentage_transformed_cells = float(transformed_cells) / float(num_cells)

    return transformed_cells, percentage_transformed_cells


hspc_data = get_zebrafish_hspc()
cell_mono_data = get_zebrafish_cell_type('Monocytes')
cell_trombo_data = get_zebrafish_cell_type('Thrombocytes')
cell_erytro_data = get_zebrafish_cell_type('Erythrocytes')

# latent_dimensions encoding differentiation of Neutrophils: [2, 11, 33, 40, 45]
latend_dimensions_to_change = [2, 11, 33, 40, 45]
lambda_std = 1
transformed_hspcs, percentage_transformed_hspc = change_cell_type(hspc_data, num_std=lambda_std,
                                                                  latent_dimensions=latend_dimensions_to_change)
