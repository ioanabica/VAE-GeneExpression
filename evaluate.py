from autoencoder_models.SimpleAutoEncoder import SimpleAutoEncoder
from autoencoder_models.VAE_models import DiffVAE, DisentangledDiffVAE
from data.data_processing import get_zebrafish_data
from evaluation.latent_dim_metrics import compute_diff_capacity_latent_dim

gene_data, labels = get_zebrafish_data()
print gene_data.shape[1]

latent_dimension = 50

VAE_model = DiffVAE(original_dim=gene_data.shape[1], latent_dim=latent_dimension, hidden_layers_dim=[512, 256],
                    batch_size=128, epochs=100, learning_rate=0.001)

DisentangledVAE_model = DisentangledDiffVAE(original_dim=gene_data.shape[1], latent_dim=latent_dimension,
                                            hidden_layers_dim=[512, 256],
                                            batch_size=128, epochs=100, learning_rate=0.001)

AE_model = SimpleAutoEncoder(original_dim=gene_data.shape[1], latent_dim=latent_dimension, hidden_layers_dim=[512, 256],
                             batch_size=128, epochs=100, learning_rate=0.001)

print "trainign disentangled"
DisentangledVAE_model.train_vae(gene_data, 'Saved-Models/Encoders/info_vae_encoder_zebrafish50test.h5',
                                'Saved-Models/Decoders/info_vae_decoder_zebrafish50test.h5')

VAE_model.train_vae(gene_data, 'Saved-Models/Encoders/vae_encoder_zebrafish50test.h5',
                    'Saved-Models/Decoders/vae_decoder_zebrafish50test.h5')

AE_model.train_ae(gene_data, 'Saved-Models/Encoders/ae_encoder_zebrafish50test.h5',
                  'Saved-Models/Decoders/ae_decoder_zebrafish50test.h5')

DisentangledVAE_model_dim_reduction = DisentangledVAE_model.dimensionality_reduction(gene_data,
                                                                                     'Saved-Models/Encoders/disentangled_vae_encoder_zebrafish50.h5')
VAE_model_dim_reduction = VAE_model.dimensionality_reduction(gene_data,
                                                             'Saved-Models/Encoders/vae_encoder_zebrafish50.h5')
AE_dim_reduction = AE_model.dimensionality_reduction(gene_data, 'Saved-Models/Encoders/ae_encoder_zebrafish50.h5')

compute_diff_capacity_latent_dim(VAE_model_dim_reduction, labels, model='DiffVAE')
compute_diff_capacity_latent_dim(DisentangledVAE_model_dim_reduction, labels, model='DiffVAEDisentangled')
