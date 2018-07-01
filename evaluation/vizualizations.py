from dimensionality_reduction import PCA_dim_reduction, TSNE_embedding, \
    AE_dim_reduction, Spectral_embedding
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def vizualize_tsne_zebrafish(gene_data, labels, n_components, compute_results):
    InfoVAE_result = AE_dim_reduction(gene_data,
                                      'Saved-Models/Encoders/info_vae_encoder_zebrafish' + str(n_components) + '.h5')
    VAE_result = AE_dim_reduction(gene_data, 'Saved-Models/Encoders/vae_encoder_zebrafish' + str(n_components) + '.h5')
    AE_result = AE_dim_reduction(gene_data, 'Saved-Models/Encoders/ae_encoder_zebrafish' + str(n_components) + '.h5')
    PCA_result = PCA_dim_reduction(gene_data, gene_data, n_components=n_components)

    cell_types = ['', 'Monocytes', 'Neutrophils', 'HSPC', 'Erythrocytes', 'Thrombocytes']
    cell_labels = [cell_types[label] for label in labels]

    if compute_results:
        tsne_vae = TSNE_embedding(VAE_result, n_components=2)
        vae_df = pd.DataFrame(
            data={'x-TSNE': tsne_vae[:, 0], 'y-TSNE': tsne_vae[:, 1], 'label': cell_labels, 'Model': 'DiffVAE'})

        tsne_ae = TSNE_embedding(AE_result, n_components=2)
        ae_df = pd.DataFrame(
            data={'x-TSNE': tsne_ae[:, 0], 'y-TSNE': tsne_ae[:, 1], 'label': cell_labels, 'Model': 'SimpleAE'})

        tsne_pca = TSNE_embedding(PCA_result, n_components=2)
        pca_df = pd.DataFrame(
            data={'x-TSNE': tsne_pca[:, 0], 'y-TSNE': tsne_pca[:, 1], 'label': cell_labels, 'Model': 'PCA'})

        tsne_info = TSNE_embedding(InfoVAE_result, n_components=2)
        info_vae_df = pd.DataFrame(
            data={'x-TSNE': tsne_info[:, 0], 'y-TSNE': tsne_info[:, 1], 'label': cell_labels,
                  'Model': 'Disentangled-DiffVAE'})

        result_df = pd.concat([vae_df, info_vae_df, ae_df, pca_df])

        result_df.to_csv('data/plots_data/tsne_results_zebrafish' + str(n_components) + '.csv')

    else:
        result_df = pd.read_csv('data/plots_data/tsne_results_zebrafish' + str(n_components) + '.csv')

    # sns.set('white')
    sns_plot = sns.lmplot(data=result_df, x='x-TSNE', y='y-TSNE', col='Model', hue='label', col_wrap=2,
                          fit_reg=False, size=4, legend_out=True)

    sns_plot.savefig('results/figures/Manifolds/tsne_results_zebrafish' + str(n_components) + '.pdf')


def vizualize_spctral_zebrafish(gene_data, labels, n_components, compute_results):
    InfoVAE_result = AE_dim_reduction(gene_data,
                                      'Saved-Models/Encoders/info_vae_encoder_zebrafish' + str(n_components) + '.h5')
    VAE_result = AE_dim_reduction(gene_data, 'Saved-Models/Encoders/vae_encoder_zebrafish' + str(n_components) + '.h5')
    AE_result = AE_dim_reduction(gene_data, 'Saved-Models/Encoders/ae_encoder_zebrafish' + str(n_components) + '.h5')
    PCA_result = PCA_dim_reduction(gene_data, gene_data, n_components=n_components)

    cell_types = ['', 'Monocytes', 'Neutrophils', 'HSPC', 'Erythrocytes', 'Thrombocytes']
    cell_labels = [cell_types[label] for label in labels]

    if compute_results:
        tsne_vae = Spectral_embedding(VAE_result, n_components=2)
        vae_df = pd.DataFrame(
            data={'x-SpectralEmbedding': tsne_vae[:, 0], 'y-SpectralEmbedding': tsne_vae[:, 1], 'label': cell_labels,
                  'Model': 'DiffVAE'})

        tsne_ae = Spectral_embedding(AE_result, n_components=2)
        ae_df = pd.DataFrame(
            data={'x-SpectralEmbedding': tsne_ae[:, 0], 'y-SpectralEmbedding': tsne_ae[:, 1], 'label': cell_labels,
                  'Model': 'SimpleAE'})

        tsne_pca = Spectral_embedding(PCA_result, n_components=2)
        pca_df = pd.DataFrame(
            data={'x-SpectralEmbedding': tsne_pca[:, 0], 'y-SpectralEmbedding': tsne_pca[:, 1], 'label': cell_labels,
                  'Model': 'PCA'})

        tsne_info = Spectral_embedding(InfoVAE_result, n_components=2)
        info_vae_df = pd.DataFrame(
            data={'x-SpectralEmbedding': tsne_info[:, 0], 'y-SpectralEmbedding': tsne_info[:, 1], 'label': cell_labels,
                  'Model': 'Disentangled-DiffVAE'})

        result_df = pd.concat([vae_df, info_vae_df, ae_df, pca_df])

        result_df.to_csv('data/plots_data/spectral_results_zebrafishRBF' + str(n_components) + '.csv')

    else:
        result_df = pd.read_csv('data/plots_data/spectral_results_zebrafish' + str(n_components) + '.csv')

    sns_plot = sns.lmplot(data=result_df, x='x-SpectralEmbedding', y='y-SpectralEmbedding', col='Model', hue='label',
                          col_wrap=2,
                          fit_reg=False, size=4, legend_out=True)

    sns_plot.savefig('results/figures/Manifolds/spectral_results_zebrafishRBF' + str(n_components) + '.pdf')


def plot_latent_dimensions(latent_dimension_1, latent_dimension_2, data, labels, model):
    cell_types = ['', 'Monocytes', 'Neutrophils', 'HSPC', 'Erythrocytes', 'Thrombocytes']
    cell_labels = [cell_types[label] for label in labels]

    sns.set(font_scale=1.3)
    flatui = ["#869495", "#209250", "#8B50A3", "#2E88C5", "#CF4436", "#34495e"]
    sns.set_palette(sns.color_palette(flatui))
    sns.set_style('white')

    data_df = pd.DataFrame(
        data={'Latent dimension ' + str(latent_dimension_1): data[:, latent_dimension_1],
              'Latent dimension ' + str(latent_dimension_2): data[:, latent_dimension_2],
              'Cell type': cell_labels, 'Model': model})
    sns_plot = sns.lmplot(data=data_df, x='Latent dimension ' + str(latent_dimension_1),
                          y='Latent dimension ' + str(latent_dimension_2), col='Model', hue='Cell type',
                          fit_reg=False, size=4, legend_out=True)
    sns_plot.savefig('results/figures/' + model + str(latent_dimension_1) + 'vs' + str(latent_dimension_2) + '.pdf')


def pairplot_latent_dimensions(data, labels, model, dataset):
    data_dict = {}
    for i in range(20, 30):
        data_dict[i] = data[:, i]
    data_dict['label'] = labels

    data_df = pd.DataFrame(data=data_dict)
    sns_plot = sns.pairplot(data=data_df, hue='label', size=4)
    sns_plot.savefig('figures/Pairplots/pairplot2030' + model + dataset + '.pdf')
