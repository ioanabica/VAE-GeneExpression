import math

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from data.data_processing import get_zebrafish_data
from dimensionality_reduction import TSNE_embedding, AE_dim_reduction, PCA_dim_reduction

gene_data, labels = get_zebrafish_data()

num_clusters = 5
n_components = 50


def compute_clustering_scores_gaussian(data, labels, num_clusters, embedding_type):
    if embedding_type is 'TSNE':
        embedding = TSNE_embedding(data, n_components=2)
    else:
        embedding = data

    gaussian = GaussianMixture(n_components=num_clusters).fit(embedding)
    gaussian_labels = gaussian.predict(embedding)
    gaussian_score = adjusted_rand_score(labels, gaussian_labels)
    return gaussian_score


def compute_clustering_scores_kmeans(data, labels, num_clusters, embedding_type):
    if embedding_type is 'TSNE':
        embedding = TSNE_embedding(data, n_components=2)
    else:
        embedding = data
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit_predict(embedding)
    kmeans_score = adjusted_rand_score(labels, kmeans)

    return kmeans_score


def compute_results_with_standard_error(data, labels, num_clusters, embedding_type, clustering_type):
    results = []

    for i in range(50):
        print (i)
        if clustering_type == 'k-means':
            score = compute_clustering_scores_kmeans(data, labels, num_clusters, embedding_type)
        else:
            print "gaussian"
            score = compute_clustering_scores_gaussian(data, labels, num_clusters, embedding_type)
        results.append(score)
    return np.array(results)


def evaluate_clustering_for_all_models(laten_dim, gene_data, labels, embedding_type, clustering_type):
    results = dict()
    results_mean_std = dict()

    InfoVAE_result = AE_dim_reduction(gene_data,
                                      'Saved-Models/Encoders/info_vae_encoder_zebrafish' + str(laten_dim) + '.h5')
    VAE_result = AE_dim_reduction(gene_data, 'Saved-Models/Encoders/vae_encoder_zebrafish' + str(laten_dim) + '.h5')
    AE_result = AE_dim_reduction(gene_data, 'Saved-Models/Encoders/ae_encoder_zebrafish' + str(laten_dim) + '.h5')
    PCA_result = PCA_dim_reduction(gene_data, gene_data, n_components=laten_dim)

    results['VAE'] = compute_results_with_standard_error(VAE_result, labels, num_clusters=5,
                                                         embedding_type=embedding_type, clustering_type=clustering_type)
    results['InfoVAE'] = compute_results_with_standard_error(InfoVAE_result, labels,
                                                             num_clusters=5, embedding_type=embedding_type,
                                                             clustering_type=clustering_type)
    results['AE'] = compute_results_with_standard_error(AE_result, labels,
                                                        num_clusters=5, embedding_type=embedding_type,
                                                        clustering_type=clustering_type)
    results['PCA'] = compute_results_with_standard_error(PCA_result, labels,
                                                         num_clusters=5, embedding_type=embedding_type,
                                                         clustering_type=clustering_type)

    for model in results.keys():
        results_mean_std[model] = [np.mean(results[model]),
                                   np.std(results[model]) / math.sqrt(50.0)]

    return results, results_mean_std
