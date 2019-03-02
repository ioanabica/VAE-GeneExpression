from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding


def PCA_dim_reduction(train_data, input_data, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(train_data)

    return pca.transform(input_data)


def Spectral_embedding(input_data, n_components):
    embedding = SpectralEmbedding(n_components=n_components, affinity='rbf').fit_transform(input_data)
    return embedding


def TSNE_embedding(input_data, n_components):
    embedding = TSNE(n_components=n_components).fit_transform(input_data)
    return embedding


def AE_dim_reduction(input_data, encoder_filename):
    encoder = load_model(encoder_filename)
    return encoder.predict(input_data)
