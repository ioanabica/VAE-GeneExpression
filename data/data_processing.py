import pandas as pd
from sklearn import preprocessing


def scale_gene_expression_df(data_df):
    return preprocessing.MinMaxScaler().fit_transform(data_df)


def get_zebrafish_data():
    zebrafish_gene_data_df = pd.read_csv("data/Zebrafish/GE_mvg.csv", header=None)
    zebrafish_labels = pd.read_csv("data/Zebrafish/Annotation_File.csv")
    labels = zebrafish_labels['State'].values

    gene_expression_normalized = scale_gene_expression_df(zebrafish_gene_data_df)

    return gene_expression_normalized, labels


def get_zebrafish_hspc():
    # Get gene expression data for HSPCs
    zebrafish_gene_data_df = pd.read_csv("data/Zebrafish/GE_mvg.csv", header=None)
    zebrafish_labels = pd.read_csv("data/Zebrafish/Annotation_File.csv")

    zebrafish_gene_data_df = zebrafish_gene_data_df.loc[zebrafish_labels['State'] == 3]
    gene_expression_normalized = scale_gene_expression_df(zebrafish_gene_data_df)

    return gene_expression_normalized


def get_zebrafish_cell_type(cell_type):
    # Get gene expression data for a particular cell type
    zebrafish_gene_data_df = pd.read_csv("data/Zebrafish/GE_mvg.csv", header=None)
    zebrafish_labels = pd.read_csv("data/Zebrafish/Annotation_File.csv")

    zebrafish_gene_data_df = zebrafish_gene_data_df.loc[zebrafish_labels['Type'] == cell_type]
    gene_expression_normalized = scale_gene_expression_df(zebrafish_gene_data_df)

    return gene_expression_normalized


def get_zebrafish_diff_data():
    # Get gene expression data for the differentiated cells
    zebrafish_gene_data_df = pd.read_csv("data/Zebrafish/GE_mvg.csv", header=None)
    zebrafish_labels = pd.read_csv("data/Zebrafish/Annotation_File.csv")

    zebrafish_gene_data_df = zebrafish_gene_data_df.loc[zebrafish_labels['State'] != 3]
    zebrafish_labels = zebrafish_labels.loc[zebrafish_labels['State'] != 3]

    labels = zebrafish_labels['State'].values

    gene_expression_normalized = scale_gene_expression_df(zebrafish_gene_data_df)

    return gene_expression_normalized, labels


def get_zebrafish_genes():
    zebrafish_genes = pd.read_csv("data/Zebrafish/CV_genes.csv", index_col=False)
    return zebrafish_genes['gene'].values
