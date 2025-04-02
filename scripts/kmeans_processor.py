import pandas as pd
import matplotlib.pyplot as plt
import clustering_models
from pandasgui import show as gui_show
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler,StandardScaler


def cluster(
        df: pd.DataFrame,
        clustering_model = None,
        transform = StandardScaler(),
        clusters: int = 4
) -> pd.DataFrame:
    """
    Apply the provided clustering model to the data.
    Built to handle month-by-month data, rather than the entire dataset.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cluster_model: An instantiated scikit-learn clustering model.
        transform: A scikit-learn transformer for standardizing the data.
        clusters (int): The number of clusters to use.

    Returns:
        pd.DataFrame: The input DataFrame with an additional 'cluster' column.
    """

# Ensure we don't modify the original DataFrame
    result = df.copy()

    # Ensure 'date' and 'ticker' are not part of the clustering features
    clean_df = df.loc[:, ~df.columns.isin(['dollar_volume', 'date', 'ticker'])]
    # Standardize the data
    clean_df = transform.fit_transform(clean_df)

    # Instantiate clustering model if not provided
    # Similar to providing a default argument
    if clustering_model is None:
        model = clustering_models.simple_kmeans(clusters=clusters)
    else:
        # If the clustering_model is callable, call it with clusters
        try:
            model = clustering_model(clusters=clusters)
        except TypeError:
            # Otherwise assume it is already instantiated and reset its n_clusters
            model = clustering_model
            model.set_params(n_clusters=clusters)

    # Fit model and assign cluster labels
    result['cluster'] = model.fit_predict(clean_df)
    return result


def plot_clusters(
        df: pd.DataFrame,
        dimensions: int = 2,
) -> None:
    
    """
    Use Principle Component Analysis (PCA) to better visualize the clustering
    """

    pca = PCA(n_components=dimensions)
    # Perform PCA on the features
    pca_components = pca.fit_transform(df.drop(columns='cluster',axis=1))
    # Create a DataFrame with the PCA components and cluster labels
    pca_df = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])
    # Add the clusters back to the PCA DataFrame
    pca_df['cluster'] = df['cluster'].values

    for cluster in pca_df['cluster'].unique():
        cluster_data = pca_df[pca_df['cluster'] == cluster]
        plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster}')

    plt.legend(); plt.show(); return


def pipeline_cluster(
        df: pd.DataFrame,
        clustering_model = None,
        transform=StandardScaler(),
        clusters: int = 4
) -> pd.DataFrame:
    """
    Apply the clustering model to the DataFrame grouped by 'date'.
    If clustering model, the default from cluster() is used, which is clustering_models.simple_kmeans.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cluster_model: An instantiated scikit-learn clustering model.
        transform: A scikit-learn transformer for standardizing the data.
        clusters (int): The number of clusters to use.

    Returns:
        pd.DataFrame: The clustered DataFrame with 'cluster' labels.
    """
    # Apply the clustering model to each group (grouped by 'date')
    data = df.dropna().groupby('date').apply(
        lambda group: cluster(df=group, clustering_model=clustering_model, transform=transform, clusters=clusters)
    )

    # Reset the index to remove duplicated 'date' column
    data.index.names = ['delete', 'date', 'ticker']
    data = data.reset_index(drop=False)
    data.drop('delete', axis=1, inplace=True)
    data = data.set_index(['date', 'ticker'])

    return data


def main():
    pass


if __name__ == '__main__':
    main()