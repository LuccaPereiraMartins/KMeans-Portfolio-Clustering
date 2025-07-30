import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler,StandardScaler

import clustering_models


def cluster(
        df: pd.DataFrame,
        clustering_model=None,
        transform=RobustScaler(),
        clusters: int = 4
) -> pd.DataFrame:
    """
    Apply the provided clustering model to the data for a single time period.
    Args:
        df (pd.DataFrame): The input DataFrame.
        clustering_model: An instantiated scikit-learn clustering model or callable.
        transform: A scikit-learn transformer for standardizing the data.
        clusters (int): The number of clusters to use.
    Returns:
        pd.DataFrame: The input DataFrame with an additional 'cluster' column.
    """
    result = df.copy()
    features = df.loc[:, ~df.columns.isin(['dollar_volume', 'date', 'ticker'])]
    features = transform.fit_transform(features)
    if clustering_model is None:
        model = clustering_models.simple_kmeans(clusters=clusters)
    else:
        try:
            model = clustering_model(clusters=clusters)
        except TypeError:
            model = clustering_model
            if hasattr(model, 'set_params'):
                model.set_params(n_clusters=clusters)
    result['cluster'] = model.fit_predict(features)
    return result


def pipeline_cluster(
        df: pd.DataFrame,
        clustering_model=None,
        transform=RobustScaler(),
        clusters: int = 4
) -> pd.DataFrame:
    """
    Apply the clustering model to the DataFrame grouped by 'date'.
    Args:
        df (pd.DataFrame): The input DataFrame.
        clustering_model: An instantiated scikit-learn clustering model or callable.
        transform: A scikit-learn transformer for standardizing the data.
        clusters (int): The number of clusters to use.
    Returns:
        pd.DataFrame: The clustered DataFrame with 'cluster' labels.
    """
    data = df.dropna().groupby('date').apply(
        lambda group: cluster(df=group, clustering_model=clustering_model, transform=transform, clusters=clusters)
    )
    # Reset the index to remove duplicated 'date' column
    data.index = data.index.droplevel(0)
    data = data.reset_index(drop=False).set_index(['date', 'ticker'])
    return data


def pca_clustering(
        df: pd.DataFrame,
        dimensions: int = 2,
        ax=None,
        title: str = None
) -> list:
    """
    Use PCA to project the data into 2D space and plot clusters on the provided axis.
    
    Args:
        df (pd.DataFrame): DataFrame with clustering results (must contain 'cluster').
        dimensions (int): Number of PCA components (default 2).
        ax: A matplotlib axis on which to plot.
        title (str): Title for the subplot (displayed as the time period).
    
    Returns:
        list: The scatter plot handles for the clusters (for the common legend).

    NOTE:
        Points in cluster X in one date do not necessarily correspond to the same points in cluster X in another date.
        The importance is to see cluster grouping and counts, not the actual points.
    """
    # Use PCA to reduce features to 'dimensions'
    pca = PCA(n_components=dimensions)
    # Drop the 'cluster' column from features and then combine it back after transform
    features = df.drop(columns='cluster', axis=1)
    pca_components = pca.fit_transform(features)
    # Combine the PCA-transformed features with the cluster labels
    pca_df = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])
    pca_df['cluster'] = df['cluster'].values

    handles = []
    for clust in sorted(pca_df['cluster'].unique()):
        cluster_data = pca_df[pca_df['cluster'] == clust]
        sc = ax.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {clust}')
        handles.append(sc)
    if title:
        ax.set_title(title)
    return handles


def plot_pca_clustering(
        df: pd.DataFrame = None,
        time_periods: list = None,
        dimensions: int = 2,
):
    if df is None:
        # Load the pre-clustered data (assumed to contain a 'date' column and 'cluster' column)
        df = pd.read_csv('processed_data/clustered_data.csv')
    # Define fixed time periods for visualization
    if time_periods is None:
        time_periods = ['2023-02', '2023-07', '2024-02', '2024-07']
    
    # Assert the dataframe has the required columns
    assert 'cluster' in df.columns, "DataFrame must contain 'cluster' column."
    assert 'date' in df.columns, "DataFrame must contain 'date' column."

    # Assert the time periods are in the DataFrame and there are 4 unique periods in time_periods
    assert all(period in df['date'].values for period in time_periods), "Some time periods are not present in the DataFrame."
    assert len(set(time_periods)) == 4, "4 unique time periods are required."

    # Create a 2 x 2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    common_handles = None
    for i, period in enumerate(time_periods):
        # Filter the DataFrame by the given period
        period_df = df[df['date'] == period].copy()
        # Drop extra columns that are not used for PCA (if needed)
        period_df = period_df.drop(columns=['date', 'ticker'], errors='ignore')
        # Plot onto the corresponding subplot
        handles = pca_clustering(period_df, dimensions=dimensions, ax=axes[i], title=period)
        if common_handles is None:
            common_handles = handles
        
    # Create one common legend for all subplots
    fig.legend(common_handles, [h.get_label() for h in common_handles], loc='upper right')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


def tsne_clustering(
        df: pd.DataFrame,
        perplexity: float = 30.0,
        max_iter: int = 1000,
        dimensions: int = 2,
        ax=None,
        title: str = None
) -> list:
    """
    Use t-SNE to project the data into 2D space and plot clusters on the provided axis.
    
    Args:
        df (pd.DataFrame): DataFrame with clustering results (must contain 'cluster').
        perplexity (float): t-SNE perplexity parameter.
        max_iter (int): Number of iterations for t-SNE.
        ax: A matplotlib axis on which to plot.
        title (str): Title for the subplot (displayed as the time period).
    
    Returns:
        list: The scatter plot handles for the clusters (for the common legend).
    """
    # Drop the 'cluster' column from features
    features = df.drop(columns='cluster', axis=1)
    tsne = TSNE(n_components=dimensions, perplexity=perplexity, max_iter=max_iter, random_state=1)
    tsne_components = tsne.fit_transform(features)
    
    tsne_df = pd.DataFrame(tsne_components, columns=['Dim1', 'Dim2'])
    tsne_df['cluster'] = df['cluster'].values

    handles = []
    for clust in sorted(tsne_df['cluster'].unique()):
        cluster_data = tsne_df[tsne_df['cluster'] == clust]
        sc = ax.scatter(cluster_data['Dim1'], cluster_data['Dim2'], label=f'Cluster {clust}')
        handles.append(sc)
    if title:
        ax.set_title(title)
    return handles


def plot_tsne_clustering(
        df: pd.DataFrame = None,
        time_periods: list = None,
        dimensions: int = 2,
):
    
    # Load the pre-clustered data (assumed to contain a 'date' column and 'cluster' column)
    df = pd.read_csv('processed_data/clustered_data.csv')
    # Define fixed time periods for visualization
    time_periods = ['2023-02', '2023-03', '2023-04', '2023-05']
    
    # Create a 2 x 2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    common_handles = None
    for i, period in enumerate(time_periods):
        # Filter the DataFrame by the given period
        period_df = df[df['date'] == period].copy()
        # Drop extra columns that are not used for t-SNE (if needed)
        period_df = period_df.drop(columns=['date', 'ticker'], errors='ignore')
        # Plot onto the corresponding subplot using t-SNE projection
        handles = tsne_clustering(period_df, perplexity=30, max_iter=1000, dimensions=dimensions, ax=axes[i], title=period)
        if common_handles is None:
            common_handles = handles
    
    # Create one common legend for all subplots
    fig.legend(common_handles, [h.get_label() for h in common_handles], loc='upper right')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()