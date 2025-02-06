
import pandas as pd
import matplotlib.pyplot as plt
import clustering_models
from pandasgui import show as gui_show
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler,StandardScaler

def cluster(
        df: pd.DataFrame,
        # model,
        clusters: int = 4,
        random_state: int = 0,
        initial_centroids = 'random',
        transform = StandardScaler()
) -> pd.DataFrame:

    """
    Apply the K-means clustering algorithm to the data
    Built to handle month by month data, rather than entire dataset
    """

    # model = clustering_models.simple_kmeans(
    #     clusters=clusters,
    #     random_state=random_state,
    #     init=initial_centroids
    # )

    model = clustering_models.kmedoids(
        clusters = clusters,
        random_state = random_state,
    )

    # robustness: ensure date and ticker are not columns
    clean_df = df.loc[:, ~df.columns.isin(['dollar_volume', 'date', 'ticker'])]
    # standardize the data column-wise per date
    clean_df = transform.fit_transform(clean_df)
    # add a column with the cluster of each row
    df['cluster'] = model.fit(clean_df).labels_

    return df


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
        clusters: int = 5
) -> pd.DataFrame:
    

    # might need a line to set the multi-index
    # LINE
    # apply the clustering model 
    data = df.dropna().groupby('date').apply(cluster)

    # reset the index to remove duplicated date column
    data.index.names = ['delete','date','ticker']
    data = data.reset_index(drop=False)
    data.drop('delete', axis=1,inplace=True)
    data = data.set_index(['date','ticker'])

    return df


def main():

    
    # read in the processed data with multi-index
    data = pd.read_csv('final_df.csv', index_col=('date','ticker'))
    data = data.drop(columns=['dollar_volume'])
    # apply the clustering algorithm to each month
    data = data.dropna().groupby('date').apply(cluster)
    # reset the index to remove duplicated date column
    data.index.names = ['delete','date','ticker']
    data = data.reset_index(drop=False)
    data.drop('delete', axis=1,inplace=True)
    data.to_csv('temp_clustered.csv', index=False)
    data = data.set_index(['date','ticker'])
    

    print(data)
    print(data.loc['2024-06'])

    plot_clusters(
        df=data.loc['2024-03'],
        )

    if False:
        gui_show(data)


if __name__ == '__main__':
    main()