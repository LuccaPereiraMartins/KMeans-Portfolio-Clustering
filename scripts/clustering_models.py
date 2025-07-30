
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids


def simple_kmeans(clusters: int, random_state: int = 1) -> KMeans:
    """Create a KMeans model with custom initial centroids for momentum clustering."""
    init = [
        [0,0,0,0,0,-1,48,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,48.5,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0.2,49,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,5,50,0,0,0,0,0,0,0,0,0]
    ]
    return KMeans(
        n_clusters=clusters,
        random_state=random_state,
        init=init,
    )


def kmedoids(clusters: int, random_state: int = 1) -> KMedoids:
    """Create a KMedoids clustering model."""
    return KMedoids(
        n_clusters=clusters,
        random_state=random_state,
    )

# TODO: Consider adding GMM, agglomerative clustering, or DBSCAN models.