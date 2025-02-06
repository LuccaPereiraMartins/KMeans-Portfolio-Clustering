
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids


def simple_kmeans(clusters: int,
                  random_state: int = 1,
                  init = 'Random',
                  ):
    
    simple_kmeans = KMeans(
            n_clusters=clusters,
            random_state=random_state,
            init=init,
        )
    
    return simple_kmeans


def kmedoids(clusters: int,
            random_state: int = 1,
            ):

    kmedoids = KMedoids(
        n_clusters = clusters,
        random_state = random_state,
        )
    
    return kmedoids


# consider a Gaussian Mixture Model (GMM)