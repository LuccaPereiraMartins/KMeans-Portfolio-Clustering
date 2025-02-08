
from sklearn.cluster import KMeans
# from sklearn_extra.cluster import KMedoids


def simple_kmeans(clusters: int,
                  random_state: int = 1,
                  init = 'k-means++',
                  ):
    
    # set the initial centroids based on MACD and RSI
    # aim is to split into low -> high momentum clusters
    init = [
            [0,0,0,0,0,-0.25,30,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,45,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0.5,60,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,70,0,0,0,0,0,0,0,0,0]
    ]

    simple_kmeans = KMeans(
            n_clusters=clusters,
            random_state=random_state,
            init=init,
        )
    
    return simple_kmeans


# def kmedoids(clusters: int,
#             random_state: int = 1,
#             ):

#     kmedoids = KMedoids(
#         n_clusters = clusters,
#         random_state = random_state,
#         )
    
#     return kmedoids


# consider a Gaussian Mixture Model (GMM)