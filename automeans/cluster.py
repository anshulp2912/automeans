"""Automatic K-means clustering."""

# Authors: Anshul Patel <anshulp2912@gmail.com>

# License: MIT


import math
import operator
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

class Point:
    """ Init Point object
    Parameters
    ----------
    initx : X-coordinate of point
    inity : Y-coordinate of point
    """
    def __init__(self,initx,inity):
        self.x = initx
        self.y = inity
    def getX(self):
        return self.x
    def getY(self):
        return self.y

    # Function to calculate point to line distance
    def distance_to_line(self, p1, p2):
        x_diff = p2.x - p1.x
        y_diff = p2.y - p1.y
        num = abs(y_diff*self.x - x_diff*self.y + p2.x*p1.y - p2.y*p1.x)
        den = math.sqrt(y_diff**2 + x_diff**2)
        return num / den
        
class ameans:
    """Init n_clusters seeds according to k-means++
    Parameters
    ----------
    max_clusters : int > 0
        The number of maximum seeds to choose.
    metrics : {"standard", "kneed", "silhouette"}, default="standard"
        Metric to choose the best number of cluster
    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.
        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.
        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.
    precompute_distances : {'auto', True, False}
        Precompute distances (faster but takes more memory).
        'auto' : do not precompute distances if n_samples * n_clusters > 12
        million. This corresponds to about 100MB overhead per job using
        double precision.
        True : always precompute distances
        False : never precompute distances
        .. deprecated:: 0.23
            'precompute_distances' was deprecated in version 0.23 and will be
            removed in 0.25. It has no effect.
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.
    verbose : bool, default=False
        Verbosity mode.
    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.
    n_jobs : int, default=None
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.
        ``None`` or ``-1`` means using all processors.
        .. deprecated:: 0.23
            ``n_jobs`` was deprecated in version 0.23 and will be removed in
            0.25.
    algorithm : {"auto", "full", "elkan"}, default="auto"
        K-means algorithm to use. The classical EM-style algorithm is "full".
        The "elkan" variation is more efficient on data with well-defined
        clusters, by using the triangle inequality. However it's more memory
        intensive due to the allocation of an extra array of shape
        (n_samples, n_clusters).
        For now "auto" (kept for backward compatibiliy) chooses "elkan" but it
        might change in the future for a better heuristic.
    """
    
    def __init__(self, max_clusters, init = 'k-means++', metrics='standard', n_init=10,
                 max_iter=300, tol=1e-4, precompute_distances='deprecated',
                 verbose=0, random_state=None, copy_x=True, n_jobs='deprecated', algorithm='auto'):
        """ Init n_clusters seeds according to k-means++ 
        """
        self.max_clusters = max_clusters
        self.init = init
        self.metrics = metrics
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.precompute_distances = precompute_distances
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs
        self.algorithm = algorithm
        
        if metrics not in ['standard', 'kneed', 'silhouette']:
            raise ValueError(f"Metrics must be 'standard', 'kneed' or 'silhouette', "
                                 f"got {metrics} instead.")
    
        if type(max_clusters) != int:
            raise ValueError(f"max_clusters must be of type 'int', "
                                f"got {type(max_clusters)} instead.")
        
    def fit(self,X):
        """Compute k-means clustering.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.
        Returns
        -------
        model: Best fitted k-means sklearn model
        cluster: int
            Optimal number of cluster
        """
        if self.metrics in ['standard', 'kneed']:
        # Calculate inertia for each cluster seed
            wcss = []
            distances = []
            for i in range(1, int(self.max_clusters)):
              kmeans = KMeans(i, init=self.init, n_init=self.n_init, max_iter=self.max_iter, 
                               tol=self.tol, precompute_distances=self.precompute_distances,
                                verbose=self.verbose, random_state=self.random_state,
                                 copy_x=self.copy_x, n_jobs=self.n_jobs, algorithm=self.algorithm)
              kmeans.fit(X)
              wcss.append(kmeans.inertia_)
    
            if self.metrics == 'standard':
                for i in range(0,int(self.max_clusters)-1):
                    p1 = Point(initx=1,inity=wcss[0])
                    p2 = Point(initx=int(self.max_clusters)-1,inity=wcss[int(self.max_clusters)-2])
                    p = Point(initx=i+1,inity=wcss[i])
                    distances.append(p.distance_to_line(p1,p2))
    
                # Find the best cluster size
                max_index, max_value = max(enumerate(distances), key=operator.itemgetter(1))
                max_index = max_index + 1
    
                # Return the best model with best cluster size
                kmeans = KMeans(max_index, init=self.init, n_init=self.n_init, max_iter=self.max_iter, 
                               tol=self.tol, precompute_distances=self.precompute_distances,
                                verbose=self.verbose, random_state=self.random_state,
                                 copy_x=self.copy_x, n_jobs=self.n_jobs, algorithm=self.algorithm)
                kmeans.fit(X)
    
                return kmeans, max_index
    
            elif self.metrics == 'kneed':
                cluster_range = range(1, int(self.max_clusters))
                clusters_df = pd.DataFrame({"cluster_errors": wcss, "num_clusters": cluster_range})
                elbow = KneeLocator(clusters_df.num_clusters.values, clusters_df.cluster_errors.values, S=1.0, curve='convex', direction='decreasing')
                
                # Find the best cluster size
                max_index = int(elbow.knee)
    
                # Return the best model with best cluster size
                kmeans = KMeans(max_index, init=self.init, n_init=self.n_init, max_iter=self.max_iter, 
                               tol=self.tol, precompute_distances=self.precompute_distances,
                                verbose=self.verbose, random_state=self.random_state,
                                 copy_x=self.copy_x, n_jobs=self.n_jobs, algorithm=self.algorithm)
                kmeans.fit(X)
    
                return kmeans, max_index
    
        elif self.metrics == 'silhouette':
            sil_score_max = -1 #this is the minimum possible score
    
            for i in range(2, int(self.max_clusters)):
              kmeans = KMeans(i, init=self.init, n_init=self.n_init, max_iter=self.max_iter, 
                               tol=self.tol, precompute_distances=self.precompute_distances,
                                verbose=self.verbose, random_state=self.random_state,
                                 copy_x=self.copy_x, n_jobs=self.n_jobs, algorithm=self.algorithm)
              labels = kmeans.fit_predict(X)
              sil_score = silhouette_score(X, labels)
              # Find the best cluster size
              if sil_score > sil_score_max:
                sil_score_max = sil_score
                max_index = i
    
            # Return the best model with best cluster size
            kmeans = KMeans(max_index, init=self.init, n_init=self.n_init, max_iter=self.max_iter, 
                               tol=self.tol, precompute_distances=self.precompute_distances,
                                verbose=self.verbose, random_state=self.random_state,
                                 copy_x=self.copy_x, n_jobs=self.n_jobs, algorithm=self.algorithm)
            kmeans.fit(X)
    
            return kmeans, max_index
        