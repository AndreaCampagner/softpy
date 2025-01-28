from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClusterMixin
from .fuzzyset import DiscreteFuzzySet
from .fuzzy_partition import FuzzyPartition
from typing import Callable

class FuzzyCMeans(BaseEstimator,ClusterMixin):
  '''
  Implements the fuzzy c-means algorithm. The interface is compatible with the scikit-learn library.
  
  Parameters
  ----------
  :param n_clusters: The number of centroids used to define the number of clusters
  :type n_clusters: int, default=2
  
  :param epsilon: Error tolerance parameter for avoid division by 0
  :type n_clusters: float, default=0.001

  :param iters: The number of iterations for the optimization routine
  :type iters: int, default=100

  :param random_state: Random seed parameter for repeatability. None corresponds to randomized execution.
  :type random_state: int|RandomState|None, default=None

  :param metric: metric for the computation of distances in the fuzzy c-means algorithm
  :type metric: str|Callable, default=euclidean

  :param fuzzifier: paramater that controls the hardness of the clustering result. Values closer to 1 will enforce a clustering result closer to that obtain with standard k-means
  :type fuzzifier: np.number (should be larger than 1), default=2
  '''
  def __init__(self, n_clusters: int=2, epsilon: float=0.001, iters: int=100, random_state:int|np.random.RandomState|None=None, metric: str|Callable ='euclidean', fuzzifier: np.number=2):
    if n_clusters <= 1:
      raise ValueError("n_clusters must be an int larger than 1, was %d" % n_clusters)
    
    if fuzzifier <= 1:
      raise ValueError("fuzzifier must be a number larger than 1, was %.2f" % fuzzifier)
    
    if epsilon <= 0:
      raise ValueError("epsilon must be a number larger than 0, was %.2f" % epsilon)
    
    if iters < 1:
      raise ValueError("iters must be an int larger than 0, was %d" % iters)
    
    self.n_clusters = n_clusters
    self.epsilon = epsilon
    self.iters = iters
    self.random_state = random_state
    self.metric = metric
    self.fuzzifier = fuzzifier

  def fit(self, X, y=None):
    '''
    Applies clustering to the given data, by iteratively partially assigning instances to clusters and then recomputing the clusters' centroids.
    '''
    self.centroids = resample(X, replace=False, n_samples=self.n_clusters, random_state=self.random_state)
    self.cluster_assignments = np.zeros((X.shape[0], self.n_clusters))
    for it in range(self.iters):
      dists = pairwise_distances(X, self.centroids, metric=self.metric)

      self.cluster_assignments = (dists+self.epsilon)**(2/(1-self.fuzzifier))/(np.sum((dists+self.epsilon)**(2/(1-self.fuzzifier)), axis=1)[:,np.newaxis])
      
      for k in range(self.n_clusters):
        self.centroids[k] = np.sum(self.cluster_assignments[:,k][:,np.newaxis]**self.fuzzifier*X, axis=0)/(np.sum(self.cluster_assignments[:,k]**self.fuzzifier))

    self.fitted = True
    return self

  def predict(self, X):
    ''' 
    For each given instance returns the cluster with maximum membership degree. The fit method must have been called before executing this method.
    '''
    try:
      self.fitted
    except:
      raise RuntimeError("Estimator must be fitted")
    dists = pairwise_distances(X, self.centroids, metric=self.metric)
    self.cluster_assignments = (dists+self.epsilon)**(2/(1-self.fuzzifier))/(np.sum((dists+self.epsilon)**(2/(1-self.fuzzifier)), axis=1)[:,np.newaxis])
    #self.cluster_assignments = self.cluster_assignments/np.sum(self.cluster_assignments, axis=1)[:, np.newaxis]
    return np.argmax(self.cluster_assignments, axis=1)
  
  def predict_proba(self, X):
    ''' 
    For each given instance returns the membership degrees to the computed clusters. The fit method must have been called before executing this method.
    '''
    try:
      self.fitted
    except:
      raise RuntimeError("Estimator must be fitted")
    dists = pairwise_distances(X, self.centroids, metric=self.metric)
    self.cluster_assignments = (dists+self.epsilon)**(2/(1-self.fuzzifier))/(np.sum((dists+self.epsilon)**(2/(1-self.fuzzifier)), axis=1)[:,np.newaxis])
    return self.cluster_assignments
  
  def predict_fuzzy(self, X, name="fcm"):
    ''' 
    For each given instance returns the membership degrees to the computed clusters. The fit method must have been called before executing this method.
    '''
    try:
      self.fitted
    except:
      raise RuntimeError("Estimator must be fitted")
    dists = pairwise_distances(X, self.centroids, metric=self.metric)
    self.cluster_assignments = (dists+self.epsilon)**(2/(1-self.fuzzifier))/(np.sum((dists+self.epsilon)**(2/(1-self.fuzzifier)), axis=1)[:,np.newaxis])
    fuzzy_sets = {}
    for cl in range(self.cluster_assignments.shape[1]):
      fuzzy_sets[str(cl)] = DiscreteFuzzySet(list(range(X.shape[0])), self.cluster_assignments[:,cl])
    fp = FuzzyPartition(name, fuzzy_sets)
    return fp

  def fit_predict(self, X, y=None):
    self.fit(X,y)
    return self.predict(X)