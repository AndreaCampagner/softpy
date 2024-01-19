from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClusterMixin
from fuzzy.fuzzyset import DiscreteFuzzySet

class FuzzyCMeans(BaseEstimator,ClusterMixin):
  '''
  Implements the fuzzy c-means algorithm. The interface is compatible with the scikit-learn library. It allows to set the number of clusters,
  a tolerance degree (for avoiding errors in numerical operations), the number of iterations before termination, the clustering metric as well
  as the fuzzifier degree.
  '''
  def __init__(self, n_clusters=2, epsilon=0.001, iters=100, random_state=None, metric='euclidean', fuzzifier=2):
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
    if not self.fitted:
      raise RuntimeError("Estimator must be fitted")
    dists = pairwise_distances(X, self.centroids, metric=self.metric)
    self.cluster_assignments = (dists+self.epsilon)**(2/(1-self.fuzzifier))/(np.sum((dists+self.epsilon)**(2/(1-self.fuzzifier)), axis=1)[:,np.newaxis])
    #self.cluster_assignments = self.cluster_assignments/np.sum(self.cluster_assignments, axis=1)[:, np.newaxis]
    return np.argmax(self.cluster_assignments, axis=1)
  
  def predict_proba(self, X):
    ''' 
    For each given instance returns the membership degrees to the computed clusters. The fit method must have been called before executing this method.
    '''
    if not self.fitted:
      raise RuntimeError("Estimator must be fitted")
    dists = pairwise_distances(X, self.centroids, metric=self.metric)
    self.cluster_assignments = (dists+self.epsilon)**(2/(1-self.fuzzifier))/(np.sum((dists+self.epsilon)**(2/(1-self.fuzzifier)), axis=1)[:,np.newaxis])
    fuzzy_sets = []
    for cl in range(self.cluster_assignments.shape[1]):
      fuzzy_sets.append(DiscreteFuzzySet(list(range(X.shape[0])), self.cluster_assignments[:,cl]))
    return np.array(fuzzy_sets)

  def fit_predict(self, X, y=None):
    self.fit(X,y)
    return self.predict(X)