import sys

from sklearn.datasets import load_iris
import pytest

sys.path.append(__file__ + "/../..")

from softpy.fuzzy.clustering import FuzzyCMeans
from tests.fuzzy_set.configuration import not_raises 

class TestFuzzyCMeans:

    @pytest.mark.parametrize(
        "n_clusters,epsilon,iters,fuzzifier,exception_expected",
        [
            (2,0.01,2,1.05,None),
            (1,0.01,2,1.05,ValueError),
            (2,-0.01,2,1.05,ValueError),
            (2,0.01,0,1.05,ValueError),
            (2,0.01,2,1,ValueError),
        ]
    )
    def test_creation(self,n_clusters,epsilon,iters,fuzzifier,exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                FuzzyCMeans(n_clusters=n_clusters,epsilon=epsilon,iters=iters,fuzzifier=fuzzifier)
        else:
            with pytest.raises(exception_expected) as e_info:
                FuzzyCMeans(n_clusters=n_clusters,epsilon=epsilon,iters=iters,fuzzifier=fuzzifier)

    @pytest.mark.parametrize(
        "exception_expected",
        [
            (None),
            (RuntimeError)
        ]
    )
    def test_fit(self,exception_expected: Exception):
        if exception_expected == None:
            with not_raises() as e_info:
                X, y = load_iris(return_X_y=True)
                f = FuzzyCMeans()
                f.fit(X,y)
                f.predict(X)
                f.predict_fuzzy(X)
                f.predict_proba(X)
        else:
            with pytest.raises(exception_expected) as e_info:
                X, y = load_iris(return_X_y=True)
                f = FuzzyCMeans()
                f.predict(X)
                f.predict_fuzzy(X)
                f.predict_proba(X)
