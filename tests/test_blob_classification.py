#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import sklearn
from sklearn.datasets import make_blobs

from src.blob_classification import blob_classification


def spy_decorator(method_to_decorate, name):
    """
    Wrap a method so calls to it are recorded on a MagicMock while the
    original implementation still runs.

    This solution to wrap a patched method without obstructing its
    implementation comes originally from
    https://stackoverflow.com/questions/25608107/
    """
    mock = MagicMock(name="%s method" % name)

    def wrapper(self, *args, **kwargs):
        mock(*args, **kwargs)
        return method_to_decorate(self, *args, **kwargs)
    wrapper.mock = mock
    return wrapper


class TestBlobClassification(unittest.TestCase):

    def test_correctness(self):
        a = np.array([[2., 2., 0., 2.5, 0.76],
                      [2., 3., 1., 1.5, 0.96],
                      [2., 2., 6., 3.5, 0.84],
                      [2., 2., 3., 1.2, 1.],
                      [2., 4., 4., 2.7, 0.8]])
        idx = np.arange(5)
        np.random.shuffle(idx)
        for row in a[idx]:
            X, y = make_blobs(100, int(row[0]), centers=int(row[1]),
                               random_state=int(row[2]), cluster_std=row[3])
            acc = blob_classification(X, y)
            self.assertAlmostEqual(
                acc, row[-1],
                msg="Incorrect accuracy score for blobs with n_features=%d, "
                    "centers=%d, random_state=%d, cluster_std=%s! Expected "
                    "%s." % (int(row[0]), int(row[1]), int(row[2]), row[3],
                             row[-1]))

    def test_calls(self):
        row = [2., 2., 0., 2.5, 0.76]
        X, y = make_blobs(100, int(row[0]), centers=int(row[1]),
                           random_state=int(row[2]), cluster_std=row[3])
        predict_method = spy_decorator(
            sklearn.naive_bayes.GaussianNB.predict, "predict")
        fit_method = spy_decorator(sklearn.naive_bayes.GaussianNB.fit, "fit")
        with patch("src.blob_classification.train_test_split",
                   wraps=sklearn.model_selection.train_test_split) as tts, \
             patch("src.blob_classification.accuracy_score",
                   wraps=sklearn.metrics.accuracy_score) as acs, \
             patch.object(sklearn.naive_bayes.GaussianNB, "fit",
                          new=fit_method), \
             patch.object(sklearn.naive_bayes.GaussianNB, "predict",
                          new=predict_method), \
             patch("src.blob_classification.GaussianNB",
                   wraps=sklearn.naive_bayes.GaussianNB) as gnb:

            blob_classification(X, y)

            # Check that train_test_split is called with correct parameters
            tts.assert_called_once()
            args, kwargs = tts.call_args
            self.assertIn(
                "random_state", kwargs,
                msg="You did not specify the random_state argument to "
                    "train_test_split!")
            self.assertEqual(
                kwargs["random_state"], 0,
                msg="Incorrect random_state argument to train_test_split! "
                    "Expected 0.")
            if "train_size" in kwargs:
                self.assertEqual(
                    kwargs["train_size"], 0.75,
                    msg="Incorrect train_size argument to train_test_split! "
                        "Expected 0.75.")
            elif "test_size" in kwargs:
                self.assertEqual(
                    kwargs["test_size"], 0.25,
                    msg="Incorrect test_size argument to train_test_split! "
                        "Expected 0.25.")

            # Check that accuracy_score is called
            acs.assert_called_once()

            # Check that GaussianNB is called
            gnb.assert_called_once()

            # Check that fit and predict methods of GaussianNB object are
            # called
            predict_method.mock.assert_called()
            fit_method.mock.assert_called()


if __name__ == '__main__':
    unittest.main()
