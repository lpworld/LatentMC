"""
The :mod:`surprise.accuracy` module provides with tools for computing accuracy
metrics on a set of predictions.

Available accuracy metrics:

.. autosummary::
    :nosignatures:

    rmse
    mae
    fcp
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import numpy as np
from six import iteritems
import json

def precision_1(predictions, n=1, verbose=True):
    """Compute Precision@N.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.

    Returns:
        The Precision of Revlevant Items within Top@N Recommendations

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    userid = [x[0] for x in predictions]
    precision = []
    for user in userid:
        user_list = [x for x in predictions if x[0] == user]
        user_list_sort = sorted(user_list, key=lambda x:x[3], reverse=True)[:n]
        user_list_sort = [x for x in user_list_sort if x[2][0] > 3.5]
        user_list = [x for x in user_list if x[2][0] > 3.5]
        if len(user_list) > 0:
            precision.append(len(user_list_sort)/len(user_list))
    precision = np.mean(precision)

    if verbose:
        print('Precision: {0:1.4f}'.format(precision))

    return precision
    
def recall_1(predictions, n=1, verbose=True):
    """Compute Recall@N.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.

    Returns:
        The Recall of Revlevant Items within Top@N Recommendations

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    userid = [x[0] for x in predictions]
    recall = []
    for user in userid:
        user_list = [x for x in predictions if x[0] == user]
        user_list = sorted(user_list, key=lambda x:x[3], reverse=True)[:n]
        user_list = [x for x in user_list if x[2][0] > 3.5]
        recall.append(len(user_list)/n)
    recall = np.mean(recall)

    if verbose:
        print('Recall: {0:1.4f}'.format(recall))

    return recall

def precision_5(predictions, n=5, verbose=True):
    """Compute Precision@N.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.

    Returns:
        The Precision of Revlevant Items within Top@N Recommendations

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    userid = [x[0] for x in predictions]
    precision = []
    for user in userid:
        user_list = [x for x in predictions if x[0] == user]
        user_list_sort = sorted(user_list, key=lambda x:x[3], reverse=True)[:n]
        user_list_sort = [x for x in user_list_sort if x[2][0] > 3.5]
        user_list = [x for x in user_list if x[2][0] > 3.5]
        if len(user_list) > 0:
            precision.append(len(user_list_sort)/len(user_list))
    precision = np.mean(precision)

    if verbose:
        print('Precision: {0:1.4f}'.format(precision))

    return precision

def recall_5(predictions, n=5, verbose=True):
    """Compute Recall@N.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.

    Returns:
        The Recall of Revlevant Items within Top@N Recommendations

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    userid = [x[0] for x in predictions]
    recall = []
    for user in userid:
        user_list = [x for x in predictions if x[0] == user]
        user_list = sorted(user_list, key=lambda x:x[3], reverse=True)[:n]
        user_list = [x for x in user_list if x[2][0] > 3.5]
        recall.append(len(user_list)/n)
    recall = np.mean(recall)

    if verbose:
        print('Recall: {0:1.4f}'.format(recall))

    return recall