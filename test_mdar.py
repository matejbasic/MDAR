# -*- coding: utf-8 -*-

"""Recommender framework based on association analysis and graph database.

Before running this file, database should be installed from one of datasets
(https://github.com/matejbasic/recomm-ecommerce-datasets) and imported to Neo4j
database with DB Importer (https://github.com/matejbasic/recommenders-playground).

Dependencies:
    py2neo
    numpy
    scipy

Constants:
    K_FOLD_SIZE: number of k parts for cross-validation.
    CONFIG_PATH: path to config file, see config_sample.json.
    K: lengths of returned recommendations.
    USED_APPROACHES: used algorithms and their weights[0-1]

Usage:
    $ python test_mdar.py
"""

import time
from mdar.recommender import MDAR
from mdar.data_manager import DataManager
from mdar.tester import Tester
from mdar.results import Results

K_FOLD_SIZE = 3
CONFIG_PATH = 'config.json'
K = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 75, 100]

USED_APPROACHES = [
    ('order_association', 1),
    ('user_history', 1),
    ('user_history2', 1),
    ('time_related', 1),
]

def get_dmrec():
    """DataManager and BaseRecommender object pairs for each k data part.

    Returns:
        list: K_FOLD_SIZE length, contains dicts of following structure
            {
                'dm': DataManager instance
                'rec': BaseRecommender instance
            }
    """
    dmrec = []
    for i in range(0, K_FOLD_SIZE):
        data_manager = DataManager(CONFIG_PATH, K_FOLD_SIZE)
        data_manager.testing_part_index = i

        rec = MDAR(used_approaches=USED_APPROACHES)
        rec.data_manager = data_manager

        start = time.time()
        rec.train(max(K))
        dmrec.append({'dm': data_manager, 'rec': rec})
        print 'training time: %f' % (time.time() - start)

    return dmrec

def test():
    """Test MDAR recommender for each k in constant K with all the approaches
    defined in USED_APPROACHES."""
    dmrec = get_dmrec()
    tester = Tester()

    for k in K:
        print 'recommending with k=%d items' % k
        tester.k = k
        results = Results()
        for i in range(0, K_FOLD_SIZE):
            tester.recommender = dmrec[i]['rec']
            tester.data_manager = dmrec[i]['dm']

            start = time.time()
            precision, recall, fallout, f1_score, specificity, \
            confusion_matrix, cases_without_history = tester.test()

            results.add(
                precision,
                recall,
                fallout,
                f1_score,
                specificity,
                confusion_matrix,
                cases_without_history
            )

            end = time.time() - start
            print 'recommendation testing time for k=%d: %f' % (k, end)

        results.summarize()
        precision, recall, fallout, f1_score, specificity = results.get_evaluation_measures()

        print results.get_confusion_matrix()
        print 'model\t k\t precision\t recall\t fallout\t F1\t specificity'
        print 'MDAR\t%d\t%f\t%f\t%f\t%f\t%s' \
        % (k, precision, recall, fallout, f1_score, specificity)
        print '-' * 22

        data_manager = DataManager(CONFIG_PATH, K_FOLD_SIZE)
        results.log_results('MDAR', k, data_manager.get_items_count('all'))

test()
