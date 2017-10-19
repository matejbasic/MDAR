# -*- coding: utf-8 -*-

from mdar.data_manager import DataManager
from mdar.recommender import MDAR


class Tester(object):
    """Tester class used for testing recommendations against test data,
    collecting results in form of confusion matrix and calculating results using
    different IR metrics.

    Args:
        config_path(str, optional): path to config file used in DataManager.
        k_fold_size(int, optional): number of k parts for cross-validation.
        Defaults to 3.
    """

    def __init__(self, config_path=None, k_fold_size=3):
        self._k = 0
        self._recommender = None

        if config_path is not None:
            self._data_manager = DataManager(config_path, k_fold_size)

    def test(self):
        """Demands the recommender property to be set up which is used for
        generating k recommendations which are tested against test data(orders).
        Results are save in confusion_matrix dict which is returned with values
        of other IR metrics.

        Returns:
            float: precision.
            float: recall.
            float: fallout.
            float: F1 score.
            float: specificity.
            dict: confusion_matrix with following structure
                {
                    'tp': int
                    'tn': int
                    'fp': int
                    'fn': int
                }
            int: cases without history.

        """
        confusion_matrix = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

        poi = []    # previous order items
        current_order_id = -1
        cases_without_history = 0

        items_count = self.data_manager.get_items_count('train')

        orders = self.data_manager.get_orders('test')
        for order in orders:
            if order['order'] != current_order_id:
                poi = []
                current_order_id = order['order']

            recommendations = self.recommender.recommend(self.k, order, poi)
            # print recommendations
            if recommendations:
                if order['item'] in recommendations:
                    confusion_matrix['tp'] += 1
                    confusion_matrix['fp'] += len(recommendations) - 1

                    true_negative = items_count - len(recommendations)
                    if true_negative > 0:
                        confusion_matrix['tn'] += true_negative
                else:
                    confusion_matrix['fn'] += 1
                    confusion_matrix['fp'] += len(recommendations)

                    true_negative = items_count - len(recommendations) - 1
                    if true_negative > 0:
                        confusion_matrix['tn'] += true_negative
            else:
                cases_without_history += 1

            # end of current iteration
            poi.append({'item': order['item'], 'cats': order['cats']})

        precision, recall, fallout, f1_score, specificity = \
        self.get_evaluation_measures(confusion_matrix)

        return precision, recall, fallout, f1_score, specificity, \
        confusion_matrix, cases_without_history

    @staticmethod
    def get_evaluation_measures(confusion_matrix):
        """Calculates IR measures such as precision, recall, fallout and other
        from given confusion matrix.

        Args:
            confusion_matrix(dict): object with following structure:
                {
                    'tp': int
                    'tn': int
                    'fp': int
                    'fn': int
                }

        Returns:
            float: precision.
            float: recall.
            float: fallout.
            float: F1 score.
            float: specificity.
        """
        try:
            precision = float(confusion_matrix['tp'])
            precision /= confusion_matrix['tp'] + confusion_matrix['fp']
        except ZeroDivisionError:
            precision = 0

        try:
            recall = float(confusion_matrix['tp'])
            recall /= confusion_matrix['tp'] + confusion_matrix['fn']
        except ZeroDivisionError:
            recall = 0

        try:
            fallout = float(confusion_matrix['fp'])
            fallout /= confusion_matrix['fp'] + confusion_matrix['tn']
        except ZeroDivisionError:
            fallout = 0

        try:
            specificity = float(confusion_matrix['tn'])
            specificity /= confusion_matrix['tn'] + confusion_matrix['fp']
        except ZeroDivisionError:
            specificity = 0

        try:
            f1_score = (2 * precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1_score = 0

        return precision, recall, fallout, f1_score, specificity

    @property
    def k(self):
        """int: number of generated recommendations for each tested order."""
        return self._k

    @k.setter
    def k(self, value):
        try:
            self._k = int(value)
        except (ValueError, TypeError):
            self._k = 0

    @property
    def recommender(self):
        """BaseRecommender: recommender instance used for testing."""
        return self._recommender

    @recommender.setter
    def recommender(self, value):
        if isinstance(value, MDAR):
            self._recommender = value
        else:
            self._recommender = None

    @property
    def data_manager(self):
        """DataManager: object used for data fetching from database."""
        return self._data_manager

    @data_manager.setter
    def data_manager(self, value):
        if isinstance(value, DataManager):
            self._data_manager = value
        else:
            self._data_manager = None
