import time
import numpy as np
from query_manager import QueryManager

class Tester:
    k  = 0
    dm = None

    cases_without_history = 0

    def __init__(self, config_path = None, k_fold_size = 3):
        if config_path is not None:
            self.dm = DataManager(config_path, k_fold_size)

    def test(self):
        confusion_matrix = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

        cases_without_history = 0
        current_order_id      = -1
        previous_order_items  = []

        start = time.time()
        items_count = self.dm.get_items_count('train')

        orders = self.dm.get_orders('test')
        for order in orders:
            if order['order'] != current_order_id:
                current_order_id     = order['order']
                previous_order_items = []

            recommendations = self.rec.recommend(self.k, order, previous_order_items)
            # print recommendations
            if len(recommendations) > 0:
                if order['item'] in recommendations:
                    confusion_matrix['tp'] += 1
                    confusion_matrix['fp'] += len(recommendations) - 1
                    confusion_matrix['tn'] += items_count - len(recommendations)
                else:
                    confusion_matrix['fn'] += 1
                    confusion_matrix['fp'] += len(recommendations)
                    confusion_matrix['tn'] += items_count - len(recommendations) - 1
            else:
                cases_without_history += 1

            # end of current iteration
            previous_order_items.append({'item': order['item'], 'cats': order['cats']})

        end = time.time()
        precision, recall, fallout, f1, specificity = self.get_confusion_metrics(confusion_matrix)

        return precision, recall, fallout, f1, specificity, confusion_matrix, cases_without_history

    def get_confusion_metrics(self, confusion_matrix):
        try:
            precision = float(confusion_matrix['tp']) / (confusion_matrix['tp'] + confusion_matrix['fp'])
        except ZeroDivisionError:
            precision = 0

        try:
            recall = float(confusion_matrix['tp']) / (confusion_matrix['tp'] + confusion_matrix['fn'])
        except ZeroDivisionError:
            recall = 0

        try:
            fallout = float(confusion_matrix['fp']) / (confusion_matrix['fp'] + confusion_matrix['tn'])
        except ZeroDivisionError:
            fallout = 0

        try:
            specificity = float(confusion_matrix['tn']) / (confusion_matrix['tn'] + confusion_matrix['fp'])
        except ZeroDivisionError:
            specificity = 0

        try:
            f1 = (2 * precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1 = 0

        return precision, recall, fallout, f1, specificity

    def set_k(self, k):
        try:
            self.k = int(k)
        except (ValueError, TypeError):
            self.k = 0

    def set_recommender(self, recommender):
        self.rec = recommender

    def set_data_manager(self, dm):
        self.dm = dm
