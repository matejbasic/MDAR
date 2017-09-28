import time
import numpy as np
from operator import itemgetter
from data_manager import DataManager
from query_manager import QueryManager
from collections import OrderedDict
from math import floor, ceil

from recommenders.base import BaseRecommender
from recommenders.oa import OrderAssociationRecommender
from recommenders.uh import UserHistoryRecommender
from recommenders.uh2 import UserHistory2Recommender
from recommenders.tr import TimeRelatedRecommender

class MDAR(BaseRecommender):
    minimal_arhr_value = .5

    user_approaches_w = {}

    used_approaches      = []
    available_approaches = [
        'order_association',
        'user_history',
        'user_history2',
        'time_related'
    ]

    def __init__(self, config_path = None, k_fold_size = 3, used_approaches = None):
        if config_path is not None:
            super(MDARRecommender, self).__init__(config_path, k_fold_size)
        self.set_used_approaches(used_approaches)

    def train(self, k = None):
        k = 10 if k < 10 else k

        self._init_model()

        poi = []
        current_order_id = -1
        start = time.time()

        orders        = self.dm.get_orders('train')
        orders_count  = len(orders)
        max_poi_count = 0

        # populate previous order items
        for i in range(0, len(orders)):
            if orders[i]['order'] != current_order_id:
                current_order_id = orders[i]['order']
                poi = []

            if len(poi)-1 > max_poi_count:
                max_poi_count = len(poi)

            orders[i]['poi'] = poi if len(poi) else []
            poi.append({'item': orders[i]['item'], 'cats': orders[i]['cats']})

        if max_poi_count > 4: # just to be on a safe side!
            max_poi_count = 4

        self._init_recommenders()
        if self._is_approach_used(self.available_approaches[0]):
            self.recommenders['oa'].set_train_data(max_poi_count, use_confidence = True, use_part_of_day = True)
        if self._is_approach_used(self.available_approaches[1]):
            self.recommenders['uh'].set_train_data(max_poi_count, use_confidence = True)
        if self._is_approach_used(self.available_approaches[3]):
            self.recommenders['tr'].set_train_data(True, True, False)

        if self._is_approach_used(self.available_approaches[1:3]):
            self.user_items = {}
            for ui in self.dm.get_user_items(None, 'train'):
                self.user_items[ui['user']] = ui['items']

        for order in orders:
            if self._is_approach_used(self.available_approaches[0]) and len(order['poi']):
                recommendations = self.recommenders['oa'].get_mem_recommendations(order['poi'], k, order['part_of_day'])
                self._test_item_against_recommendations(order['item'], recommendations, self.available_approaches[0], order['user'])

            if self._is_approach_used(self.available_approaches[1]) and len(self.user_items[order['user']]):
                recommendations = self.recommenders['uh'].get_mem_recommendations(order['user'], self.user_items[order['user']], k)
                self._test_item_against_recommendations(order['item'], recommendations, self.available_approaches[1], order['user'])

            if self._is_approach_used(self.available_approaches[2]) and len(self.user_items[order['user']]):
                recommendations = self.recommenders['uh2'].get_recommendations(order['user'], self.user_items[order['user']], k)
                self._test_item_against_recommendations(order['item'], recommendations, self.available_approaches[2], order['user'])

            if self._is_approach_used(self.available_approaches[3]):
                recommendations = self.recommenders['tr'].get_mem_recommendations(order['part_of_day'], order['day_in_week'], None, k)
                self._test_item_against_recommendations(order['item'], recommendations, self.available_approaches[3], order['user'])

        self._calculate_model()
        self.init_approaches_order()
        self.train_time = time.time() - start

    def recommend(self, k, order, previous_order_items = [], use_approach_offsets = True):
        recommendations = []

        # set the priority of the recommendations algorithms
        approaches_order = self.get_approaches_order_for_user(order['user'])

        # set the number of recommendation slots for each algorithm
        k_per_approach = self.get_k_per_approach(k, approaches_order)

        # iterate over approaches and populate recommendations list
        approach_index = 0
        user_items = self.dm.get_user_items(order['user'], 'train')

        for approach in approaches_order.keys():
            # approach requirements
            if approach is self.available_approaches[0] and len(previous_order_items) == 0:
                continue
            elif approach in self.available_approaches[1:3] and len(user_items) == 0:
                continue
            # recommendation generation
            if approach is self.available_approaches[0]:
                # approach_recommendations = self.recommenders['oa'].get_recommendations(previous_order_items, k, 1, order['part_of_day'], None, None, True, False, True)
                approach_recommendations = self.recommenders['oa'].get_mem_recommendations(previous_order_items, k, order['part_of_day'])
            elif approach is self.available_approaches[1]:
                # approach_recommendations = self.recommenders['uh'].get_recommendations(order['user'], user_items, k, 1, None, None, None, True, False, False)
                approach_recommendations = self.recommenders['uh'].get_mem_recommendations(order['user'], self.user_items[order['user']], k)
            elif approach is self.available_approaches[2]:
                approach_recommendations = self.recommenders['uh2'].get_recommendations(order['user'], self.user_items[order['user']], k)
            elif approach is self.available_approaches[3]:
                # approach_recommendations = self.recommenders['tr'].get_recommendations(order['part_of_day'], order['day_in_week'], None, k)
                approach_recommendations = self.recommenders['tr'].get_mem_recommendations(order['part_of_day'], order['day_in_week'], None, k)

            r_count = len(approach_recommendations)
            if r_count:
                recommendation_slot_index = 0
                for i in range(0, approach_index):
                    recommendation_slot_index += k_per_approach[i]

                # get recommendation offset for current approach
                slots_left = k_per_approach[approach_index]

                if use_approach_offsets:
                    offset = self._get_approach_offset('user', order['user'], approach, slots_left, r_count)
                else:
                    offset = 0

                # populate recommendation list
                # print offset
                for i in range(offset, r_count):
                    recommendations.insert(recommendation_slot_index, approach_recommendations[i])
                    recommendation_slot_index += 1
                    slots_left -= 1
                    if len(recommendations) >= k and slots_left <= 0:
                        break

            approach_index += 1

        return recommendations[:k]

    def _test_item_against_recommendations(self, item, recommendations, approach, user):
        result = recommendations.index(item) + 1 if item in recommendations else 0

        if item not in self.model['item'].keys():
            self.model['item'][item] = {}
        if approach not in self.model['item'][item]:
            self.model['item'][item][approach] = []
        self.model['item'][item][approach].append(result)

        if user not in self.model['user'].keys():
            self.model['user'][user] = {}
        if approach not in self.model['user'][user]:
            self.model['user'][user][approach] = []
        self.model['user'][user][approach].append(result)

        if approach not in self.model['global']:
            self.model['global'][approach] = []
        self.model['global'][approach].append(result)

    def _init_model(self):
        self.model = {
            'user': {},
            'item': {},
            'global': {},
            'rpr': {
                'item': [],
                'user': {'positive': [], 'negative': []}
            }
        }

    def _init_recommenders(self):
        self.recommenders = {}
        recommenders = [
            ('oa', OrderAssociationRecommender),
            ('uh', UserHistoryRecommender),
            ('uh2', UserHistory2Recommender),
            ('tr', TimeRelatedRecommender)
        ]
        for i in range(0, len(self.available_approaches)):
            if self._is_approach_used(self.available_approaches[i]):
                self.recommenders[recommenders[i][0]] = recommenders[i][1](self.min_support, self.min_confidence)
                self.recommenders[recommenders[i][0]].set_data_manager(self.dm)

    def _is_approach_used(self, approach):
        if type(approach) is list:
            for a in approach:
                if a in self.used_approaches:
                    return True
        elif approach in self.used_approaches:
            return True
        return False

    def set_used_approaches(self, used_approaches):
        if type(used_approaches) is list:
            self.used_approaches = []
            for ua in used_approaches:
                if type(ua) is str:
                    self.used_approaches.append(ua)
                elif type(ua) is tuple:
                    self.used_approaches.append(ua[0])
                    self.user_approaches_w[ua[0]] = ua[1]
        else:
            self.used_approaches = self.available_approaches

    def _calculate_model(self, calculate_rpr = False):
        def get_subject_metrics(hit_data):
            try:
                data = np.array(hit_data)
                n = len(data)
                h = np.divide(1.0, data)
                h[h == np.inf] = 0

                mcv = np.bincount(data).argmax()
                if mcv == 0:
                    mcv = 1
                return {'mcv': mcv, 'arhr': np.sum(h) / n}
            except ZeroDivisionError:
                return {'mcv': 1, 'arhr': 0}

        # calucate ARHR from values in model
        for subject in ['user', 'item']:
            for id_value in self.model[subject].keys():
                for approach in self.model[subject][id_value].keys():
                    self.model[subject][id_value][approach] = get_subject_metrics(self.model[subject][id_value][approach])
                    if approach in self.user_approaches_w:
                        self.model[subject][id_value][approach]['arhr'] *= self.user_approaches_w[approach]

        for approach in self.model['global'].keys():
            self.model['global'][approach] = get_subject_metrics(self.model['global'][approach])
            if approach in self.user_approaches_w:
                self.model['global'][approach]['arhr'] *= self.user_approaches_w[approach]

        # purchase rates
        if calculate_rpr:
            user_rpr_positive = np.array(self.model['rpr']['user']['positive'])
            user_rpr_negative = np.array(self.model['rpr']['user']['negative'])
            item_rpr = np.array(self.model['rpr']['item'])

            self.max_user_rpr = np.percentile(user_rpr_positive, 75)
            self.max_item_rpr = np.percentile(item_rpr, 75)

    def _update_model(self, subject, approach, result = 0, id_value = ''):
        if subject is 'global':
            if approach not in self.model[subject]:
                self.model[subject][approach] = []
            self.model[subject][approach].append(result)
        else:
            if id_value not in self.model[subject]:
                self.model[subject][id_value] = {}

            if approach not in self.model[subject][id_value]:
                self.model[subject][id_value][approach] = []

            self.model[subject][id_value][approach].append(result)

    def _get_approach_offset(self, subject, user, approach, slots_left, recommendations_count):
        if recommendations_count <= slots_left:
            return 0
        else:
            try:
                return self.model[subject][user][approach]['mcv'] - 1
            except Exception:
                return self.model['global'][approach]['mcv'] - 1

    def init_approaches_order(self):
        self.approaches_order = OrderedDict()
        for approach in sorted(self.model['global'].items(), key=itemgetter(1), reverse=True):
            self.approaches_order[approach[0]] = approach[1]['arhr']

    def update_rpr_from_recommendations(order, recommendations):
        if len(recommendations):
            user_rpr = self.dm.get_user_rpr(order['user'], 'train')
            if order['item'] in recommendations:
                rpr = self.dm.get_rpr(order['item'], 'train')
                self.model['rpr']['user']['positive'].append(user_rpr)
                self.model['rpr']['item'].append(rpr)
            else:
                self.model['rpr']['user']['negative'].append(user_rpr)

    def get_approaches_order(self, excluded = None):
        if excluded is None or len(excluded) == 0:
            return self.approaches_order
        elif len(excluded) == len(self.approaches_order):
            return {}
        else:
            approaches = excluded
            for approach_key in self.approaches_order.keys():
                if approach_key in approaches.keys():
                    approaches[approach_key] += self.approaches_order[approach_key]
                else:
                    approaches[approach_key] = self.approaches_order[approach_key]
            return approaches

    def get_approaches_order_for_user(self, user):
        if user in self.model['user']:
            approaches_order = OrderedDict()
            for approach in sorted(self.model['user'][user].items(), key=itemgetter(1), reverse=True):
                if approach[1] >= self.minimal_arhr_value:
                    approaches_order[approach[0]] = approach[1]['arhr']
            approaches_order.update(self.get_approaches_order(approaches_order))
            return approaches_order

        return self.get_approaches_order()

    def get_k_per_approach(self, k, approaches_order):
        k_per_approach = []
        k_left = k
        for approach_weight in approaches_order.values():
            if k_left > 0 and approach_weight > 0:
                k_for_approach = int(ceil( (approach_weight / sum(approaches_order.values())) * k ))
                if k_for_approach > k_left:
                    k_for_approach = k_left
                    k_left = 0
                else:
                    k_left -= k_for_approach
                k_per_approach.append(k_for_approach)
            else:
                k_per_approach.append(0)
        if k_left > 0:
            k_per_approach[0] += k_left

        return k_per_approach

    def get_train_time(self):
        return self.train_time

    def set_minimal_arhr(self, minimal_arhr):
        try:
            self.minimal_arhr_value = float(minimal_arhr)
        except (ValueError, TypeError):
            self.minimal_arhr_value = 0
