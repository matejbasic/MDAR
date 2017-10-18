# -*- coding: utf-8 -*-

from operator import itemgetter
from mdar.recommenders.base import BaseRecommender


class AssociationRecommender(BaseRecommender):
    """Recommender class that uses association analysis for generating rules
    with the given set of items which are used as a source for recommendations.

    Args:
        min_support(float, optional): minimal support for an association rule
        to be considered valid. Defaults to 0.02.
        min_confidence(float, optional): minimal confidence for an association
        rule to be considered valid. Defaults to 0.05.
    """

    def get_association_recommendations(self, item_ids, k, degree=1, \
        part_of_day=None, day_in_week=None, month=None, use_confidence=False, \
        use_lift=False, search_for_n_itemset=False):
        """Return recommendations generated from the association rules which are
        based on given args.

        Args:
            item_ids(list): contains item IDs(int)
            k(int): number of expected recommendations
            degree(int, optional): degree to which graph is mined for
            associations. Defaults to 1.
            part_of_day(string, optional): used as a constraint for assciation
            rules as other time-related args.
            day_in_week(string, optional)
            month(int, optional)
            use_confidence(bool, optional): if confidence measure should be
            used in generating and estimating assocation rules. Defaults to False.
            use_lift(bool, optional): Defaults to False.
            search_for_n_itemset(bool, optional): should algorithm generate
            rules with more than one item in rule head. Defaults to False.

        Returns:
            list
        """
        recommendations = []
        global_candidates = []
        if not item_ids:
            return []

        for current_degree in (1, degree):
            candidates = self.data_manager.get_associated_items(
                item_ids, part_of_day, day_in_week, month,
                search_for_n_itemset, 'train')

            if not candidates:
                continue
            if use_confidence:
                candidates = self._append_confidence_values(candidates)
            if use_lift:
                candidates = self._apppend_lift_values(candidates)

            if degree == 1:
                recommendations = self._update_recommendations(
                    recommendations, candidates, k, True)
            else:
                item_ids = []
                for candidate in candidates:
                    if candidate['support'] < self.min_support * current_degree:
                        break

                    if candidate['item'] not in item_ids:
                        item_ids.append(candidate['item'])

                    global_candidates = self._update_items(
                        global_candidates, candidate
                    )

        return self._update_recommendations(
            recommendations, global_candidates, k, False,
            self._get_sorting_key(use_confidence, use_lift)
        )

    def _update_recommendations(self, recommendations, items, k, use_min_support, sorting_key=None):
        """Append items to recommendations until length of recommendations is k.

        Args:
            recommendations(list): contains item IDs(int)
            items(list): list of dicts with the following structure:
                {
                    'item': int
                    'support': float
                    'item_x': int or list of ints
                    'support_x': float
                }
            k(int): maximum allowed length of recommendations
            use_min_support(bool): should the items be tested against minimum
            support.
            sorting_key(itemgetter)
        Returns:
            list
        """
        if items:
            if sorting_key:
                items = sorted(items, key=sorting_key, reverse=True)
            for item in items:
                if use_min_support and item['support'] < self.min_support:
                    break
                if len(recommendations) == k:
                    break
                recommendations.append(item['item'])

        return recommendations

    @staticmethod
    def _update_items(items, item):
        """Append new item to a items list or, if the new item already exists
        in the list, update it's support.

        Args:
            items(list): contains dicts with following structure:
                {
                    'item': int
                    'support': float
                    'item_x': int or list of ints
                    'support_x': float
                }
            item(dict): with the same structure as the one in the previous list
            (items).

        Returns:
            list
        """
        is_in_items = False
        for i in range(0, len(items)):
            if items[i]['item'] == item['item']:
                is_in_items = True
                if item['support'] > items[i]['support']:
                    items[i] = item
                break

        if not is_in_items:
            items.append(item)

        return items

    @staticmethod
    def _get_sorting_key(use_confidence, use_lift):
        """Return appropriate itemgetter for sorting.
        Args:
            use_confidence(bool)
            use_lift(bool)

        Returns:
            itemgetter
        """
        if use_lift and use_confidence:
            return itemgetter('support', 'confidence', 'lift')
        elif use_lift:
            return itemgetter('support', 'lift')
        elif use_confidence:
            return itemgetter('support', 'confidence')

        return itemgetter('support')

    def _append_confidence_values(self, items):
        """Calcuate confidence for each item in items and append it to same list.

        Args:
            items(list): contains dicts with following structure:
            {
                'item': int
                'support': float
                'item_x': int or list of ints
                'support_x': float
            }

        Returns:
            list: same as 'items' arg but now with 'confidence' in each dict
        """
        items = self.data_manager.append_confidence(items)
        items = sorted(items, key=itemgetter('support', 'confidence'), reverse=True)
        items = [item for item in items if item['confidence'] >= self.min_confidence]

        return items

    def _apppend_lift_values(self, items):
        """Calcuate lift for each item in items and append it to same list.

        Args:
            items(list): contains dicts with following structure:
            {
                'item': int
                'support': float
                'item_x': int or list of ints
                'support_x': float
            }

        Returns:
            list: same as 'items' arg but now with 'lift' in each dict
        """
        items = self.data_manager.append_lift(items, 'train')
        items = sorted(items, key=itemgetter('support', 'lift'), reverse=True)
        items = [item for item in items if item['lift'] >= self.min_lift]

        return items
