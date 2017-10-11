# -*- coding: utf-8 -*-

from mdar.recommenders.base import BaseRecommender


class UserHistory2Recommender(BaseRecommender):
    """Most simple recommender which basically uses already known items. Usually
    not really efficient.

    Args:
        min_support(float, optional): minimal support for an association rule
        to be considered valid. Defaults to 0.02.
        min_confidence(float, optional): minimal confidence for an association
        rule to be considered valid. Defaults to 0.05.
    """

    def get_recommendations(self, user_id, user_items, k, use_user_rpr=False, \
        use_item_rpr=False, min_user_rpr=None, min_item_rpr=None):
        """Return recommendations for given arguments.

        Args:
            user_id(int)
            user_items(list): contains item IDs which are purchased before.
            k(int)
            use_user_rpr(bool, optional): if True, method checks if user's
            repeat purchase rate (RPR) is greater than minimum. If no, method
            returns empty list of recommendations. Defaults to False.
            use_item_rpr(bool, optional): if True, method checks for each
            known item if it has RPR greater than given minimum. If not, items
            is discarded. Defaults to False.
            min_user_rpr(float, optional)
            min_item_rpr(float, optional)

        Returns:
            list: recommendations, contains item IDs (int)
        """

        recommendations = []
        if use_user_rpr:
            user_rpr = self.data_manager.get_user_rpr(user_id, 'train')
            if user_rpr < min_user_rpr:
                return []

        for item in user_items:
            if len(recommendations) == k:
                break
            if isinstance(item, dict):
                item = item['item']
            if use_item_rpr:
                rpr = self.data_manager.get_rpr(item, 'train')
                if rpr < min_item_rpr:
                    continue
            recommendations.append(item)

        return recommendations
