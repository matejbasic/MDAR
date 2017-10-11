# -*- coding: utf-8 -*-

from operator import itemgetter
from scipy.stats import itemfreq
from mdar.recommenders.base import BaseRecommender


class TimeRelatedRecommender(BaseRecommender):
    """TimeRelatedRecommender returns recommendations based on given time
    constraints. Fallbacks on global popular items if none.

    Args:
        min_support(float, optional): minimal support for an association rule
        to be considered valid. Defaults to 0.02.
        min_confidence(float, optional): minimal confidence for an association
        rule to be considered valid. Defaults to 0.05.
    """

    _popular_items = []
    _time_related_items = []

    def get_recommendations(self, part_of_day, day_in_week, month, k):
        """Return time related recommendations for given parameters.

        Args:
            part_of_day(string)
            day_in_week(string)
            month(int)
            k(int)

        Returns:
            list: recommendations, contains item IDs (int)
        """

        items = self.data_manager.get_items_by_time(part_of_day, day_in_week, month, 'train')
        recommendations = []

        for i in items:
            if i['support'] < self.min_support or len(recommendations) == k:
                break
            recommendations.append(i['item'])

        # fallback on popular items recommender
        if len(recommendations) < k:
            items = self.get_popular_recommendations(k - len(recommendations))
            recommendations += items

        return recommendations

    def get_mem_recommendations(self, part_of_day, day_in_week, month, k):
        """Return time related recommendations for given parameters from fast
        memory(prefetched), not directly from graph DB.

        Args:
            part_of_day(string)
            day_in_week(string)
            month(int)
            k(int)

        Returns:
            list: k recommendations, contains item IDs (int).
        """

        recommendations = []
        for time_slice in self.time_related_items:
            if month is not None and month != time_slice['month']:
                continue
            if day_in_week is not None and day_in_week != time_slice['day_in_week']:
                continue
            if part_of_day is not None and part_of_day != time_slice['part_of_day']:
                continue

            if recommendations:
                for item in time_slice['items']:
                    if item not in recommendations:
                        recommendations.append(item)
                        if len(recommendations) >= k:
                            return recommendations
            else:
                recommendations = time_slice['items'][:k]

            if len(recommendations) >= k:
                return recommendations

        if len(recommendations) < k:
            for i in self.popular_items:
                recommendations.append(i['item'])
                if len(recommendations) == k:
                    break

        return recommendations

    def get_popular_recommendations(self, k):
        """Return k most popular items in the system.

        Args:
            k(int)

        Returns:
            list: contain k items(int).
        """

        popular_items = self.data_manager.get_popular_items(None, 'train')
        recommendations = []
        for i in popular_items:
            recommendations.append(i['item'])
            if len(recommendations) == k:
                break

        return recommendations

    def set_train_data(self, use_part_of_day=False, use_day_in_week=False, use_month=False):
        """Define items related with given time attributes and globally popular
        items which are used as a fallback.

        Args:
            use_part_of_day(bool, optional): defaults to False.
            use_day_in_week(bool, optional): defaults to False.
            use_month(bool, optional): defaults to False.
        """
        self.time_related_items = self.data_manager.get_all_items_by_time(
            use_part_of_day,
            use_day_in_week,
            use_month,
            'train'
        )

        for i in range(0, len(self.time_related_items)):
            sorted_items = []
            items_freq = sorted(
                itemfreq(self.time_related_items[i]['items']),
                key=itemgetter(1),
                reverse=True
            )
            for item_freq in items_freq:
                sorted_items.append(item_freq[0])

            self.time_related_items[i]['items'] = sorted_items

        self.popular_items = self.data_manager.get_popular_items(None, 'train')

    @property
    def time_related_items(self):
        """list: consists of dicts with following structure
        {
            'day_in_week': string, optional
            'part_of_day': string, optional
            'month': int, optional
            'items': list of ints (product IDs related with time values in dict)
            'items_count': int
        }
        """
        return self._time_related_items

    @time_related_items.setter
    def time_related_items(self, value):
        if isinstance(value, list):
            self._time_related_items = value

    @property
    def popular_items(self):
        """list: consists of dicts with following structure
        {
            'item': int (item ID)
            'support': float
        }
        """
        return self._popular_items

    @popular_items.setter
    def popular_items(self, value):
        if isinstance(value, list):
            self._popular_items = value
