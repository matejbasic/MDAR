from base import BaseRecommender
from scipy.stats import itemfreq
from operator import itemgetter

class TimeRelatedRecommender(BaseRecommender):

    def __init__(self, min_support = None, min_confidence = None):
        if min_support is not None:
            self.set_min_support(min_support)
        if min_confidence is not None:
            self.set_min_confidence(min_confidence)

    def set_train_data(self, use_part_of_day = False, use_day_in_week = False, use_month = False):
        self.time_slices = self.dm.get_all_items_by_time(use_part_of_day, use_day_in_week, use_month, 'train')
        for i in range(0, len(self.time_slices)):
            sorted_items = []
            items_freq   = sorted(itemfreq(self.time_slices[i]['items']), key = itemgetter(1), reverse = True)
            for item_freq in items_freq:
                sorted_items.append(item_freq[0])

            self.time_slices[i]['items'] = sorted_items

        self.popular_items = self.dm.get_popular_items(None, 'train')

    def get_mem_recommendations(self, part_of_day, day_in_week, month, k):
        recommendations = []
        ts = []
        for ts in self.time_slices:
            if month is not None and month != ts['month']:
                continue
            if day_in_week is not None and day_in_week != ts['day_in_week']:
                continue
            if part_of_day is not None and part_of_day != ts['part_of_day']:
                continue

            if len(recommendations):
                for item in ts['items']:
                    if item not in recommendations:
                        recommendations.append(item)
                        if len(recommendations) >= k:
                            return recommendations
            else:
                recommendations = ts['items'][:k]

            if len(recommendations) >= k:
                return recommendations

        if len(recommendations) < k:
            for i in self.popular_items:
                recommendations.append(i['item'])
                if len(recommendations) == k:
                    break

        return recommendations

    def get_recommendations(self, part_of_day, day_in_week, month, k):
        items = self.dm.get_items_by_time(part_of_day, day_in_week, month, 'train')
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

    def get_popular_recommendations(self, k, orders_count = None):
        popular_items = self.dm.get_popular_items(orders_count, 'train')

        recommendations = []
        for i in popular_items:
            recommendations.append(i['item'])
            if len(recommendations) == k:
                break
        return recommendations
