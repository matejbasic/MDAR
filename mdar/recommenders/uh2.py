from base import BaseRecommender

class UserHistory2Recommender(BaseRecommender):

    def __init__(self, min_support = None, min_confidence = None):
        if min_support is not None:
            self.set_min_support(min_support)
        if min_confidence is not None:
            self.set_min_confidence(min_confidence)

    def get_recommendations(self, user, user_items, k, use_user_rpr = False, use_item_rpr = False, max_user_rpr = None, max_item_rpr = None):
        recommendations = []
        if use_user_rpr:
            user_rpr = self.dm.get_user_rpr(user, 'train')
            if user_rpr > max_user_rpr:
                return recommendations

        for item in user_items:
            if len(recommendations) == k:
                break
            if type(item) is dict:
                item = item['item']
            if use_item_rpr:
                rpr = self.dm.get_rpr(item, 'train')
                if rpr > max_item_rpr:
                    continue
            recommendations.append(item)

        return recommendations
