from association import AssociationRecommender

class UserHistoryRecommender(AssociationRecommender):

    def get_recommendations(self, user, user_items, k, degree = 1, part_of_day = None, day_in_week = None, month = None, use_confidence = False, use_lift = False, search_for_n_itemset = False):
        items_ids = [ui['item'] for ui in user_items if ui['num'] > 1]
        return self.get_association_recommendations(items_ids, k, degree, part_of_day, day_in_week, month, use_confidence, use_lift, search_for_n_itemset)

    def set_train_data(self, max_x_count = 2, use_part_of_day = False, use_day_in_week = False, use_month = False, use_confidence = False):
        self.rules = self.dm.get_association_rules(self.min_support, max_x_count, use_part_of_day, use_day_in_week, use_month, use_confidence)

    def get_mem_recommendations(self, user, user_items, k, part_of_day = None, day_in_week = None, month = None):
        recommendations = []
        for rule in self.rules:
            if any(i in rule['x'] for i in user_items):
                if part_of_day != None and part_of_day != rule['part_of_day']:
                    break
                if day_in_week != None and day_in_week != rule['day_in_week']:
                    break
                if month != None and month != rule['month']:
                    break

                if rule['y'] not in recommendations:
                    recommendations.append(rule['y'])
                if k >= len(recommendations):
                    break
        return recommendations
