from association import AssociationRecommender
from operator import itemgetter

class OrderAssociationRecommender(AssociationRecommender):

    def get_recommendations(self, previous_order_items, k, degree = 1, part_of_day = None, day_in_week = None, month = None, use_confidence = False, use_lift = False, search_for_n_itemset = False):
        if previous_order_items is None:
            return []

        item_ids = [poi['item'] for poi in previous_order_items]
        return self.get_association_recommendations(item_ids, k, degree, part_of_day, day_in_week, month, use_confidence, use_lift, search_for_n_itemset)

    def set_train_data(self, max_x_count = 2, use_part_of_day = False, use_day_in_week = False, use_month = False, use_confidence = False):
        self.rules = self.dm.get_association_rules(self.min_support, max_x_count, use_part_of_day, use_day_in_week, use_month, use_confidence)

        # delete previous association relationships
        self.dm.delete_associations()

        if use_confidence:
            self.rules = sorted(self.rules, key = itemgetter('y', 'support', 'confidence'), reverse=True)
        else:
            self.rules = sorted(self.rules, key = itemgetter('y', 'support'), reverse=True)

        # write to db
        self.dm.write_associations(self.rules)

    def get_mem_recommendations(self, previous_order_items, k, part_of_day = None, day_in_week = None, month = None):
        poi = []
        for i in previous_order_items:
            poi.append(i['item'])

        recommendations = []
        for rule in self.rules:
            if any(i in rule['x'] for i in poi):
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
