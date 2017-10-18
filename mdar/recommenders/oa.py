# -*- coding: utf-8 -*-

from operator import itemgetter
from mdar.recommenders.association import AssociationRecommender


class OrderAssociationRecommender(AssociationRecommender):
    """Recommender class which uses current cart items as the body rule items
    in association analysis which generates rules which are, at the end, a
    source for recommendations.

    Args:
        min_support(float, optional): minimal support for an association rule
        to be considered valid. Defaults to 0.02.
        min_confidence(float, optional): minimal confidence for an association
        rule to be considered valid. Defaults to 0.05.
    """
    _association_rules = []

    def get_recommendations(self, previous_order_items, k, degree=1, \
        part_of_day=None, day_in_week=None, month=None, \
        use_confidence=False, use_lift=False, search_for_n_itemset=False):
        """Return recommendations generated from the association rules which are
        based on given args.

        Args:
            previous_order_items(list): should contain dicts with 'item' key
            which holds item's ID(int)
            k(int): expected number of recommendations
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
            list: recommendations, contains item IDs (int)
        """
        if previous_order_items is None:
            return []

        item_ids = [poi['item'] for poi in previous_order_items]
        return self.get_association_recommendations(
            item_ids, k, degree, part_of_day, day_in_week, month,
            use_confidence, use_lift, search_for_n_itemset
        )

    def get_mem_recommendations(self, previous_order_items, k, \
        part_of_day=None, day_in_week=None, month=None):
        """Return recommendations generated from the association rules which are
        based on given args from the fast memory(prefetched), not directly from
        graph DB.

        Args:
            previous_order_items(list): should contain dicts with 'item' key
            which holds item's ID(int)
            k(int): number of expected recommendations
            part_of_day(string, optional): used as a constraint for assciation
            rules as other time-related args.
            day_in_week(string, optional)
            month(int, optional)

        Returns:
            list: recommendations, contains item IDs (int)
        """
        poi = []
        for i in previous_order_items:
            poi.append(i['item'])

        recommendations = []
        for rule in self.association_rules:
            if any(i in rule['x'] for i in poi):
                if part_of_day is not None and part_of_day != rule['part_of_day']:
                    break
                if day_in_week is not None and day_in_week != rule['day_in_week']:
                    break
                if month is not None and month != rule['month']:
                    break

                if rule['y'] not in recommendations:
                    recommendations.append(rule['y'])
                if k >= len(recommendations):
                    break
        return recommendations

    def set_train_data(self, max_x_count=2, use_part_of_day=False, \
        use_day_in_week=False, use_month=False, use_confidence=False):
        """Define association rules based on the given args.

        Args:
            max_x_count(int, optional): maximum number of items in rule body.
            Defaults to 2.
            use_part_of_day(bool, optional): defaults to False.
            use_day_in_week(bool, optional): defaults to False.
            use_month(bool, optional): defaults to False.
            use_confidence(bool, optional): if confidence measure should be
            used in generating and estimating rules. Defaults to False.
        """
        self.association_rules = self.data_manager.get_association_rules(
            self.min_support, max_x_count, use_part_of_day, use_day_in_week,
            use_month, use_confidence
        )

        # delete previous association relationships
        self.data_manager.delete_associations()
        if use_confidence:
            self.association_rules = sorted(
                self.association_rules,
                key=itemgetter('y', 'support', 'confidence'),
                reverse=True
            )
        else:
            self.association_rules = sorted(
                self.association_rules,
                key=itemgetter('y', 'support'),
                reverse=True
            )

        # write to db
        self.data_manager.write_associations(self.association_rules)

    @property
    def association_rules(self):
        """list: contains dicts with following structure:
        {
            'x': list
            'y': list (of int) or int
            'support': float
            'confidence': float
        }
        """
        return self._association_rules

    @association_rules.setter
    def association_rules(self, value):
        if isinstance(value, list):
            self._association_rules = value
