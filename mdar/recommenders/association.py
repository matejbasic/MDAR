from base import BaseRecommender
from operator import itemgetter

class AssociationRecommender(BaseRecommender):

    def __init__(self, min_support = None, min_confidence = None):
        if min_support is not None:
            self.set_min_support(min_support)
        if min_confidence is not None:
            self.set_min_confidence(min_confidence)

    def get_association_recommendations(self, item_ids, k, degree = 1, part_of_day = None, day_in_week = None, month = None, use_confidence = False, use_lift = False, search_for_n_itemset = False):
        recommendations   = []
        global_candidates = []

        if len(item_ids) > 0:
            for current_degree in (1, degree):
                candidates = self.dm.get_connected_items(item_ids, part_of_day, day_in_week, month, search_for_n_itemset, 'train')
                if len(candidates) > 0:
                    if use_confidence:
                        candidates = self.dm.get_confidence(candidates, 'train')
                        candidates = sorted(candidates, key=itemgetter('support', 'confidence'), reverse = True)
                        candidates = [candidate for candidate in candidates if candidate['confidence'] >= self.min_confidence]

                    if use_lift:
                        candidates = self.dm.get_lift(candidates, 'train')
                        candidates = sorted(candidates, key=itemgetter('support', 'lift'), reverse = True)
                        candidates = [candidate for candidate in candidates if candidate['lift'] >= self.minimal_lift]

                    if degree == 1:
                        for candidate in candidates:
                            if candidate['support'] < self.min_support or len(recommendations) == k:
                                break
                            if candidate['item'] not in recommendations:
                                recommendations.append(candidate['item'])
                    else:
                        item_ids = []
                        for candidate in candidates:
                            if candidate['support'] < self.min_support * current_degree:
                                break

                            if candidate['item'] not in item_ids:
                                item_ids.append(candidate['item'])

                            is_candidate_global = False
                            for i in range(0, len(global_candidates)):
                                if global_candidates[i]['item'] == candidate['item']:
                                    is_candidate_global = True
                                    if candidate['support'] > global_candidates[i]['support']:
                                        global_candidates[i] = candidate
                                    break

                            if not is_candidate_global:
                                global_candidates.append(candidate)

        if len(global_candidates) and degree > 1:
            if use_lift and use_confidence:
                key = itemgetter('support', 'confidence', 'lift')
            elif use_lift:
                key = itemgetter('support', 'lift')
            elif use_confidence:
                key = itemgetter('support', 'confidence')
            else:
                key = itemgetter('support')

            global_candidates = sorted(global_candidates, key=key, reverse = True)
            for gc in global_candidates:
                if len(recommendations) == k:
                    break
                recommendations.append(gc['item'])

        return recommendations
