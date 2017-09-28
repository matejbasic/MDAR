import json
import hashlib
from itertools import combinations
from operator import itemgetter
from query_manager import QueryManager
from py2neo import Node, Relationship

class DataManager(QueryManager):

    def __init__(self, config_path = None, k_fold_size = 3, multiprocessing_friendly = False):
        if config_path is not None:
            super(DataManager, self).__init__(config_path, k_fold_size, multiprocessing_friendly)

    def get_orders(self, data_type = 'all'):
        return self._query_db(
            '(tf:TIME_FRAME)<-[:CREATED_AT]-(o:ORDER)-[cr:CONTAINS]->(p:PRODUCT)-[df:DEFINED]->(c:CAT), (o)<-[pr:PURCHASED]-(u:USER)',
            'u.oid AS user, o.oid AS order, p.oid AS item, collect(c.oid) AS cats, tf.timestamp AS timestamp, tf.day_in_week AS day_in_week, tf.part_of_day AS part_of_day, tf.month AS month ORDER BY timestamp',
            None,
            data_type
        )

    def get_time_constraints(self, part_of_day, day_in_week, month):
        where_stat = ''
        if part_of_day is not None:
            where_stat = 'tf.part_of_day="' + str(part_of_day) + '"'
        if day_in_week is not None:
            if len(where_stat):
                where_stat += ' AND '
            where_stat += 'tf.day_in_week="' + str(day_in_week) + '"'
        if month is not None:
            if len(where_stat):
                where_stat += ' AND '
            where_stat += 'tf.month=' + str(month)
        return where_stat

    def get_connected_items(self, item, part_of_day = None, day_in_week = None, month = None, search_for_n_itemset = False, data_type = 'all'):
        def get_items_by_multiple_items(items):
            match_stat  = '(p:PRODUCT)<-[:CONTAINS]-(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME), (o:ORDER)-[:CONTAINS]->(p1:PRODUCT)'

            items_list = self._list_to_string(item)
            where_stat = 'p.oid IN ' + items_list + ' AND NOT p1.oid IN' + items_list

            return_stat = items_list + ' AS item_x, p1.oid AS item, toFloat(count(o.oid)/' + str(orders_count) + ') AS support ORDER BY support DESC'

            items = self._query_db(match_stat, return_stat, where_stat, data_type)
            return append_items_xy_support(items)

        def append_items_xy_support(items):
            items_x_support = self.get_support(item, orders_count, data_type)
            for i in range(0, len(items)):
                items[i]['support_x'] = items_x_support
            return items

        orders_count = self.get_orders_count(data_type)
        return_stat  = 'p.oid AS item_x, p1.oid AS item, toFloat(count(o)/' + str(orders_count) + ') AS support ORDER BY support DESC'

        if type(item) is int:
            where_stat = 'p.oid=' + str(item)
            time_constraints = self.get_time_constraints(part_of_day, day_in_week, month)
            if len(time_constraints):
                where_stat += ' AND ' + time_constraints

            items = self._query_db(
                '(p:PRODUCT)<-[:CONTAINS]-(o:ORDER)-[:CONTAINS]->(p1:PRODUCT), (o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)',
                return_stat, where_stat, data_type
            )
            items = append_items_xy_support(items)
            return items
        elif type(item) is list:
            if not search_for_n_itemset:
                items_list = self._list_to_string(item)
                where_stat = 'p.oid IN ' + items_list + ' AND NOT p1.oid IN ' + items_list
                time_constraints = self.get_time_constraints(part_of_day, day_in_week, month)
                if len(time_constraints):
                    where_stat += ' AND ' + time_constraints

                connected_items = self._query_db(
                    '(p:PRODUCT)<-[:CONTAINS]-(o:ORDER)-[:CONTAINS]->(p1:PRODUCT), (o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)',
                    return_stat, where_stat, data_type
                )
                return append_items_xy_support(connected_items)
            else:
                max_combination_length = len(item) + 1
                if max_combination_length > 3:
                    max_combination_length = 3

                item_combinations = sum([map(list, combinations(item, i)) for i in range(1, max_combination_length)], [])

                connected_items    = []
                unprosperous_items = []
                for ic in item_combinations:
                    is_unprosperous = False
                    for i in ic:
                        if [i] in unprosperous_items:
                            is_unprosperous = True
                            break
                    if is_unprosperous:
                        continue
                    new_connected_items = get_items_by_multiple_items(ic)
                    if len(new_connected_items):
                        connected_items += new_connected_items
                        new_connected_items_found = True
                    else:
                        unprosperous_items.append(ic)

                connected_items_all = []
                for connected_item in sorted(connected_items, key = itemgetter('support'), reverse = True):
                    is_in_list = False
                    for connected_item_all in connected_items_all:
                        if connected_item['item'] == connected_item_all['item']:
                            is_in_list = True
                            break
                    if not is_in_list:
                        connected_items_all.append(connected_item)

                return connected_items_all

    def get_association_rules(self, min_support, max_x_count = 2, use_part_of_day = False, use_day_in_week = False, use_month = False, use_confidence = False, data_type = 'train'):
        rules = []
        orders_count = self.get_orders_count(data_type)

        for x_count in range(1, max_x_count + 1):
            match_stat  = '(p:PRODUCT)<-[:CONTAINS]-(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)'
            return_stat = ''
            p_oid_list  = '['
            p_diff      = 'p.oid'
            p_lex_order = ''

            for i in range(0, x_count):
                match_stat += ', (o:ORDER)-[:CONTAINS]->(p' + str(i) + ':PRODUCT)'
                if i > 0:
                    p_oid_list += ','
                p_oid = 'p' + str(i) + '.oid'
                p_oid_list += p_oid
                p_diff += '<>' + p_oid

                if x_count > 1:
                    p_lex_order += p_oid
                    if i < x_count-1:
                        p_lex_order += '<'

            p_oid_list += ']'

            where_stat = 'NOT p.oid IN ' + p_oid_list + ' AND ' + p_diff
            if x_count > 1:
                where_stat += ' AND ' + p_lex_order

            return_stat += p_oid_list + ' AS x, p.oid AS y, '
            return_stat += self._get_tf_props(use_part_of_day, use_day_in_week, use_month)
            return_stat += 'toFloat(count(o)/' + str(orders_count) + ') AS support ORDER BY support DESC'

            # print 'MATCH', match_stat, 'WHERE', where_stat, 'RETURN', return_stat

            current_rules = self._query_db(match_stat, return_stat, where_stat, data_type)
            if current_rules[0]['support'] >= min_support:
                if use_confidence:
                    match_stat  = '(p0:PRODUCT)<-[:CONTAINS]-(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)'
                    return_stat = '[p0.oid'
                    if x_count > 1:
                        where_stat = 'p0.oid'
                    else:
                        where_stat = ''

                    for i in range(1, x_count):
                        match_stat  += ', (o:ORDER)-[:CONTAINS]->(p' + str(i) + ':PRODUCT)'
                        where_stat  += '<>p' + str(i) + '.oid'
                        return_stat += ', p' + str(i) + '.oid'

                    return_stat += '] AS x, '
                    return_stat += self._get_tf_props(use_part_of_day, use_day_in_week, use_month)
                    return_stat += 'toFloat(count(o)/' + str(orders_count) + ') AS support_x ORDER BY support_x DESC'

                    items_x = self._query_db(match_stat, return_stat, where_stat, data_type)
                    for j in range(0, len(current_rules)):
                        current_rules[j]['confidence'] = current_rules[j]['support'] / (item['support_x'] for item in items_x if item['x'] == current_rules[j]['x']).next()

                rules += current_rules

            if current_rules[len(current_rules)-1]['support'] < min_support:
                break

        return rules


    def delete_associations(self):
        self.graph.data('MATCH ()-[r:ASSOCIATED]->() DELETE r')
        self.graph.data('MATCH ()-[r:GROUPED]->() DELETE r')

    def write_associations(self, rules, neighbourhood_size = 20):
        batch = self.graph.begin()

        item_ids       = set(map(itemgetter('y'), rules))
        item_nodes_gen = self.graph.find(label='PRODUCT', property_key='oid', property_value=item_ids)
        item_nodes     = []
        for item_node in item_nodes_gen:
            item_nodes.append(item_node)

        y_current = False
        neighbourhood_counter = 0
        for rule in rules:
            # obtain current Y node
            if y_current is False or y_current['oid'] != rule['y']:
                for item_node in item_nodes:
                    if item_node['oid'] == rule['y']:
                        y_current = item_node
                        neighbourhood_counter = 0
                        break

            if neighbourhood_counter >= neighbourhood_size:
                continue
            else:
                neighbourhood_counter += 1

            # obtain X node/s
            x_nodes = []
            for x_oid in rule['x']:
                for item_node in item_nodes:
                    if item_node['oid'] == x_oid:
                        x_nodes.append(item_node)

            x_nodes_count = len(x_nodes)
            if x_nodes_count:
                if x_nodes_count == 1:
                    batch.create(Relationship(x_nodes[0], 'ASSOCIATED', y_current, support = rule['support'], confidence = rule['confidence'], single = True))
                    pass
                else:
                    associated_id = str(y_current['oid'])
                    for x_node in x_nodes:
                        associated_id += str(x_node['oid'])
                    associated_id = hashlib.sha224(associated_id).hexdigest()
                    # print len(x_nodes), range(0, len(x_nodes) - 1)
                    for i in range(0, len(x_nodes) - 1):
                        batch.create(Relationship(x_nodes[i], 'GROUPED', x_nodes[i+1], support = rule['support'], confidence = rule['confidence'], rel_id = associated_id))
                    batch.create(Relationship(x_nodes[i+1], 'ASSOCIATED', y_current, support = rule['support'], confidence = rule['confidence'], rel_id = associated_id, single = False))

        batch.commit()

    def get_associations(self, items):
        items_list_string = self._list_to_string(items)
        ai = self._query_db(
            '(tf:TIME_FRAME)<-[:CREATED_AT]-(p:PRODUCT)-[a_rel:ASSOCIATED]->(p1:PRODUCT)',
            'p1.oid AS item, a_rel.support AS support, a_rel.confidence AS confidence ORDER BY support, confidence',
            'p.oid IN ' + items_list_string + ' AND NOT p1.oid IN ' + items_list_string,
        )
        associations = []
        for i in ai:
            if i['item'] not in associations:
                associations.append(i['item'])

        return associations

    def get_confidence(self, items, data_type = 'all'):
        return self._get_rules_attribute(items, 'confidence', data_type)

    def get_lift(self, items, data_type = 'all'):
        return self._get_rules_attribute(items, 'lift', data_type)

    def _get_rules_attribute(self, items, attribute, data_type = 'all'):
        if attribute == 'confidence':
            for i in range(0, len(items)):
                items[i][attribute] = items[i]['support'] / items[i]['support_x']
        elif attribute == 'lift':
            orders_count = self.get_orders_count(data_type)
            for i in range(0, len(items)):
                support_y = self.get_support(items[i]['item'], orders_count, data_type)
                items[i][attribute] = items[i]['support'] / (items[i]['support_x'] * support_y)

        return items

    def _get_tf_props(self, use_part_of_day, use_day_in_week, use_month):
        tc = ''
        if use_part_of_day:
            tc += 'tf.part_of_day AS part_of_day, '
        if use_day_in_week:
            tc += 'tf.day_in_week AS day_in_week, '
        if use_month:
            tc += 'tf.month AS month, '
        return tc

    def get_support(self, item, orders_count, data_type = 'all'):
        if type(item) is list:
            where_stat = 'p.oid IN ' + self._list_to_string(item)
        elif type(item) is int:
            where_stat = 'p.oid=' + str(item)
        s = self._query_db(
            '(p:PRODUCT)<-[:CONTAINS]-(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)',
            'toFloat(count(o)/' + str(orders_count) + ') AS support',
            where_stat,
            data_type
        )
        return s[0]['support']

    def get_orders_count(self, data_type = 'all'):
        order = self._query_db(
            '(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)',
            'count(o) AS orders_count',
            None,
            data_type
        )
        return float(order[0]['orders_count'])

    def get_items_count(self, data_type = 'all'):
        item = self._query_db(
            '(p:PRODUCT)-[:CREATED_AT]->(tf:TIME_FRAME)',
            'count(p) AS items_count',
            None,
            data_type
        )
        return item[0]['items_count']

    def get_user_items(self, user = None, data_type = 'all'):
        if user is None:
            return self._query_db(
                '(u:USER)-[:PURCHASED]->(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME), (o)-[:CONTAINS]->(p:PRODUCT)',
                'u.oid AS user, collect(distinct p.oid) AS items ORDER BY user',
                None,
                data_type
            )
        else:
            return self._query_db(
                '(u:USER)-[:PURCHASED]->(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME), (o)-[:CONTAINS]->(p:PRODUCT)',
                'p.oid AS item, count(p.oid) AS num ORDER BY num DESC',
                'u.oid=' + str(user),
                data_type
            )

    def get_popular_items(self, orders_count = None, data_type = 'all'):
        if orders_count is None:
            orders_count = float(self.get_orders_count(data_type))

        return self._query_db(
            '(p:PRODUCT)<-[:CONTAINS]-(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)',
            'p.oid AS item, toFloat(count(o)/' + str(orders_count) + ') AS support ORDER BY support DESC',
            None, data_type
        )

    def get_items_by_time(self, part_of_day = None, day_in_week = None, month = None, data_type = 'all'):
        where_stat = self.get_time_constraints(part_of_day, day_in_week, month)
        orders_count = self._query_db(
            '(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)',
            'count(o) AS orders_count',
            where_stat,
            data_type
        )
        orders_count = float(orders_count[0]['orders_count'])

        return self._query_db(
            '(p:PRODUCT)<-[:CONTAINS]-(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)',
            'p.oid AS item, toFloat(count(o)/' + str(orders_count) + ') AS support ORDER BY support DESC',
            where_stat, data_type
        )

    def get_all_items_by_time(self, use_part_of_day, use_day_in_week, use_month, data_type = 'all'):
        return_stat = ''
        if use_part_of_day:
            return_stat += 'tf.part_of_day AS part_of_day, '
        if use_day_in_week:
            return_stat += 'tf.day_in_week AS day_in_week, '
        if use_month:
            return_stat += 'tf.month AS month, '
        return_stat += 'collect(p.oid) AS items, count(p) AS items_count ORDER BY items_count DESC'

        return self._query_db(
            '(tf:TIME_FRAME)<-[:CREATED_AT]-(o:ORDER)-[:CONTAINS]->(p:PRODUCT)',
            return_stat,
            None,
            data_type
        )

    def get_item_rpr(self, item_id = None, data_type = 'all'):
        if item_id is None: # do all items, get global rate
            where_stat = None
        else:
            where_stat = 'p.oid=' + str(item_id)

        pt  = self._query_db(
            '(p:PRODUCT)-[c_r:CONTAINS]-(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)',
            'count(c_r) AS purchases_total', where_stat, data_type
        )
        rpt_items = self._query_db(
            '(u:USER)-[:PURCHASED]->(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME), (o)-[:CONTAINS]->(p:PRODUCT)',
            'u.oid AS user, p.oid AS item, count(u.oid)-1 AS repeated_purchases',
            where_stat,
            data_type
        )

        rpt = sum(item['repeated_purchases'] for item in rpt_items)
        pt  = pt[0]['purchases_total']
        if pt == 0:
            return 0
        else:
            return rpt / float(pt)

    def get_user_rpr(self, user_id = None, data_type = 'all'):
        if user_id is None:
            where_stat = None
        else:
            where_stat = 'u.oid=' + str(user_id)

        pt = self._query_db(
            '(u:USER)-[:PURCHASED]->(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME), (o)-[:CONTAINS]->(p:PRODUCT)',
            'count(p) AS purchases_total', where_stat, data_type
        )
        rpt_items = self._query_db(
            '(u:USER)-[:PURCHASED]->(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME), (o)-[:CONTAINS]->(p:PRODUCT)',
            'u.oid AS user, p.oid AS item, count(u.oid)-1 AS repeated_purchases',
            where_stat,
            data_type
        )

        rpt = sum(item['repeated_purchases'] for item in rpt_items)
        pt  = pt[0]['purchases_total']
        if pt == 0:
            return 0
        else:
            return rpt / float(pt)
