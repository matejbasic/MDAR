# -*- coding: utf-8 -*-

import hashlib
from itertools import combinations
from operator import itemgetter
from py2neo import Relationship, Node
from mdar.query_manager import QueryManager


class DataManager(QueryManager):
    """inherits QueryManager, it's used for fetching data from database and
    transforming it into appropriate format for further usage.

    Args:
        config_path(string): path to a config.json file.
        k_fold_size(int, optional): number of data partitions. Defaults to 3.
    """
    def get_orders(self, data_type='all'):
        """Return all orders in defined data partition.

        Args:
            data_type(string, optional): 'train', 'test', or 'all' which is default.

        Returns:
            list: contains dicts with the following structure:
                {
                    'user': int
                    'order': int
                    'item': int,
                    'cats': list of category IDs(int)
                    'timestamp': string
                    'day_in_week': string
                    'part_of_day': string
                    'month': int
                }
        """
        match = (
            '(tf:TIME_FRAME)<-[:CREATED_AT]-(o:ORDER)-[cr:CONTAINS]->(p:PRODUCT)'
            + '-[df:DEFINED]->(c:CAT), (o)<-[pr:PURCHASED]-(u:USER)')

        return_values = (
            'u.oid AS user, o.oid AS order, p.oid AS item, collect(c.oid) AS cats, '
            + 'tf.timestamp AS timestamp, tf.day_in_week AS day_in_week, '
            + 'tf.part_of_day AS part_of_day, tf.month AS month ORDER BY timestamp',)

        return self._query_db(match, return_values, None, data_type)

    def get_associated_items(self, items_x, part_of_day=None, day_in_week=None, \
        month=None, search_for_n_itemset=False, data_type='all'):
        """Get items associated with given items as being part of the same rule
        as a body of that rule.

        Args:
            items_x(list): contains item IDs(int)
            part_of_day(string, optional)
            day_in_week(string, optional)
            month(int, optional)
            search_for_n_itemset(bool, optional): should the method generate
            association rules with more than one item in the body. Defaults to False.
            data_type(string, optional): 'train', 'test', or 'all' which is default.

        Returns:
            list: contains dicts with the following structure:
                {
                    'x': list of IDs(int)
                    'y': int
                    'support': float
                    'confidence': float
                }
        """
        orders_count = self.get_orders_count(data_type)
        return_values = (
            'p.oid AS item_x, p1.oid AS item, toFloat(count(o)/%f) AS support' % orders_count
            + ' ORDER BY support DESC')

        if not search_for_n_itemset:
            items_list = self._list_to_string(items_x)

            where = 'p.oid IN %s AND NOT p1.oid IN %s' % (items_list, items_list)
            time_constraints = self._get_time_constraints(part_of_day, day_in_week, month)
            if time_constraints:
                where += ' AND %s' % time_constraints

            match = (
                '(p:PRODUCT)<-[:CONTAINS]-(o:ORDER)-[:CONTAINS]->(p1:PRODUCT),'
                + ' (o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)')

            connected_items = self._query_db(match, return_values, where, data_type)
            return self._append_x_support(items_x, connected_items, orders_count, data_type)
        else:
            max_combination_length = len(items_x) + 1
            if max_combination_length > 3:
                max_combination_length = 3

            item_combinations = sum([map(list, combinations(items_x, i)) \
                for i in range(1, max_combination_length)], [])

            connected_items = []
            unprosperous_items = []
            for item_combination in item_combinations:
                is_unprosperous = False
                for i in item_combination:
                    if [i] in unprosperous_items:
                        is_unprosperous = True
                        break
                if is_unprosperous:
                    continue
                new_connected_items = self._get_connected_items(
                    item_combination, orders_count, data_type)
                if new_connected_items:
                    connected_items += new_connected_items
                else:
                    unprosperous_items.append(item_combination)

            connected_items_all = []

            for connected_item in sorted(connected_items, \
                key=itemgetter('support'), reverse=True):
                is_in_list = False
                for connected_item_all in connected_items_all:
                    if connected_item['item'] == connected_item_all['item']:
                        is_in_list = True
                        break
                if not is_in_list:
                    connected_items_all.append(connected_item)

            return connected_items_all

    def get_association_rules(self, min_support, max_x_count=2, \
        use_part_of_day=False, use_day_in_week=False, use_month=False, \
        use_confidence=False, data_type='train'):
        """Generates and returns a list of association rules with all the enabled
        measures with all the time attributes that are enabled via method args.

        Args:
            min_support(float, optional): Minimum support for a rule to be accepted.
            max_x_count(int, optional): Maximum number of items in rule's body.
            Defaults to 2.
            use_part_of_day(bool, optional): defaults to False.
            use_day_in_week(bool, optional): defaults to False.
            use_month(bool, optional): defaults to False.
            use_confidence(bool, optional): defaults to False.
            data_type(string, optional): 'train', 'test', or 'all' which is default.

        Returns:
            list: contains dicts with the following structure:
                {
                    'x': list of IDs(int)
                    'y': int
                    'support': float
                    'confidence': float
                }
        """
        rules = []
        orders_count = self.get_orders_count(data_type)

        for x_count in range(1, max_x_count + 1):
            match, where, return_values = self._get_association_rules_query_clauses(
                x_count, orders_count, use_part_of_day,
                use_day_in_week, use_month)

            current_rules = self._query_db(match, return_values, where, data_type)
            if current_rules[0]['support'] >= min_support:
                if use_confidence:
                    match = '(p0:PRODUCT)<-[:CONTAINS]-(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)'
                    return_values = '[p0.oid'
                    where = 'p0.oid' if x_count > 1 else ''

                    for i in range(1, x_count):
                        match += ', (o:ORDER)-[:CONTAINS]->(p%d:PRODUCT)' % i
                        where += '<>p%d.oid' % i
                        return_values += ', p%d.oid' % i

                    return_values += '] AS x, '
                    return_values += self._get_tf_props(use_part_of_day, use_day_in_week, use_month)
                    return_values += (
                        'toFloat(count(o)/%f) AS support_x ORDER BY support_x DESC'
                        % orders_count)

                    items_x = self._query_db(match, return_values, where, data_type)
                    for j in range(0, len(current_rules)):
                        current_rules[j]['confidence'] = \
                            current_rules[j]['support'] / (item['support_x'] \
                            for item in items_x if item['x'] == current_rules[j]['x']).next()

                rules += current_rules

            if current_rules[len(current_rules) - 1]['support'] < min_support:
                break

        return rules

    def _get_association_rules_query_clauses(self, x_count, orders_count, \
        use_part_of_day, use_day_in_week, use_month):
        """Generate MATCH, WHERE and RETURN parts of Cypher query for obtaining
        association rules.

        Args:
            x_count(int): number of items in association rule's body.
            orders_count(int)
            use_part_of_day(bool, optional): defaults to False.
            use_day_in_week(bool, optional): defaults to False.
            use_month(bool, optional): defaults to False.

        Returns:
            string: MATCH part
            string: WHERE part
            string: RETURN values
        """
        match = '(p:PRODUCT)<-[:CONTAINS]-(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)'
        p_oid_list = '['
        p_diff = 'p.oid'
        p_lex_order = ''

        for i in range(0, x_count):
            match += ', (o:ORDER)-[:CONTAINS]->(p%d:PRODUCT)' % i
            if i > 0:
                p_oid_list += ','
            p_oid = 'p%d.oid' % i
            p_oid_list += p_oid
            p_diff += '<>%s' % p_oid

            if x_count > 1:
                p_lex_order += p_oid
                if i < x_count - 1:
                    p_lex_order += '<'

        p_oid_list += ']'
        where = 'NOT p.oid IN %s AND %s' % (p_oid_list, p_diff)
        if x_count > 1:
            where += ' AND %s' % p_lex_order

        return_values = '%s AS x, p.oid AS y, ' % p_oid_list
        return_values += self._get_tf_props(use_part_of_day, use_day_in_week, use_month)
        return_values += 'toFloat(count(o)/%f) AS support ORDER BY support DESC' % orders_count

        return match, where, return_values

    def _get_connected_items(self, items, orders_count, data_type):
        """Return items connected to a given items via association rules.
        Args:
            items_x(list): contains item IDs(int)
            orders_count(int)
            data_type(string): 'train', 'test' or 'all'
        Returns:
            list: contains dict with the following structure:
                {
                    'item': int
                    'item_x': int
                    'support': float
                    'support_x': float
                }
        """
        items_string = self._list_to_string(items)

        match = (
            '(p:PRODUCT)<-[:CONTAINS]-(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME),'
            + ' (o:ORDER)-[:CONTAINS]->(p1:PRODUCT)')
        where = 'p.oid IN %s AND NOT p1.oid IN %s' % (items_string, items_string)
        return_values = (
            '%s AS item_x, p1.oid AS item,' % items_string
            + 'toFloat(count(o.oid)/%d) AS support ORDER BY support DESC' % orders_count)

        connected_items = self._query_db(match, return_values, where, data_type)
        return self._append_x_support(items, connected_items, orders_count, data_type)

    def _append_x_support(self, items_x, items, orders_count, data_type):
        """Calcuate support for items_x and append it to items list.

        Args:
            items_x(list): contains item IDs(int)
            items(list): contains dicts with the following structure:
                {
                    'item': int
                    'item_x': int
                    'support': float
                }
        Returns:
            list: same as items arg, plus 'support_x' key/value in each dict.
        """
        items_x_support = self.get_support(items_x, orders_count, data_type)
        for i in range(0, len(items)):
            items[i]['support_x'] = items_x_support
        return items

    def delete_associations(self):
        """Delete all the association relationships in the graph database."""
        self.graph.data('MATCH ()-[r:ASSOCIATED]->() DELETE r')
        self.graph.data('MATCH ()-[r:GROUPED]->() DELETE r')

    def write_associations(self, rules, neighbourhood_size=20):
        """Write rules to the graph database as relationships between PRODUCT
        nodes with all the measure values.

        Args:
            rules(list): should contain dicts with following structure:
                {
                    'x': list of item IDs(int)
                    'y': int
                    'support': float
                    'confidence': float
                }
            neighbourhood_size(int, optional): maximum number of items connected
            to a single item. Defaults to 20.
        """
        batch = self.graph.begin()

        item_ids = set(map(itemgetter('y'), rules))
        item_nodes_generator = self.graph.find(
            label='PRODUCT', property_key='oid', property_value=item_ids)

        item_nodes = []
        for item_node in item_nodes_generator:
            item_nodes.append(item_node)

        y_current = Node()
        neighbourhood_counter = 0
        for rule in rules:
            if not y_current or y_current['oid'] != rule['y']:
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
                    batch.create(Relationship(
                        x_nodes[0], 'ASSOCIATED',
                        y_current, support=rule['support'],
                        confidence=rule['confidence'], single=True))
                else:
                    associated_id = str(y_current['oid'])
                    for x_node in x_nodes:
                        associated_id += str(x_node['oid'])
                    associated_id = hashlib.sha224(associated_id).hexdigest()

                    x_node_index = 0
                    for x_node_index in range(0, len(x_nodes) - 1):
                        batch.create(Relationship(
                            x_nodes[x_node_index], 'GROUPED', x_nodes[x_node_index + 1],
                            support=rule['support'],
                            confidence=rule['confidence'], rel_id=associated_id))

                    batch.create(Relationship(
                        x_nodes[x_node_index + 1], 'ASSOCIATED', y_current,
                        support=rule['support'], confidence=rule['confidence'],
                        rel_id=associated_id, single=False))

        batch.commit()

    def get_associations(self, items):
        """Return associated items with given one sorted by support and confidence.

        Args:
            items(list): list of item IDs(int)

        Returns:
            list: contains item IDs(int)
        """
        return_values = (
            'p1.oid AS item, a_rel.support AS support,'
            + ' a_rel.confidence AS confidence ORDER BY support, confidence')

        items_string = self._list_to_string(items)
        associated_items = self._query_db(
            '(tf:TIME_FRAME)<-[:CREATED_AT]-(p:PRODUCT)-[a_rel:ASSOCIATED]->(p1:PRODUCT)',
            return_values,
            'p.oid IN %s AND NOT p1.oid IN %s' % (items_string, items_string))

        associations = []
        for associated_item in associated_items:
            if associated_item['item'] not in associations:
                associations.append(associated_item['item'])

        return associations

    @staticmethod
    def _get_time_constraints(part_of_day=None, day_in_week=None, month=None):
        """Generate and return string of time attributes constraints which should
        be used in Cypher's WHERE clause.

        Args:
            part_of_day(string, optional)
            day_in_week(string, optional)
            month(int, optional)

        Returns:
            string
        """
        where = ''
        if part_of_day is not None:
            where = 'tf.part_of_day="%s"' % part_of_day
        if day_in_week is not None:
            if where:
                where += ' AND '
            where += 'tf.day_in_week="%s"' % day_in_week
        if month is not None:
            if where:
                where += ' AND '
            where += 'tf.month=%d' % month
        print where
        return where

    @staticmethod
    def append_confidence(items):
        """Calculate and append confidence to each dict of provided items list.

        Args:
            items(list): contains dicts following structure:
            {
                'item': int
                'item_x': int
                'support': float
                'support_x': float
            }

        Returns:
            list: contains dicts with the same structure as the items arg but now
            with the confidence(float)
        """
        for i in range(0, len(items)):
            items[i]['confidence'] = items[i]['support'] / items[i]['support_x']
        return items

    def append_lift(self, items, data_type='all'):
        """Calculate and append lift to each dict of provided items list.

        Args:
            items(list): contains dicts following structure:
            {
                'item': int
                'item_x': int
                'support': float
                'support_x': float
            }
            data_type(string, optional): 'train', 'test', or 'all' which is default.

        Returns:
            list: contains dicts with the same structure as the items arg but now
            with the lift(float)
        """
        orders_count = self.get_orders_count(data_type)
        for i in range(0, len(items)):
            support_y = self.get_support(items[i]['item'], orders_count, data_type)
            items[i]['lift'] = items[i]['support'] / (items[i]['support_x'] * support_y)
        return items

    @staticmethod
    def _get_tf_props(use_part_of_day, use_day_in_week, use_month):
        """Return a string of TIME_FRAME node properties which should be used in
        a Cypher's RETURN clause.

        Args:
            use_part_of_day(bool)
            use_day_in_week(bool)
            use_month(bool)

        Returns:
            string
        """
        tf_props = ''
        if use_part_of_day:
            tf_props += 'tf.part_of_day AS part_of_day, '
        if use_day_in_week:
            tf_props += 'tf.day_in_week AS day_in_week, '
        if use_month:
            tf_props += 'tf.month AS month, '
        return tf_props

    def get_support(self, item, orders_count, data_type='all'):
        """Return a support for an item or set of items.

        Args:
            item(list): if only one item, it can be an int.
            orders_count(int): total number of orders.
            data_type(string, optional): 'train', 'test', or 'all' which is default.

        Returns:
            float
        """
        if isinstance(item, list):
            where = 'p.oid IN %s' % self._list_to_string(item)
        elif isinstance(item, int):
            where = 'p.oid=%d' % item

        support = self._query_db(
            '(p:PRODUCT)<-[:CONTAINS]-(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)',
            'toFloat(count(o)/%f) AS support' % orders_count, where, data_type)

        return support[0]['support']

    def get_orders_count(self, data_type='all'):
        """ Return the number of orders in the system for given data type.

        Args:
            data_type(string, optional): 'train', 'test', or 'all' which is default.

        Returns:
            float
        """
        order = self._query_db(
            '(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)', 'count(o) AS orders_count',
            None, data_type)
        return float(order[0]['orders_count'])

    def get_items_count(self, data_type='all'):
        """ Return the number of items in the system for given data type.

        Args:
            data_type(string, optional): 'train', 'test', or 'all' which is default.

        Returns:
            int
        """
        item = self._query_db(
            '(p:PRODUCT)-[:CREATED_AT]->(tf:TIME_FRAME)', 'count(p) AS items_count',
            None, data_type)
        return item[0]['items_count']

    def get_user_items(self, user_id=None, data_type='all'):
        """Return items for the user if provided or all the items for each user
        in the system.

        Args:
            user_id(int, optional)
            data_type(string, optional): 'train', 'test', or 'all' which is default.

        Returns:
            list: if the user ID is provided, it contains dicts with the structure:
                {
                    'item': int
                    'num': int
                }
            otherwise:
                {
                    'user': int
                    'items': list of IDs(int)
                }
        """
        match = (
            '(u:USER)-[:PURCHASED]->(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)'
            + ', (o)-[:CONTAINS]->(p:PRODUCT)')

        if user_id is None:
            return self._query_db(
                match, 'u.oid AS user, collect(distinct p.oid) AS items ORDER BY user',
                None, data_type)

        return self._query_db(
            match, 'p.oid AS item, count(p.oid) AS num ORDER BY num DESC',
            'u.oid=%d' % user_id, data_type)

    def get_popular_items(self, orders_count=None, data_type='all'):
        """Return all the items in the system sorted by their support.

        Args:
            orders_count(int, optional): total number of orders.
            data_type(string, optional): 'train', 'test', or 'all' which is default.

        Returns:
            list: contains dicts with the following structure:
                {
                    'item': int
                    'support': float
                }
        """
        if orders_count is None:
            orders_count = float(self.get_orders_count(data_type))

        return self._query_db(
            '(p:PRODUCT)<-[:CONTAINS]-(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)',
            'p.oid AS item, toFloat(count(o)/%f) AS support ORDER BY support DESC' % orders_count,
            None, data_type)

    def get_items_by_time(self, part_of_day=None, day_in_week=None, month=None, data_type='all'):
        """Return items with their support for given time args.

        Args:
            part_of_day(string, optional)
            day_in_week(string, optional)
            month(int, optional)

        Returns:
            list: contains dicts with the following structure:
                {
                    'item': int
                    'support': float
                }
        """
        where = self._get_time_constraints(part_of_day, day_in_week, month)
        orders_count = self._query_db(
            '(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)', 'count(o) AS orders_count',
            where, data_type)
        orders_count = float(orders_count[0]['orders_count'])

        return self._query_db(
            '(p:PRODUCT)<-[:CONTAINS]-(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)',
            'p.oid AS item, toFloat(count(o)/%f) AS support ORDER BY support DESC' % orders_count,
            where, data_type)

    def get_all_items_by_time(self, use_part_of_day, use_day_in_week, use_month, data_type='all'):
        """Return item IDs segmented by the time attributes which are defined by
        given args.

        Args:
            use_part_of_day(bool)
            use_day_in_week(bool)
            use_month(bool)
            data_type(string, optional): 'train', 'test', or 'all' which is default.

        Returns:
            list: contains dicts with following structure
                {
                    'part_of_day': string
                    'day_in_week': string
                    'month': int
                    'items': list of IDs(int)
                }
        """
        return_values = ''
        if use_part_of_day:
            return_values += 'tf.part_of_day AS part_of_day, '
        if use_day_in_week:
            return_values += 'tf.day_in_week AS day_in_week, '
        if use_month:
            return_values += 'tf.month AS month, '
        return_values += (
            'collect(p.oid) AS items, count(p) AS items_count'
            + ' ORDER BY items_count DESC')

        return self._query_db(
            '(tf:TIME_FRAME)<-[:CREATED_AT]-(o:ORDER)-[:CONTAINS]->(p:PRODUCT)',
            return_values, None, data_type)

    def get_item_rpr(self, item_id=None, data_type='all'):
        """Return repeated purchase rate (RPR) for the given data type globally
        or for certain item if ID is provided.

        Args:
            item_id(int, optional)
            data_type(string, optional): 'train', 'test' or 'all' which is default.

        Returns:
            float
        """
        if item_id is None:
            where = None
        else:
            where = 'p.oid=%s' % item_id

        purchases_total = self._query_db(
            '(p:PRODUCT)-[c_r:CONTAINS]-(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)',
            'count(c_r) AS purchases_total', where, data_type)
        purchases_total = purchases_total[0]['purchases_total']
        if purchases_total == 0:
            return 0

        match = (
            '(u:USER)-[:PURCHASED]->(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)'
            + ', (o)-[:CONTAINS]->(p:PRODUCT)')

        rpt_items = self._query_db(
            match,
            'u.oid AS user, p.oid AS item, count(u.oid)-1 AS repeated_purchases',
            where, data_type)
        rpt = sum(item['repeated_purchases'] for item in rpt_items)

        return rpt / float(purchases_total)

    def get_user_rpr(self, user_id=None, data_type='all'):
        """Return repeated purchase rate (RPR) for the given data type globally
        or for certain user if ID is provided.

        Args:
            user_id(int, optional)
            data_type(string, optional): 'train', 'test' or 'all' which is default.

        Returns:
            float
        """
        if user_id is None:
            where = None
        else:
            where = 'u.oid=%d' % user_id

        match = (
            '(u:USER)-[:PURCHASED]->(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)'
            + ', (o)-[:CONTAINS]->(p:PRODUCT)')

        purchases_total = self._query_db(
            match, 'count(p) AS purchases_total', where, data_type)
        purchases_total = purchases_total[0]['purchases_total']
        if purchases_total == 0:
            return 0

        rpt_items = self._query_db(
            match, 'u.oid AS user, p.oid AS item, count(u.oid)-1 AS repeated_purchases',
            where, data_type)
        rpt = sum(item['repeated_purchases'] for item in rpt_items)

        return rpt / float(purchases_total)
