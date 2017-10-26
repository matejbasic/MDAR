# -*- coding: utf-8 -*-

import json
from py2neo import authenticate, Graph


class QueryManager(object):
    """Used for communicating with Neo4j graph database, constructing TIME_FRAME
    nodes constraints for test and train dataset parts (k-fold cross validation),
    and Cypher query building.

    Args:
        config_path(string): path to a config.json file.
        k_fold_size(int, optional): number of data partitions. Defaults to 3.
    """
    _k_fold_size = 3

    k_fold_tfs = None
    tf_conditions = None
    _testing_part_index = 0

    def __init__(self, config_path=None, k_fold_size=3):
        if config_path is not None:
            self.set_graph(config_path)

        self.set_k_fold_tfs(k_fold_size)
        self.k_fold_size = k_fold_size

    def set_graph(self, config_path):
        """Define graph instance with data from config file.

        Args:
            config_path(string): path to a config.json file.

        Returns:
            Graph or None if failed to define.
        """
        self.graph = None
        with open(config_path) as config_data:
            config = json.load(config_data)
            host = config['host']

            if host['use_ssl']:
                db_url = 'https://'
            elif host['use_bolt']:
                db_url = 'bolt://'
            else:
                db_url = 'http://'
            db_url += '%s:%d/%s' % (host['address'], host['port'], host['data_path'])

            authenticate(
                host['address'] + ':' + str(host['port']),
                user=host['username'], password=host['password'])
            self.graph = Graph(db_url)

        return self.graph

    def set_k_fold_tfs(self, k_fold_size):
        """Define which TIME_FRAME nodes should act as boundary between k data
        partitions.

        Args:
            k_fold_size(int): number of data partitions.

        Returns:
            list: contains TIME_FRAME nodes. Length of k_fold_size - 1.
        """
        k_fold_size = 2 if k_fold_size < 2 else k_fold_size

        orders_count = self._query_db(
            '(o:ORDER)', 'count(o) AS orders_count', '(o)-[:CONTAINS]->()')
        orders_count = orders_count[0]['orders_count']
        part_size = orders_count / k_fold_size

        tf_counter = 0
        self.k_fold_tfs = []
        tfs = self._query_db(
            '(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)',
            'tf', '(o)-[:CONTAINS]->()')

        for time_frame in tfs:
            tf_counter += 1
            if tf_counter % part_size == 0:
                self.k_fold_tfs.append(time_frame['tf'])
            if len(self.k_fold_tfs) == k_fold_size - 1:
                break

        return self.k_fold_tfs

    def _define_tf_conditions(self):
        """Define TIME_FRAME conditions for each data type. These conditions are
        used in Cypher's WHERE clause when building a query for distincting
        different datasets such as 'train' or 'test'.

        Returns:
            bool: Are conditions successfully defined or not.
        """
        if self.k_fold_tfs is None or not self.k_fold_tfs:
            return False

        tf_indices = {}
        self.tf_conditions = {}
        if self.testing_part_index == 0:
            tf_indices = {
                'test': [None, 0],
                'train': [0, None]
            }
        elif self.testing_part_index == self.k_fold_size - 1:
            tf_indices = {
                'test': [self.k_fold_size - 2, None],
                'train': [None, self.k_fold_size - 2]
            }
        else:
            tf_indices = {
                'test': [self.testing_part_index - 1, self.testing_part_index],
                'train': [None, self.testing_part_index - 1, self.testing_part_index, None]
            }

        for data_type in ['test', 'train']:
            self.tf_conditions[data_type] = ''
            for i in xrange(0, len(tf_indices[data_type]), 2):
                has_bottom_condition = False
                if i > 0:
                    self.tf_conditions[data_type] += 'OR '

                if tf_indices[data_type][i] is not None:
                    has_bottom_condition = True
                    self.tf_conditions[data_type] += 'tf.timestamp > "%s" ' \
                        % self.k_fold_tfs[tf_indices[data_type][i]]['timestamp']

                if tf_indices[data_type][i + 1] is not None:
                    if has_bottom_condition:
                        self.tf_conditions[data_type] += 'AND '
                    self.tf_conditions[data_type] += 'tf.timestamp <= "%s" ' \
                        % self.k_fold_tfs[tf_indices[data_type][i + 1]]['timestamp']
        return True

    def get_tf_conditions(self, data_type='train'):
        """ Return TIME_FRAME conditions for given data type. Should be used in
        WHERE Cypher clause.

        Args:
            data_type(string, optional): 'train', 'test', 'all'. Defaults to 'train'.
        Returns:
            string
        """
        return self.tf_conditions[data_type]

    def _query_db(self, match, return_values, where_conditions=None, data_type='all'):
        """Build and return Cypher query with given args.

        Args:
            match(string): nodes and relationships which should be matched by
            builded query.
            return_values(string)
            where_conditions(string, optional)
            data_type(string, optional): 'train', 'test', or 'all' which is default.
        Returns:
            string
        """
        query = 'MATCH %s ' % match

        if where_conditions is not None and where_conditions:
            query += 'WHERE %s ' % where_conditions

        if data_type != 'all':
            if where_conditions is not None and where_conditions:
                query += 'AND '
            else:
                query += 'WHERE '

        query += self._get_tf_query_part(data_type)
        query += ' RETURN %s' % return_values
        # print query, '\n'
        return self.graph.data(query)

    def _get_tf_query_part(self, data_type):
        """ Return TIME_FRAME Cypher conditionals or empty string if data_type
        is 'all'.

        Args:
            data_type(string): 'train', 'test' or 'all'

        Returns:
            string
        """
        if data_type in ['train', 'test']:
            return '( %s)' % self.get_tf_conditions(data_type)

        return ''

    @property
    def testing_part_index(self):
        """int: index of testing data partition."""
        return self._testing_part_index

    @testing_part_index.setter
    def testing_part_index(self, value):
        if not isinstance(value, int):
            value = 0
        self._testing_part_index = value
        self._define_tf_conditions()

    @property
    def k_fold_size(self):
        """int: number of data partions."""
        return self._k_fold_size

    @k_fold_size.setter
    def k_fold_size(self, value):
        try:
            self._k_fold_size = int(value)
        except (ValueError, TypeError):
            self._k_fold_size = 0

    @staticmethod
    def _list_to_string(list_):
        """Transform given list to string.

        Args:
            list_(list)
        Returns:
            string
        """
        return '[%s]' % ', '.join(str(e) for e in list_)
