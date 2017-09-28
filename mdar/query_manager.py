import json
from py2neo import authenticate, Graph

from neo4j.v1 import GraphDatabase, basic_auth

class QueryManager(object):

    k_fold_size        = 2
    k_fold_tfs         = None
    testing_part_index = 0
    tf_conditionals    = None
    config_path        = None

    def __init__(self, config_path = None, k_fold_size = 2, multiprocessing_friendly = False):
        if multiprocessing_friendly:
            self.multiprocessing_friendly = True
        else:
            self.multiprocessing_friendly = False
            self.set_graph(config_path)

        self.set_k_fold_tfs(k_fold_size)
        self.set_k_fold_size(k_fold_size)

        self.set_config_path(config_path)

    def set_graph(self, config_path):
        """
        Parameters
        --------------
        config_path
            path to a config.json with data needed to connect to a db

        Return
        --------------
        None
        """
        with open(config_path) as config_data:
            config = json.load(config_data)
            [str(x) for x in config]
            host = config['host']

            if host['use_ssl']:
                db_url = 'https://'
            elif host['use_bolt']:
                db_url = 'bolt://'
            else:
                db_url = 'http://'
            db_url += host['address'] + ':' + str(host['port']) + '/' + host['data_path']

            authenticate(host['address'] + ':' + str(host['port']), user=host['username'], password=host['password'])
            self.graph = Graph(db_url)

    def get_graph(self):
        """
        Return
        --------------
        Graph instance
        """
        if self.graph is None:
            if type(self.config_path) is str:
                self.set_graph(self.config_path)
            else:
                return None
        return self.graph

    def set_k_fold_tfs(self, k_fold_size = 2):
        k_fold_size = 2 if k_fold_size < 2 else k_fold_size

        orders_count    = self._query_db('(o:ORDER)', 'count(o) AS orders_count', '(o)-[:CONTAINS]->()')
        orders_count    = orders_count[0]['orders_count']
        part_size       = orders_count / k_fold_size
        self.k_fold_tfs = []
        tfs  = self._query_db('(o:ORDER)-[:CREATED_AT]->(tf:TIME_FRAME)', 'tf', '(o)-[:CONTAINS]->()')
        tf_c = 0

        for tf in tfs:
            tf_c += 1
            if tf_c % part_size == 0:
                self.k_fold_tfs.append(tf['tf'])
            if len(self.k_fold_tfs) == k_fold_size-1:
                break

        return self.k_fold_tfs

    def set_testing_part_index(self, index):
        if type(index) != int:
            index = 0
        self.testing_part_index = index
        self._set_tf_conditionals()

    def get_testing_part_index(self):
        return self.testing_part_index

    def set_config_path(self, config_path):
        self.config_path = str(config_path)

    def get_config_path(self):
        return self.config_path

    def set_k_fold_size(self, k_fold_size):
        self.k_fold_size = int(k_fold_size)

    def get_k_fold_size(self):
        return self.k_fold_size

    def _set_tf_conditionals(self):
        # print self.k_fold_tfs
        if self.k_fold_tfs is None or len(self.k_fold_tfs) is 0:
            return

        self.tf_conditionals = {}
        tf_indices = {}
        if self.testing_part_index == 0:
            tf_indices = {
                'test': [None, 0],
                'train': [0, None]
            }
        elif self.testing_part_index == self.k_fold_size-1:
            tf_indices = {
                'test': [self.k_fold_size-2, None],
                'train': [None, self.k_fold_size-2]
            }
        else:
            tf_indices = {
                'test': [self.testing_part_index-1, self.testing_part_index],
                'train': [None, self.testing_part_index-1, self.testing_part_index, None],
            }

        # print tf_indices
        for data_type in ['test', 'train']:
            self.tf_conditionals[data_type] = ''
            for i in xrange(0, len(tf_indices[data_type]), 2):
                has_bottom_conditional = False
                if i > 0:
                    self.tf_conditionals[data_type] += 'OR '

                if tf_indices[data_type][i] is not None:
                    self.tf_conditionals[data_type] += 'tf.timestamp > "' + self.k_fold_tfs[tf_indices[data_type][i]]['timestamp'] + '" '
                    has_bottom_conditional = True
                if tf_indices[data_type][i+1] is not None:
                    if has_bottom_conditional:
                        self.tf_conditionals[data_type] += 'AND '
                    self.tf_conditionals[data_type] += 'tf.timestamp <= "' + self.k_fold_tfs[tf_indices[data_type][i+1]]['timestamp'] + '" '

    def _list_to_string(self, l):
        return '['+ ', '.join(str(e) for e in l) + ']'

    def _get_tf_conditionals(self, data_type = 'train'):
        return self.tf_conditionals[data_type]

    def _query_db(self, match_nodes, return_values, where_conditional = None, data_type = 'all'):
        query = 'MATCH ' + match_nodes + ' '

        if where_conditional is not None and len(where_conditional) > 0:
            query += 'WHERE ' + where_conditional + ' '


        if data_type != 'all':
            if where_conditional is not None and len(where_conditional) > 0:
                query += 'AND '
            else:
                query += 'WHERE '

        query += self._get_tf_query_part(data_type)
        query += ' RETURN ' + return_values
        # print query, '\n'
        return self.graph.data(query)

    def _get_tf_query_part(self, data_type):
        if type(data_type) is str and data_type in ['train', 'test']:
            return '(' + self._get_tf_conditionals(data_type) + ')'
        elif type(data_type) is list:
            is_first = True
            query = '('
            for dt in data_type:
                if is_first:
                    is_first = False
                else:
                    query += ' OR '
                if dt in ['train', 'test']:
                    query += '(' + self._get_tf_conditionals(dt) + ')'
            query += ') '
            return query
        else:
            return ''
