# -*- coding: utf-8 -*-

import logging


class Results(object):
    """Results class used for stacking the test results (confusion matrix and
    IR measures) and summarizing values by different approaches.

    Attributes:
        precision(float)
        recall(float)
        fallout(float)
        f1_score(float)
        specificity(float)
        confusion_matrix(dict): should be of following structure:
            {
                'tp': int
                'tn': int
                'fp': int
                'fn': int
            }
        cases_without_history(int)
        increments(int): current number of test results sets.
    """

    def __init__(self):
        self._log = None

        self.reset()

    def reset(self):
        """Reset all the attributes/measures to their inital values."""
        self._precision = []
        self._fallout = []
        self._recall = []
        self._f1_score = []
        self._specificity = []

        self._cases_without_history = []

        self._confusion_matrix = {'tp': [], 'tn': [], 'fp': [], 'fn': []}

        self._increments = 0

    def add(self, precision, recall, fallout, f1_score, specificity,
            confusion_matrix, cases_without_history):
        """Append measure values to appropriate lists and increment counter.

        Args:
            precision(float)
            recall(float)
            fallout(float)
            f1_score(float)
            specificity(float)
            confusion_matrix(dict): should be of following structure:
                {
                    'tp': int
                    'tn': int
                    'fp': int
                    'fn': int
                }
            cases_without_history(int)
        """
        self._increments += 1

        self._precision.append(float(precision))
        self._fallout.append(float(fallout))
        self._recall.append(float(recall))
        self._f1_score.append(float(f1_score))
        self._specificity.append(float(specificity))

        self._cases_without_history.append(int(cases_without_history))

        if isinstance(confusion_matrix, dict):
            for key in confusion_matrix:
                self._confusion_matrix[key].append(int(confusion_matrix[key]))

    def summarize(self, approach='average'):
        """Crunch all the attributes of type 'list' to a single float or int.
        As every test is done on K parts, results contain K different values
        for each attribute/measure. Before further usage of this values, they
        need to be summarized with two different approaches.

        Args:
            approach(str, optional): 'average' or 'best'. Defaults to 'average'.
        """
        if self._increments > 0:
            if approach == 'average':
                self._precision = sum(self._precision) / self._increments
                self._fallout = sum(self._fallout) / self._increments
                self._recall = sum(self._recall) / self._increments
                self._f1_score = sum(self._f1_score) / self._increments
                self._specificity = sum(self._specificity) / self._increments

                self._cases_without_history = sum(self._cases_without_history)
                self._cases_without_history /= self._increments

                for key in self._confusion_matrix:
                    self._confusion_matrix[key] = sum(self._confusion_matrix[key])
                    self._confusion_matrix[key] /= self._increments
            elif approach == 'best':
                f1_score_max = max(self._f1_score)
                i_max = self._f1_score.index(f1_score_max)

                self._precision = self._precision[i_max]
                self._fallout = self._fallout[i_max]
                self._recall = self._recall[i_max]
                self._f1_score = self._f1_score[i_max]
                self._specificity = self._specificity[i_max]

                self._cases_without_history = self._cases_without_history[i_max]

                for key in self._confusion_matrix:
                    self._confusion_matrix[key] = self._confusion_matrix[key][i_max]

            self._increments = 0

    def get_confusion_matrix(self):
        """ Returns confusion matrix.

        Returns:
            dict: confusion matrix
        """
        return self._confusion_matrix

    def get_evaluation_measures(self):
        """Returns the evaluation measures.

        Returns:
            float: precision.
            float: recall.
            float: fallout.
            float: F1 score.
            float: specificity.
        """
        return self._precision, self._recall, self._fallout, self._f1_score, self._specificity

    def set_logger(self, logger_name='recomm-tester', file_name='test.log'):
        """Defines a logger with the given name and assign a file with the given
        name. Creates a file if missing.

        Args:
            logger_name(string, optional): Name of the logger. Defaults to
            'recomm-tester'
            file_name(string, optional). Name of the file which is assigned to
            the defined logger. Defaults to 'test.log'.
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        if logger.handlers:
            logger.handlers = []

        file_handler = logging.FileHandler(file_name)
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        logger.addHandler(file_handler)

        self._log = logger

    def log_results(self, model_type, k, items_total):
        """Logs all the data gathered from testing."""
        if self._log is None:
            self.set_logger()

        self.log_basic_test_info(model_type, k, items_total)
        self.log_confusion_matrix()
        self.log_evaluation_measures()
        self._log.info('-' * 23)

    def log_basic_test_info(self, model_type, k, items_total):
        """Writes model type, k size, number of cases without history and total
        transaction items to the log file.
        """
        self._log.info('MODEL: %s', model_type)
        self._log.info('K size: %d', k)
        self._log.info('# cases without history: %d', self._cases_without_history)
        self._log.info('# transaction items: %d', items_total)

    def log_confusion_matrix(self):
        """Writes confusion matrix values to the log file."""
        self._log.info('CONFUSION MATRIX:')
        self._log.info(
            '# TP(retrieved relevant): %d', self._confusion_matrix['tp']
        )
        self._log.info(
            '# FP(retrieved irrelevant): %d', self._confusion_matrix['fp']
        )
        self._log.info(
            '# FN(not retrieved relevant): %d', self._confusion_matrix['fn']
        )
        self._log.info(
            '# TN(not retrieved irrelevant): %d', self._confusion_matrix['tn']
        )

    def log_evaluation_measures(self):
        """Writes evaluation measures to the log file."""
        self._log.info('PRECISION: %f', self._precision)
        self._log.info('RECALL: %f', self._recall)
        self._log.info('FALLOUT: %f', self._fallout)
        self._log.info('SPECIFICITY: %f', self._specificity)
        self._log.info('F1 SCORE: %f', self._f1_score)
