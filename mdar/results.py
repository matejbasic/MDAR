import logging

class Results:
    log = None

    def __init__(self):
        self.precision   = []
        self.recall      = []
        self.fallout     = []
        self.f1          = []
        self.specificity = []

        self.cases_without_history = []

        self.confusion_matrix = {'tp': [], 'tn': [], 'fp': [], 'fn': []}

        self.increments = 0

    def add(self, precision, recall, fallout, f1, specificity, confusion_matrix, cases_without_history):
        self.increments += 1

        self.precision.append(precision)
        self.fallout.append(fallout)
        self.recall.append(recall)
        self.f1.append(f1)
        self.specificity.append(specificity)

        self.cases_without_history.append(cases_without_history)

        for key in confusion_matrix.keys():
            self.confusion_matrix[key].append(confusion_matrix[key])

    def crunch(self, approach = 'average'):
        if self.increments > 0:
            if approach is 'average':
                self.precision   = sum(self.precision) / self.increments
                self.fallout     = sum(self.fallout) / self.increments
                self.recall      = sum(self.recall) / self.increments
                self.f1          = sum(self.f1) / self.increments
                self.specificity = sum(self.specificity) / self.increments

                self.cases_without_history = sum(self.cases_without_history) / self.increments

                for key in self.confusion_matrix.keys():
                    self.confusion_matrix[key] = sum(self.confusion_matrix[key]) / self.increments
            elif approach is 'best':
                f1_max = max(self.f1)
                i_max  = self.f1.index(f1_max)

                self.precision   = self.precision[i_max]
                self.fallout     = self.fallout[i_max]
                self.recall      = self.recall[i_max]
                self.f1          = self.f1[i_max]
                self.specificity = self.specificity[i_max]

                self.cases_without_history = self.cases_without_history[i_max]

                for key in self.confusion_matrix.keys():
                    self.confusion_matrix[key] = self.confusion_matrix[key][i_max]

            self.increments = 0

    def get(self):
        self.crunch()
        return self.precision, self.recall, self.fallout, self.f1, self.specificity, self.confusion_matrix

    def get_confusion_matrix(self):
        self.crunch()
        return self.confusion_matrix

    def get_metrics(self):
        self.crunch()
        return self.precision, self.recall, self.fallout, self.f1, self.specificity

    def set_logger(self, logger_name = 'recomm-tester', file_name = 'test.log'):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        if logger.handlers:
            logger.handlers = []

        fh = logging.FileHandler(file_name)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

        self.log = logger

    def log_results(self, model_type, k, items_total):
        self.crunch()
        if self.log is None:
            self.set_logger()

        self.log_basic_test_info(model_type, k, items_total)
        self.log_confusion_matrix()
        self.log_confusion_metrics()
        self.log.info('-' * 23)

    def log_basic_test_info(self, model_type, k, items_total):
        self.log.info('MODEL: ' + model_type)
        self.log.info('K size: ' + str(k))
        self.log.info('# cases without history: ' + str(self.cases_without_history))
        self.log.info('# transaction items: ' + str(items_total))

    def log_confusion_matrix(self):
        self.log.info('CONFUSION MATRIX:')
        self.log.info('# TP(retrieved relevant): ' + str(self.confusion_matrix['tp']))
        self.log.info('# FP(retrieved irrelevant): ' + str(self.confusion_matrix['fp']))
        self.log.info('# FN(not retrieved relevant): ' + str(self.confusion_matrix['fn']))
        self.log.info('# TN(not retrieved irrelevant): ' + str(self.confusion_matrix['tn']))

    def log_confusion_metrics(self):
        self.log.info('PRECISION: ' + str(self.precision))
        self.log.info('RECALL: ' + str(self.recall))
        self.log.info('FALLOUT: ' + str(self.fallout))
        self.log.info('SPECIFICITY: ' + str(self.specificity))
        self.log.info('F1: ' + str(self.f1))
