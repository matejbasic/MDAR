class BaseRecommender(object):
    min_support    = .002
    min_confidence = .05
    minimal_lift   = 3

    def __init__(self, config_path = None, k_fold_size = 3):
        if config_path is not None:
            self.dm = DataManager(config_path, k_fold_size)

    def set_data_manager(self, dm):
        self.dm = dm

    def set_min_support(self, support):
        try:
            self.min_support = float(support)
        except (ValueError, TypeError):
            self.min_support = 0

    def set_min_confidence(self, confidence):
        try:
            self.min_confidence = float(confidence)
        except (ValueError, TypeError):
            self.min_confidence = 0

    def set_minimal_lift(self, lift):
        try:
            self.minimal_lift = float(lift)
        except (ValueError, TypeError):
            self.minimal_lift = 0
