# -*- coding: utf-8 -*-

from mdar.data_manager import DataManager


class BaseRecommender(object):
    """BaseRecommender class used for aggregating all the methods and attributes
    used by different recommender classes.

    Args:
        min_support(float, optional): minimal support for an association rule
        to be considered valid. Defaults to 0.02.
        min_confidence(float, optional): minimal confidence for an association
        rule to be considered valid. Defaults to 0.05.
    """

    _min_lift = 3
    _min_support = .02
    _min_confidence = .05

    def __init__(self, min_support=None, min_confidence=None):
        self._data_manager = None

        if min_support is not None:
            self.min_support = min_support
        if min_confidence is not None:
            self.min_confidence = min_confidence

    @property
    def data_manager(self):
        """DataManager: object used for data fetching from database."""
        return self._data_manager

    @data_manager.setter
    def data_manager(self, value):
        if isinstance(value, DataManager):
            self._data_manager = value
        else:
            self._data_manager = None

    @property
    def min_support(self):
        """float: minimal support for an association rule to be considered valid.
        """
        return self._min_support

    @min_support.setter
    def min_support(self, value):
        try:
            self._min_support = float(value)
        except (ValueError, TypeError):
            self._min_support = 0

    @property
    def min_confidence(self):
        """float: minimal confidence value for an association rule to be valid.
        """
        return self._min_confidence

    @min_confidence.setter
    def min_confidence(self, value):
        try:
            self._min_confidence = float(value)
        except (ValueError, TypeError):
            self._min_confidence = 0

    @property
    def min_lift(self):
        """int: minimal lift value for a valid association rule."""
        return self._min_lift

    @min_lift.setter
    def min_lift(self, value):
        try:
            self._min_lift = float(value)
        except (ValueError, TypeError):
            self._min_lift = 0
