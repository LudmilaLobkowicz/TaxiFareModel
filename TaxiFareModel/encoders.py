from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

from TaxiFareModel.utils import haversine_vectorized

class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """Extract the day of week (dow), the hour, the month and the year from a
    time column."""
    def __init__(self, time_column, time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'"""
        assert isinstance(X, pd.DataFrame)
        X2 = X.copy()
        X2.index = pd.to_datetime(X2[self.time_column])
        X2.index = X2.index.tz_convert(self.time_zone_name)
        X2["hour"] = X2.index.hour
        X2["month"] = X2.index.month
        X2["dow"] = X2.index.dayofweek
        X2["year"] = X2.index.year
        return X2[['dow', 'hour', 'month', 'year']].reset_index(drop=True)


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """Compute the haversine distance between two GPS points."""
    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X2 = X.copy()
        X2['distance'] = haversine_vectorized(X2)
        return X2[['distance']]
