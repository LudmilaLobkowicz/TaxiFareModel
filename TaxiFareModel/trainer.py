from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.utils import haversine_vectorized, compute_rmse
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y


    def holdout(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.1)


    def set_pipeline(self):
        print("pipeline")
        """defines the pipeline as a class attribute"""
        pipe_distance = make_pipeline(DistanceTransformer(), RobustScaler())
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']

        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), StandardScaler())
        time_cols = ['pickup_datetime']

        feat_eng_bloc = ColumnTransformer([('time', pipe_time, time_cols),
                                  ('distance', pipe_distance, dist_cols)])
        self.pipeline = Pipeline(steps=[('feat_eng_bloc', feat_eng_bloc),
                            ('LinearRegression', LinearRegression())])


    def run(self):
        """set and train the pipeline"""
        print("let's train")
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)

    def evaluate(self):
        """evaluates the pipeline on df_test and return the RMSE"""
        self.y_pred = self.pipeline.predict(self.X_test)
        print (compute_rmse(self.y_pred, self.y_test))



if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    X = df.drop(columns='fare_amount')
    y = df['fare_amount']
    trainer = Trainer(X, y)
    trainer.holdout()
    pipeline = trainer.run()
    evaluate = trainer.evaluate()

    print('TODO')
