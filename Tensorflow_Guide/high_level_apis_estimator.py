import tensorflow as tf
from tensorflow_estimator import estimator as est


# input fn
def input_fn(dataset):
    feature_dict = {}
    label = []
    return feature_dict, label


# features
population = tf.feature_column.numeric_column('population')
crime_rate = tf.feature_column.numeric_column('crime_rate')
global_education_mean = 1
median_education = tf.feature_column.numeric_column('median_education',
                                                    normalizer_fn=lambda x: x - global_education_mean)

# estimator
estimator = est.LinearClassifier(
    feature_columns=[population, crime_rate, median_education],
)

# train
estimator.train(input_fn=input_fn, steps=2000)