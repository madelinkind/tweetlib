import os
import sys

FILE = __file__
DATA_SET_FOLDER = os.path.split(FILE)[0]
TWEET_LIB_FOLDER = os.path.split(DATA_SET_FOLDER)[0]
PROJECT_FOLDER = os.path.split(TWEET_LIB_FOLDER)[0]

sys.path.append(PROJECT_FOLDER)

from tweetlib.definitions import TypeDataSet
from tweetlib.data_set.data_set import DataSet

class DataSetDeportistas(DataSet):

    def __init__(self, count_tweets_x_user: int):
        super(DataSetDeportistas, self).__init__(TypeDataSet.deportista, count_tweets_x_user)