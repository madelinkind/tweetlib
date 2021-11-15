import os
import sys

# Django specific settings

FILE = __file__
DATA_SET_FOLDER = os.path.split(FILE)[0]
TWEET_LIB_FOLDER = os.path.split(DATA_SET_FOLDER)[0]
PROJECT_FOLDER = os.path.split(TWEET_LIB_FOLDER)[0]

sys.path.append(PROJECT_FOLDER)
sys.path.append(f"{PROJECT_FOLDER}/orm")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")

# import and setup django
import django
django.setup()
from joblib import dump

# Import your models for use in your script
# pylint: disable=import-error
from db.models import Tweet, TwitterUser
from tweetlib.definitions import TypeDataSet

class DataSet():

    def __init__(self, user_type: TypeDataSet, count_tweets_x_user: int):
        data, y = self.get_tuple_tweets_user(user_type, count_tweets_x_user)
        self.data = data
        self.y = y

    #Se obtiene la tupla (list de tweets (data) y list de user (y)) por tipo de usuario.
    def get_tuple_tweets_user(self, user_type: TypeDataSet, count_tweets_x_user):
        if user_type == TypeDataSet.politico:
            return self.get_user_tweets(user_type, count_tweets_x_user)
        if user_type == TypeDataSet.youtuber:
            return self.get_user_tweets(user_type, count_tweets_x_user)
        if user_type == TypeDataSet.artista:
            return self.get_user_tweets(user_type, count_tweets_x_user)
        if user_type == TypeDataSet.deportista:
            return self.get_user_tweets(user_type, count_tweets_x_user)
        elif user_type == TypeDataSet.all_results:
            return self.get_user_tweets(user_type, count_tweets_x_user) 
        else:
            raise Exception("You must enter a valid user type, See TypeDataSet.")

    #Devuelve lista de usuario segun el tipo
    def get_list_users(self, user_type: TypeDataSet, count_tweets_x_user) -> list:
        if user_type == TypeDataSet.all_results:
            users_map = map(
                lambda item: item['screen_name'], 
                TwitterUser.objects.values('screen_name')
            )
            list_users = list(users_map)
            self.get_user_tweets(list_users, count_tweets_x_user)
        else:
            users_map = map(lambda item: item['screen_name'], TwitterUser.objects.filter(type_user=user_type).values('screen_name'))
            list_users = list(users_map)
        return list_users

    # Se obtiene una tupla (lista de tweets, lista del mismo usuario repetido con misma longitud que la lista de tweets)
    def get_user_tweets(self, user_type: TypeDataSet, count_tweets_x_user):

        list_users = self.get_list_users(user_type, count_tweets_x_user)
        print(f'Descargando dataset {user_type}...')
        # Lists of Data(texts) and Class(y) in (es)
        data = []
        y = []
        cont_users = 0
        cont_id = 0
        for user in list_users:
            #borrar luego, es para limitar la cantidad de usuarios de cada dateset
            cont_users += 1
            cont_id += 1
            
            # if user == 'PabloIglesias' or user == 'onacarbonell' or user == 'onacarbonell' or user == 'RomualdFons' or user == 'antoniobanderas':

            # if user != 'PabloIglesias' and user != 'onacarbonell' and user != 'onacarbonell' and user != 'RomualdFons' and user != 'antoniobanderas':
            length = 0
            user_name = TwitterUser.objects.get(screen_name=user)
            last_tweets = Tweet.objects.filter(twitter_user=user_name.id).all()
            for tweet in last_tweets:
                if not tweet.is_retweet and tweet.tweet_lang == 'es' and len(tweet.tweet_text) > 3:
                # if not tweet.is_retweet and tweet.to_predict and tweet.tweet_lang == 'es' and len(tweet.tweet_text) > 2:
                    length += 1
                    if length != count_tweets_x_user+1:
                        data.append(tweet.tweet_text)
                        y.append(cont_id - 1)
                    else:
                        break
            # else:
                # continue
        # print(len(data))
        # print(len(y))
        # print(y)
        # print(cont_users)
        # print(y)

        # dump(data, 'models/X_test_bert_all_pred_4')
        # dump(y, 'models/y_test_bert_all_pred_36')
        print(f'Dataset {user_type}, descargado correctamente.')
        return data, y

    def get_data(self):
        return self.data

    def get_y(self):
        return self.y

    def is_valid(self):
        if self.data is None or not len(self.data) or self.y is None or not len(self.y):
            return False 

        return len(self.data) == len(self.y)
