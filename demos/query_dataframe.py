# import os
# import sys

# # Django specific settings

# FILE = __file__
# DATA_SET_FOLDER = os.path.split(FILE)[0]
# TWEET_LIB_FOLDER = os.path.split(DATA_SET_FOLDER)[0]
# PROJECT_FOLDER = os.path.split(TWEET_LIB_FOLDER)[0]

# sys.path.append(TWEET_LIB_FOLDER)
# sys.path.append(f"{TWEET_LIB_FOLDER}/orm")
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")

# # import and setup django
# import django
# django.setup()

# from db.models import Tweet, TwitterUser
# from tweetlib.definitions import TypeDataSet
# from django.db.models.functions import Length
# from django.db.models import Q

# from joblib import dump, load
# import psycopg2
# import pandas as pd

# conn = psycopg2.connect(
#     host="raspberrypi.home",
#     database="twitter_db",
#     user="root",
#     password="root"
# )

#         # query_db = Tweet.objects.filter(
#         #     # Q(twitter_user=user_name.id) &
#         #     Q(is_retweet=False) &
#         #     Q(tweet_lang="es") &
#         #     Q(twitter_user__type_user="politico")
#         # )

#         # q2 = query_db.annotate(text_len=Length('tweet_text')).filter(text_len__gt=3)

#     #     last_tweets = Tweet.objects.filter(
#     #         twitter_user=user_name.id) | Tweet.objects.filter(is_retweet=False) | Tweet.objects.filter(tweet_lang='es') | Tweet.objects.annotate(text_len=Length('tweet_text')).filter(
#     # text_len__gt=3)

#     # q = str(query_db.query)

# Q_USERS = """
#     SELECT "twitter_users"."id", 
#         "twitter_users"."screen_name", 
#         "twitter_users"."type_user" 
#     FROM "twitter_users"
# """
# df_users = pd.read_sql(Q_USERS, conn)
# dump(df_users, 'models/df_users')
# ids = df_users.twitter_user_id.unique()

# # Q_TODOS = """
# #     SELECT
# #         "tweets"."id",
# #         "tweets"."twitter_user_id",
# #         "tweets"."tweet_text",
# #         "tweets"."tweet_date",
# #         "tweets"."tweet_lang",
# #         "tweets"."tweet_id",
# #         "tweets"."tweet_info",
# #         "tweets"."is_retweet",
# #         "tweets"."to_predict",
# #         "tweets"."retweet_count",
# #         LENGTH("tweets"."tweet_text") AS "text_len"
# #     FROM
# #         "tweets" INNER JOIN "twitter_users" ON
# #             ("tweets"."twitter_user_id" = "twitter_users"."id")
# #     WHERE (
# #         NOT "tweets"."is_retweet" AND
# #         "tweets"."tweet_lang" = 'es'
# #         AND "twitter_users"."type_user" = 'youtuber'
# #         AND LENGTH("tweets"."tweet_text") > 3
# #     )
# # """
# # df_youtubers = pd.read_sql(Q_TODOS, conn)
# # dump(df_youtubers, 'models/df_youtubers')
# # ids = df_youtubers.twitter_user_id.unique()

# # if __name__ == '__main__':
# #     get_tuple_tweets_user(user_type)