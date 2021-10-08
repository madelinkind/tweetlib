import os
import sys

# Django specific settings
sys.path.append('./orm')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")

# import and setup django
import django
django.setup()

# Import your models for use in your script
# pylint: disable=import-error
from db.models import Tweet, TwitterUser

def data_test():

    users_map = map(lambda item: item['screen_name'], TwitterUser.objects.order_by('id').values('screen_name'))
    users_list = list(users_map)

    for idx, i in enumerate(users_list):

        user = TwitterUser.objects.get(screen_name=users_list[idx])
        # list_tweets = Tweet.objects.filter(twitter_user=user.id)
        count_to_predict = Tweet.objects.filter(twitter_user=user.id, to_predict=True, tweet_lang='es', is_retweet=False).count()
        have_tweet = Tweet.objects.filter(twitter_user=user.id, to_predict=False, tweet_lang='es', is_retweet=False).exists()
        if count_to_predict < 20 and have_tweet:
            list_tweets_to_predict_false = Tweet.objects.filter(twitter_user=user.id, to_predict=False, tweet_lang='es', is_retweet=False)[:(20-count_to_predict)]
            for tweet in list_tweets_to_predict_false:
                if count_to_predict == 20:
                    break
                tweet.to_predict = True
                tweet.save()
                count_to_predict+=1
        elif count_to_predict > 20:
            list_tweets_to_predict_true = Tweet.objects.filter(twitter_user=user.id, to_predict=True, tweet_lang='es', is_retweet=False)[:(count_to_predict-20)]
            for tweet_true in list_tweets_to_predict_true:
                if count_to_predict == 20:
                    break
                tweet_true.to_predict = False 
                tweet_true.save()
                count_to_predict-=1


if __name__ == '__main__':
    data_test()