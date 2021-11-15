from tweetlib.preprocessing.emoticons import rm_emoticons
from tweetlib.preprocessing.lemmatize import lemmatize
from tweetlib.preprocessing.num import rm_num
from tweetlib.preprocessing.punct import rm_punct
from tweetlib.preprocessing.remove_stop_words import rm_stop_words
from tweetlib.preprocessing.links import rm_links
# from tweetlib.preprocessing.tokenize import tokenize
from tweetlib.preprocessing.lowercase import lowercase
from tweetlib.preprocessing.emails import rm_emails
from tweetlib.preprocessing.remove_alpha_numeric import rm_alpha_numeric
from tweetlib.preprocessing.hashtags import rm_hashtags, fix_hashtags_in_text
from tweetlib.preprocessing.mentions import rm_mentions
from tweetlib.definitions import Preprocessing

dict_preprocessing = {
    Preprocessing.NUM: rm_num,
    Preprocessing.PUNCT: rm_punct,
    Preprocessing.REMOVE_ALPHA_NUMERIC: rm_alpha_numeric,
    Preprocessing.EMOTICONS: rm_emoticons,
    Preprocessing.LEMMATIZE: lemmatize,
    Preprocessing.REMOVE_STOP_WORDS: rm_stop_words,
    Preprocessing.LINKS: rm_links,
    Preprocessing.EMAILS: rm_emails,
    # Preprocessing.TOKENIZE: tokenize,
    Preprocessing.LOWERCASE: lowercase,
    Preprocessing.HASHTAG: rm_hashtags,
    Preprocessing.MENTIONS: rm_mentions,
    Preprocessing.FIX_HASHTAG_TEXT: fix_hashtags_in_text
}