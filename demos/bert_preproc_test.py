import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from tweetlib.pipeline.execute_pipeline import TwitterPipeline
from tweetlib.config.configuration import Configuration
from tweetlib.definitions import Preprocessing, EncodingMethod, ClassificationMethod, TaggingMethod, TypeTask, TypeDataSet
from tweetlib.classification.classification import Classification
from dict_commands import dict_encoding, dict_classifier, dict_tagging

from tweetlib.data_set.data_set import DataSet
from tweetlib.data_set.politicos import DataSetPoliticos
from tweetlib.data_set.artistas import DataSetArtistas
from tweetlib.data_set.deportistas import DataSetDeportistas
from tweetlib.data_set.youtubers import DataSetYoutubers
from tweetlib.data_set.todos import DataSetTodos

#--------------------------------------PREPROSSESING - BERT--------------------------------
###########################################################################################
#########################################POLITICOS#########################################
###########################################################################################
#------------------------------------------------------------------------------------------

# 1. Config:
        #Preprocessing: -
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: -
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetPoliticos(count_tweets)

# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
def politic_bert_within_prep(count_tweets):
    config = Configuration([

        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        # TaggingMethod.SPACY,
        type_dataset = TypeDataSet.politico
    )

    data_set = DataSetPoliticos(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config

#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetPoliticos(count_tweets)

# 4. Classifier: Classification()
#---------------------------------------LIBRARY--CLASSIFIER------------------------------
#########################################SPACY#######BERT#################################
#----------------------------------------------------------------------------------------

def politic_bert_spacy_token(count_tweets):
    config = Configuration([

            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        TaggingMethod.SPACY,
        TypeDataSet.politico
    )

    data_set = DataSetPoliticos(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE, REMOVE_STOP_WORDS, LOWERCASE, LEMMATIZE
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetPoliticos(count_tweets)

# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def politic_bert_token_stopword_lowercase_lemma(count_tweets):
    config = Configuration([

            Preprocessing.FIX_HASHTAG_TEXT, 
            Preprocessing.REMOVE_STOP_WORDS,
            Preprocessing.LOWERCASE,
            Preprocessing.LEMMATIZE
        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        TaggingMethod.SPACY,
        TypeDataSet.politico
    )

    data_set = DataSetPoliticos(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: REMOVE_ALPHA_NUMERIC
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetPoliticos(count_tweets)

# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def politic_bert_rm_alphanumeric(count_tweets):
    config = Configuration([

            Preprocessing.REMOVE_ALPHA_NUMERIC
        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        TaggingMethod.SPACY,
        TypeDataSet.politico
    )

    data_set = DataSetPoliticos(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: EMOTICONS, LINKS, NUM, PUNCT
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetPoliticos(count_tweets)

# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def politic_bert_emoji_link_num_punct(count_tweets):
    config = Configuration([

            Preprocessing.EMOTICONS,
            Preprocessing.LINKS,
            Preprocessing.NUM,
            Preprocessing.PUNCT
        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        TaggingMethod.SPACY,
        TypeDataSet.politico
    )

    data_set = DataSetPoliticos(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config

#--------------------------------------PREPROSSESING - BERT--------------------------------
###########################################################################################
#########################################ARTISTA###########################################
###########################################################################################
#------------------------------------------------------------------------------------------

# 1. Config:
        #Preprocessing: -
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: -
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetArtistas(count_tweets)

# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
def artist_bert_within_prep(count_tweets):
    config = Configuration([

        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        # TaggingMethod.SPACY,
        type_dataset = TypeDataSet.artista
    )

    data_set = DataSetArtistas(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config

#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetArtistas(count_tweets)

# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
#---------------------------------------LIBRARY--CLASSIFIER-------------------------------
#########################################SPACY#######BERT##################################
#-----------------------------------------------------------------------------------------

def artist_bert_spacy_token(count_tweets):
    config = Configuration([

            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        TaggingMethod.SPACY,
        TypeDataSet.artista
    )

    data_set = DataSetArtistas(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config

#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE, REMOVE_STOP_WORDS, LOWERCASE, LEMMATIZE
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetArtistas(count_tweets)

# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def artist_bert_token_stopword_lowercase_lemma(count_tweets):
    config = Configuration([

            Preprocessing.FIX_HASHTAG_TEXT, 
            Preprocessing.REMOVE_STOP_WORDS,
            Preprocessing.LOWERCASE,
            Preprocessing.LEMMATIZE
        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        TaggingMethod.SPACY,
        TypeDataSet.artista
    )

    data_set = DataSetArtistas(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config

#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: REMOVE_ALPHA_NUMERIC
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetArtistas(count_tweets)

# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def artist_bert_rm_alphanumeric(count_tweets):
    config = Configuration([

            Preprocessing.REMOVE_ALPHA_NUMERIC
        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        TaggingMethod.SPACY,
        TypeDataSet.artista
    )

    data_set = DataSetArtistas(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config

#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: EMOTICONS, LINKS, NUM, PUNCT
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetArtistas(count_tweets)

# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def artist_bert_emoji_link_num_punct(count_tweets):
    config = Configuration([

        Preprocessing.EMOTICONS,
        Preprocessing.LINKS,
        Preprocessing.NUM,
        Preprocessing.PUNCT
    ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        TaggingMethod.SPACY,
        TypeDataSet.artista
    )

    data_set = DataSetArtistas(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config


#--------------------------------------PREPROSSESING - BERT---------------------------------------
###########################################################################################
#########################################DEPORTISTA########################################
###########################################################################################
#------------------------------------------------------------------------------------------

# 1. Config:
        #Preprocessing: -
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: -
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetDeportistas(count_tweets)

# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
def athlete_bert_within_prep(count_tweets):
    config = Configuration([

        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        # TaggingMethod.SPACY,
        type_dataset = TypeDataSet.deportista
    )

    data_set = DataSetDeportistas(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config

#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetDeportistas(count_tweets)

# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
#---------------------------------------LIBRARY--CLASSIFIER-------------------------------
#########################################SPACY#######BERT##################################
#-----------------------------------------------------------------------------------------

def athlete_bert_spacy_token(count_tweets):
    config = Configuration([

            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        TaggingMethod.SPACY,
        TypeDataSet.deportista
    )

    data_set = DataSetDeportistas(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config

#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE, REMOVE_STOP_WORDS, LOWERCASE, LEMMATIZE
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetDeportistas(count_tweets)

# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def athlete_bert_token_stopword_lowercase_lemma(count_tweets):
    config = Configuration([

            Preprocessing.FIX_HASHTAG_TEXT, 
            Preprocessing.REMOVE_STOP_WORDS,
            Preprocessing.LOWERCASE,
            Preprocessing.LEMMATIZE
        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        TaggingMethod.SPACY,
        TypeDataSet.deportista
    )

    data_set = DataSetDeportistas(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config

#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: REMOVE_ALPHA_NUMERIC
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetDeportistas(count_tweets)

# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def athlete_bert_rm_alphanumeric(count_tweets):
    config = Configuration([

            Preprocessing.REMOVE_ALPHA_NUMERIC
        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        TaggingMethod.SPACY,
        TypeDataSet.deportista
    )

    data_set = DataSetDeportistas(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config

#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: EMOTICONS, LINKS, NUM, PUNCT
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetDeportistas(count_tweets)

# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def athlete_bert_emoji_link_num_punct(count_tweets):
    config = Configuration([

            Preprocessing.EMOTICONS,
            Preprocessing.LINKS,
            Preprocessing.NUM,
            Preprocessing.PUNCT
        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        TaggingMethod.SPACY,
        TypeDataSet.deportista
    )

    data_set = DataSetDeportistas(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config


#-------------------------------------PREPROSSESING - BERT----------------------------------------
###########################################################################################
#########################################YOUTUBER##########################################
###########################################################################################
#------------------------------------------------------------------------------------------

# 1. Config:
        #Preprocessing: -
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: -
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetYoutubers(count_tweets)

# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
def youtuber_bert_within_prep(count_tweets):
    config = Configuration([

        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        # TaggingMethod.SPACY,
        type_dataset = TypeDataSet.youtuber
    )

    data_set = DataSetYoutubers(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config

#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetYoutubers(count_tweets)

# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
#---------------------------------------LIBRARY--CLASSIFIER-------------------------------
#########################################SPACY#######BERT##################################
#-----------------------------------------------------------------------------------------

def youtuber_bert_spacy_token(count_tweets):
    config = Configuration([

            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        TaggingMethod.SPACY,
        TypeDataSet.youtuber
    )

    data_set = DataSetYoutubers(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config

#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE, REMOVE_STOP_WORDS, LOWERCASE, LEMMATIZE
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetYoutubers(count_tweets)

# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def youtuber_bert_token_stopword_lowercase_lemma(count_tweets):
    config = Configuration([

            Preprocessing.FIX_HASHTAG_TEXT, 
            Preprocessing.REMOVE_STOP_WORDS,
            Preprocessing.LOWERCASE,
            Preprocessing.LEMMATIZE
        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        TaggingMethod.SPACY,
        TypeDataSet.youtuber
    )

    data_set = DataSetYoutubers(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config

#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: REMOVE_ALPHA_NUMERIC
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetYoutubers(count_tweets)

# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def youtuber_bert_rm_alphanumeric(count_tweets):
    config = Configuration([

            Preprocessing.REMOVE_ALPHA_NUMERIC
        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        TaggingMethod.SPACY,
        TypeDataSet.youtuber
    )

    data_set = DataSetYoutubers(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config

#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: EMOTICONS, LINKS, NUM, PUNCT
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetYoutubers(count_tweets)

# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def youtuber_bert_emoji_link_num_punct(count_tweets):
    config = Configuration([

            Preprocessing.EMOTICONS,
            Preprocessing.LINKS,
            Preprocessing.NUM,
            Preprocessing.PUNCT
        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        TaggingMethod.SPACY,
        TypeDataSet.youtuber
    )

    data_set = DataSetYoutubers(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config

#--------------------------------------PREPROSSESING - BERT---------------------------------------
###########################################################################################
###########################################TODOS###########################################
###########################################################################################
#------------------------------------------------------------------------------------------

# 1. Config:
        #Preprocessing: -
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: -
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetTodos()
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
def all_bert_within_prep(count_tweets):
    config = Configuration([

        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        # TaggingMethod.SPACY,
        type_dataset = TypeDataSet.all_results
    )

    data_set = DataSetTodos(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config

#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetTodos()
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
#---------------------------------------LIBRARY--CLASSIFIER-------------------------------
#########################################SPACY#######BERT##################################
#-----------------------------------------------------------------------------------------

def all_bert_spacy_token(count_tweets):
    config = Configuration([

            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        TaggingMethod.SPACY,
        TypeDataSet.all_results
    )

    data_set = DataSetTodos(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config

#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE, REMOVE_STOP_WORDS, LOWERCASE, LEMMATIZE
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetTodos()
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def all_bert_token_stopword_lowercase_lemma(count_tweets):
    config = Configuration([

            Preprocessing.FIX_HASHTAG_TEXT, 
            Preprocessing.REMOVE_STOP_WORDS,
            Preprocessing.LOWERCASE,
            Preprocessing.LEMMATIZE
        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        TaggingMethod.SPACY,
        TypeDataSet.all_results
    )

    data_set = DataSetTodos(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config

#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: REMOVE_ALPHA_NUMERIC
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetTodos()
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def all_bert_rm_alphanumeric(count_tweets):
    config = Configuration([

            Preprocessing.REMOVE_ALPHA_NUMERIC
        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        TaggingMethod.SPACY,
        TypeDataSet.all_results
    )

    data_set = DataSetTodos(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config

#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: EMOTICONS, LINKS, NUM, PUNCT
        #Encoding: BERT
        #Classifier: BERT
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetTodos()
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def all_bert_emoji_link_num_punct(count_tweets):
    config = Configuration([

            Preprocessing.EMOTICONS,
            Preprocessing.LINKS,
            Preprocessing.NUM,
            Preprocessing.PUNCT
        ], 
        EncodingMethod.BERT,
        ClassificationMethod.BERT,
        TaggingMethod.SPACY,
        TypeDataSet.all_results
    )

    data_set = DataSetTodos(count_tweets)

    pipeline = TwitterPipeline(config, dataset=data_set)

    return  pipeline.run(), config

