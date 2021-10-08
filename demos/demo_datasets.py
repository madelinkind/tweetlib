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

from tweetlib.data_set.politicos import DataSetPoliticos
from tweetlib.data_set.artistas import DataSetArtistas
from tweetlib.data_set.deportistas import DataSetDeportistas
from tweetlib.data_set.youtubers import DataSetYoutubers
from tweetlib.data_set.todos import DataSetTodos

from joblib import dump, load

from demos.bert_preproc_test import *

from inspect import signature

import pandas as pd
df = pd.DataFrame()


list_count_tweets_x_user_to_download = [50, 100, 200, 500, 1000]
# list_count_tweets_x_user_to_download = [50]

#--------------------------------------PREPROSSESING - POS---------------------------------------
###########################################################################################
#########################################POLITICOS#########################################
###########################################################################################
#------------------------------------------------------------------------------------------

# 1. Config:
        #Preprocessing: -
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetPoliticos(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
def politic_pos_svm_spacy_within_prep(count_tweets):
    config = Configuration([
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.SVM,
        TaggingMethod.SPACY,
        TypeDataSet.politico
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #-------------------------------------------------------

    data_set = DataSetPoliticos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetPoliticos(count_tweets)
# 4. Classifier: Classification()
#---------------------------------------LIBRARY--CLASSIFIER------------------------------
#########################################SPACY#######SVM#################################
#----------------------------------------------------------------------------------------

def politic_pos_svm_spacy_token(count_tweets):
    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.SVM,
        TaggingMethod.SPACY,
        TypeDataSet.politico
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetPoliticos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE, REMOVE_STOP_WORDS, LOWERCASE, LEMMATIZE
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetPoliticos(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def politic_pos_svm_token_stopword_lowercase_lemma(count_tweets):
    config = Configuration([
        Preprocessing.FIX_HASHTAG_TEXT, 
        Preprocessing.REMOVE_STOP_WORDS,
        Preprocessing.LOWERCASE,
        Preprocessing.LEMMATIZE
    ], 
    EncodingMethod.POSTAGGING,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.politico
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetPoliticos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: REMOVE_ALPHA_NUMERIC
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetPoliticos(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def politic_pos_svm_rm_alphanumeric(count_tweets):
    config = Configuration([
        Preprocessing.REMOVE_ALPHA_NUMERIC
    ], 
    EncodingMethod.POSTAGGING,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.politico
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetPoliticos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: EMOTICONS, LINKS, NUM, PUNCT
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetPoliticos(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def politic_pos_svm_emoji_link_num_punct(count_tweets):
    config = Configuration([
        Preprocessing.EMOTICONS,
        Preprocessing.LINKS,
        Preprocessing.NUM,
        Preprocessing.PUNCT
    ], 
    EncodingMethod.POSTAGGING,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.politico
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetPoliticos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------PREPROSSESING - POS---------------------------------------
###########################################################################################
#########################################ARTISTA###########################################
###########################################################################################
#------------------------------------------------------------------------------------------

# 1. Config:
        #Preprocessing: -
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetArtistas(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
def artist_pos_svm_spacy_within_prep(count_tweets):
    config = Configuration([
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.SVM,
        TaggingMethod.SPACY,
        TypeDataSet.artista
    )
    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #-------------------------------------------------------

    data_set = DataSetArtistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetArtistas(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
#---------------------------------------LIBRARY--CLASSIFIER-------------------------------
#########################################SPACY#######SVM##################################
#-----------------------------------------------------------------------------------------

def artist_pos_svm_spacy_token(count_tweets):
    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.SVM,
        TaggingMethod.SPACY,
        TypeDataSet.artista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetArtistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE, REMOVE_STOP_WORDS, LOWERCASE, LEMMATIZE
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetArtistas(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def artist_pos_svm_token_stopword_lowercase_lemma(count_tweets):
    config = Configuration([
        Preprocessing.FIX_HASHTAG_TEXT, 
        Preprocessing.REMOVE_STOP_WORDS,
        Preprocessing.LOWERCASE,
        Preprocessing.LEMMATIZE
    ], 
    EncodingMethod.POSTAGGING,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.artista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetArtistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: REMOVE_ALPHA_NUMERIC
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetArtistas(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def artist_pos_svm_rm_alphanumeric(count_tweets):
    config = Configuration([
        Preprocessing.REMOVE_ALPHA_NUMERIC
    ], 
    EncodingMethod.POSTAGGING,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.artista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetArtistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: EMOTICONS, LINKS, NUM, PUNCT
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetArtistas(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def artist_pos_svm_emoji_link_num_punct(count_tweets):
    config = Configuration([
        Preprocessing.EMOTICONS,
        Preprocessing.LINKS,
        Preprocessing.NUM,
        Preprocessing.PUNCT
    ], 
    EncodingMethod.POSTAGGING,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.artista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetArtistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------PREPROSSESING - POS---------------------------------------
###########################################################################################
#########################################DEPORTISTA########################################
###########################################################################################
#------------------------------------------------------------------------------------------

# 1. Config:
        #Preprocessing: -
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetDeportistas(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
def athlete_pos_svm_spacy_within_prep(count_tweets):
    config = Configuration([
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.SVM,
        TaggingMethod.SPACY,
        TypeDataSet.deportista
    )
    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #-------------------------------------------------------

    data_set = DataSetDeportistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetDeportistas(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
#---------------------------------------LIBRARY--CLASSIFIER-------------------------------
#########################################SPACY#######SVM##################################
#-----------------------------------------------------------------------------------------

def athlete_pos_svm_spacy_token(count_tweets):
    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.SVM,
        TaggingMethod.SPACY,
        TypeDataSet.deportista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetDeportistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE, REMOVE_STOP_WORDS, LOWERCASE, LEMMATIZE
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetDeportistas(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def athlete_pos_svm_token_stopword_lowercase_lemma(count_tweets):
    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT, 
            Preprocessing.REMOVE_STOP_WORDS,
            Preprocessing.LOWERCASE,
            Preprocessing.LEMMATIZE
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.SVM,
        TaggingMethod.SPACY,
        TypeDataSet.deportista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetDeportistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: REMOVE_ALPHA_NUMERIC
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetDeportistas(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def athlete_pos_svm_rm_alphanumeric(count_tweets):
    config = Configuration([
        Preprocessing.REMOVE_ALPHA_NUMERIC
    ], 
    EncodingMethod.POSTAGGING,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.deportista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetDeportistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: EMOTICONS, LINKS, NUM, PUNCT
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetDeportistas(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def athlete_pos_svm_emoji_link_num_punct(count_tweets):
    config = Configuration([
        Preprocessing.EMOTICONS,
        Preprocessing.LINKS,
        Preprocessing.NUM,
        Preprocessing.PUNCT
    ], 
    EncodingMethod.POSTAGGING,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.deportista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetDeportistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#-------------------------------------PREPROSSESING - POS----------------------------------------
###########################################################################################
#########################################YOUTUBER##########################################
###########################################################################################
#------------------------------------------------------------------------------------------

# 1. Config:
        #Preprocessing: -
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetYoutubers(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
def youtuber_pos_svm_spacy_within_prep(count_tweets):
    config = Configuration([
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.SVM,
        TaggingMethod.SPACY,
        TypeDataSet.youtuber
    )
    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #-------------------------------------------------------

    data_set = DataSetYoutubers(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetYoutubers(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
#---------------------------------------LIBRARY--CLASSIFIER-------------------------------
#########################################SPACY#######SVM##################################
#-----------------------------------------------------------------------------------------

def youtuber_pos_svm_spacy_token(count_tweets):
    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.SVM,
        TaggingMethod.SPACY,
        TypeDataSet.youtuber
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetYoutubers(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE, REMOVE_STOP_WORDS, LOWERCASE, LEMMATIZE
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetYoutubers(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def youtuber_pos_svm_token_stopword_lowercase_lemma(count_tweets):
    config = Configuration([
        Preprocessing.FIX_HASHTAG_TEXT, 
        Preprocessing.REMOVE_STOP_WORDS,
        Preprocessing.LOWERCASE,
        Preprocessing.LEMMATIZE
    ], 
    EncodingMethod.POSTAGGING,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.youtuber
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetYoutubers(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: REMOVE_ALPHA_NUMERIC
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetYoutubers(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def youtuber_pos_svm_rm_alphanumeric(count_tweets):
    config = Configuration([
        Preprocessing.REMOVE_ALPHA_NUMERIC
    ], 
    EncodingMethod.POSTAGGING,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.youtuber
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetYoutubers(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: EMOTICONS, LINKS, NUM, PUNCT
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetYoutubers(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def youtuber_pos_svm_emoji_link_num_punct(count_tweets):
    config = Configuration([
        Preprocessing.EMOTICONS,
        Preprocessing.LINKS,
        Preprocessing.NUM,
        Preprocessing.PUNCT
    ], 
    EncodingMethod.POSTAGGING,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.youtuber
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetYoutubers(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#--------------------------------------PREPROSSESING - POS---------------------------------------
###########################################################################################
###########################################TODOS###########################################
###########################################################################################
#------------------------------------------------------------------------------------------

# 1. Config:
        #Preprocessing: -
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetTodos(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
def all_pos_svm_spacy_within_prep(count_tweets):
    config = Configuration([
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.SVM,
        TaggingMethod.SPACY,
        TypeDataSet.all_results
    )
    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #-------------------------------------------------------

    data_set = DataSetTodos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetTodos(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
#---------------------------------------LIBRARY--CLASSIFIER-------------------------------
#########################################SPACY#######SVM##################################
#-----------------------------------------------------------------------------------------

def all_pos_svm_spacy_token(count_tweets):
    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.SVM,
        TaggingMethod.SPACY,
        TypeDataSet.all_results
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetTodos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE, REMOVE_STOP_WORDS, LOWERCASE, LEMMATIZE
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetTodos(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def all_pos_svm_token_stopword_lowercase_lemma(count_tweets):
    config = Configuration([
        Preprocessing.FIX_HASHTAG_TEXT, 
        Preprocessing.REMOVE_STOP_WORDS,
        Preprocessing.LOWERCASE,
        Preprocessing.LEMMATIZE
    ], 
    EncodingMethod.POSTAGGING,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.all_results
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetTodos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: REMOVE_ALPHA_NUMERIC
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetTodos(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def all_pos_svm_rm_alphanumeric(count_tweets):
    config = Configuration([
        Preprocessing.REMOVE_ALPHA_NUMERIC
    ], 
    EncodingMethod.POSTAGGING,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.all_results
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetTodos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: EMOTICONS, LINKS, NUM, PUNCT
        #Encoding: POS
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetTodos(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def all_pos_svm_emoji_link_num_punct(count_tweets):
    config = Configuration([
        Preprocessing.EMOTICONS,
        Preprocessing.LINKS,
        Preprocessing.NUM,
        Preprocessing.PUNCT
    ], 
    EncodingMethod.POSTAGGING,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.all_results
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetTodos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------PREPROSSESING - BIGRAM---------------------------------------
###########################################################################################
#########################################POLITICOS#########################################
###########################################################################################
#------------------------------------------------------------------------------------------

# 1. Config:
        #Preprocessing: -
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetPoliticos(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
def politic_big_svm_spacy_within_prep(count_tweets):
    config = Configuration([
        ], 
        EncodingMethod.BIGRAM,
        ClassificationMethod.SVM,
        TaggingMethod.SPACY,
        TypeDataSet.politico
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #-------------------------------------------------------

    data_set = DataSetPoliticos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetPoliticos(count_tweets)
# 4. Classifier: Classification()
#---------------------------------------LIBRARY--CLASSIFIER------------------------------
#########################################SPACY#######SVM#################################
#----------------------------------------------------------------------------------------

def politic_big_svm_spacy_token(count_tweets):
    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.BIGRAM,
        ClassificationMethod.SVM,
        TaggingMethod.SPACY,
        TypeDataSet.politico
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetPoliticos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE, REMOVE_STOP_WORDS, LOWERCASE, LEMMATIZE
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetPoliticos(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def politic_big_svm_token_stopword_lowercase_lemma(count_tweets):
    config = Configuration([
        Preprocessing.FIX_HASHTAG_TEXT, 
        Preprocessing.REMOVE_STOP_WORDS,
        Preprocessing.LOWERCASE,
        Preprocessing.LEMMATIZE
    ], 
    EncodingMethod.BIGRAM,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.politico
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetPoliticos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: REMOVE_ALPHA_NUMERIC
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetPoliticos(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def politic_big_svm_rm_alphanumeric(count_tweets):
    config = Configuration([
        Preprocessing.REMOVE_ALPHA_NUMERIC
    ], 
    EncodingMethod.BIGRAM,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.politico
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetPoliticos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: EMOTICONS, LINKS, NUM, PUNCT
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetPoliticos(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def politic_big_svm_emoji_link_num_punct(count_tweets):
    config = Configuration([
        Preprocessing.EMOTICONS,
        Preprocessing.LINKS,
        Preprocessing.NUM,
        Preprocessing.PUNCT
    ], 
    EncodingMethod.BIGRAM,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.politico
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetPoliticos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------PREPROSSESING - BIGRAM---------------------------------------
###########################################################################################
#########################################ARTISTA###########################################
###########################################################################################
#------------------------------------------------------------------------------------------

# 1. Config:
        #Preprocessing: -
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetArtistas(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
def artist_big_svm_spacy_within_prep(count_tweets):
    config = Configuration([
        ], 
        EncodingMethod.BIGRAM,
        ClassificationMethod.SVM,
        TaggingMethod.SPACY,
        TypeDataSet.artista
    )
    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #-------------------------------------------------------

    data_set = DataSetArtistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetArtistas(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
#---------------------------------------LIBRARY--CLASSIFIER-------------------------------
#########################################SPACY#######SVM##################################
#-----------------------------------------------------------------------------------------

def artist_big_svm_spacy_token(count_tweets):
    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.BIGRAM,
        ClassificationMethod.SVM,
        TaggingMethod.SPACY,
        TypeDataSet.artista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetArtistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE, REMOVE_STOP_WORDS, LOWERCASE, LEMMATIZE
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetArtistas(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def artist_big_svm_token_stopword_lowercase_lemma(count_tweets):
    config = Configuration([
        Preprocessing.FIX_HASHTAG_TEXT, 
        Preprocessing.REMOVE_STOP_WORDS,
        Preprocessing.LOWERCASE,
        Preprocessing.LEMMATIZE
    ], 
    EncodingMethod.BIGRAM,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.artista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetArtistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: REMOVE_ALPHA_NUMERIC
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetArtistas(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def artist_big_svm_rm_alphanumeric(count_tweets):
    config = Configuration([
        Preprocessing.REMOVE_ALPHA_NUMERIC
    ], 
    EncodingMethod.BIGRAM,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.artista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetArtistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: EMOTICONS, LINKS, NUM, PUNCT
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetArtistas(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def artist_big_svm_emoji_link_num_punct(count_tweets):
    config = Configuration([
        Preprocessing.EMOTICONS,
        Preprocessing.LINKS,
        Preprocessing.NUM,
        Preprocessing.PUNCT
    ], 
    EncodingMethod.BIGRAM,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.artista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetArtistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------PREPROSSESING - BIGRAM---------------------------------------
###########################################################################################
#########################################DEPORTISTA########################################
###########################################################################################
#------------------------------------------------------------------------------------------

# 1. Config:
        #Preprocessing: -
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetDeportistas(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
def athlete_big_svm_spacy_within_prep(count_tweets):
    config = Configuration([
        ], 
        EncodingMethod.BIGRAM,
        ClassificationMethod.SVM,
        TaggingMethod.SPACY,
        TypeDataSet.deportista
    )
    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #-------------------------------------------------------

    data_set = DataSetDeportistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetDeportistas(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
#---------------------------------------LIBRARY--CLASSIFIER-------------------------------
#########################################SPACY#######SVM##################################
#-----------------------------------------------------------------------------------------

def athlete_big_svm_spacy_token(count_tweets):
    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.BIGRAM,
        ClassificationMethod.SVM,
        TaggingMethod.SPACY,
        TypeDataSet.deportista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetDeportistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE, REMOVE_STOP_WORDS, LOWERCASE, LEMMATIZE
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetDeportistas(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def athlete_big_svm_token_stopword_lowercase_lemma(count_tweets):
    config = Configuration([
        Preprocessing.FIX_HASHTAG_TEXT, 
        Preprocessing.REMOVE_STOP_WORDS,
        Preprocessing.LOWERCASE,
        Preprocessing.LEMMATIZE
    ], 
    EncodingMethod.BIGRAM,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.deportista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetDeportistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: REMOVE_ALPHA_NUMERIC
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetDeportistas(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def athlete_big_svm_rm_alphanumeric(count_tweets):
    config = Configuration([
        Preprocessing.REMOVE_ALPHA_NUMERIC
    ], 
    EncodingMethod.BIGRAM,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.deportista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetDeportistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: EMOTICONS, LINKS, NUM, PUNCT
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetDeportistas(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def athlete_big_svm_emoji_link_num_punct(count_tweets):
    config = Configuration([
        Preprocessing.EMOTICONS,
        Preprocessing.LINKS,
        Preprocessing.NUM,
        Preprocessing.PUNCT
    ], 
    EncodingMethod.BIGRAM,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.deportista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetDeportistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#-------------------------------------PREPROSSESING - BIGRAM----------------------------------------
###########################################################################################
#########################################YOUTUBER##########################################
###########################################################################################
#------------------------------------------------------------------------------------------

# 1. Config:
        #Preprocessing: -
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetYoutubers(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
def youtuber_big_svm_spacy_within_prep(count_tweets):
    config = Configuration([
        ], 
        EncodingMethod.BIGRAM,
        ClassificationMethod.SVM,
        TaggingMethod.SPACY,
        TypeDataSet.youtuber
    )
    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #-------------------------------------------------------

    data_set = DataSetYoutubers(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetYoutubers(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
#---------------------------------------LIBRARY--CLASSIFIER-------------------------------
#########################################SPACY#######SVM##################################
#-----------------------------------------------------------------------------------------

def youtuber_big_svm_spacy_token(count_tweets):
    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.BIGRAM,
        ClassificationMethod.SVM,
        TaggingMethod.SPACY,
        TypeDataSet.youtuber
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetYoutubers(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE, REMOVE_STOP_WORDS, LOWERCASE, LEMMATIZE
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetYoutubers(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def youtuber_big_svm_token_stopword_lowercase_lemma(count_tweets):
    config = Configuration([
        Preprocessing.FIX_HASHTAG_TEXT, 
        Preprocessing.REMOVE_STOP_WORDS,
        Preprocessing.LOWERCASE,
        Preprocessing.LEMMATIZE
    ], 
    EncodingMethod.BIGRAM,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.youtuber
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetYoutubers(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: REMOVE_ALPHA_NUMERIC
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetYoutubers(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def youtuber_big_svm_rm_alphanumeric(count_tweets):
    config = Configuration([
        Preprocessing.REMOVE_ALPHA_NUMERIC
    ], 
    EncodingMethod.BIGRAM,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.youtuber
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetYoutubers(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: EMOTICONS, LINKS, NUM, PUNCT
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetYoutubers(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def youtuber_big_svm_emoji_link_num_punct(count_tweets):
    config = Configuration([
        Preprocessing.EMOTICONS,
        Preprocessing.LINKS,
        Preprocessing.NUM,
        Preprocessing.PUNCT
    ], 
    EncodingMethod.BIGRAM,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.youtuber
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetYoutubers(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#--------------------------------------PREPROSSESING - BIGRAM---------------------------------------
###########################################################################################
###########################################TODOS###########################################
###########################################################################################
#------------------------------------------------------------------------------------------

# 1. Config:
        #Preprocessing: -
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetTodos(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
def all_big_svm_spacy_within_prep(count_tweets):
    config = Configuration([
        ], 
        EncodingMethod.BIGRAM,
        ClassificationMethod.SVM,
        TaggingMethod.SPACY,
        TypeDataSet.all_results
    )
    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #-------------------------------------------------------

    data_set = DataSetTodos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetTodos(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------
#---------------------------------------LIBRARY--CLASSIFIER-------------------------------
#########################################SPACY#######SVM##################################
#-----------------------------------------------------------------------------------------

def all_big_svm_spacy_token(count_tweets):
    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.BIGRAM,
        ClassificationMethod.SVM,
        TaggingMethod.SPACY,
        TypeDataSet.all_results
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetTodos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: TOKENIZE, REMOVE_STOP_WORDS, LOWERCASE, LEMMATIZE
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetTodos(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def all_big_svm_token_stopword_lowercase_lemma(count_tweets):
    config = Configuration([
        Preprocessing.FIX_HASHTAG_TEXT, 
        Preprocessing.REMOVE_STOP_WORDS,
        Preprocessing.LOWERCASE,
        Preprocessing.LEMMATIZE
    ], 
    EncodingMethod.BIGRAM,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.all_results
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetTodos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: REMOVE_ALPHA_NUMERIC
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetTodos(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def all_big_svm_rm_alphanumeric(count_tweets):
    config = Configuration([
        Preprocessing.REMOVE_ALPHA_NUMERIC
    ], 
    EncodingMethod.BIGRAM,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.all_results
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetTodos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config
#-----------------------------------------------------------------------------------------
# 1. Config:
        #Preprocessing: EMOTICONS, LINKS, NUM, PUNCT
        #Encoding: BIGRAM
        #Classifier: SVM
        #Tagging: SPACY
# 2. type_task: VALIDATE_MODEL
# 3. DataSet: DataSetTodos(count_tweets)
# 4. Classifier: Classification()
#-----------------------------------------------------------------------------------------

def all_big_svm_emoji_link_num_punct(count_tweets):
    config = Configuration([
        Preprocessing.EMOTICONS,
        Preprocessing.LINKS,
        Preprocessing.NUM,
        Preprocessing.PUNCT
    ], 
    EncodingMethod.BIGRAM,
    ClassificationMethod.SVM,
    TaggingMethod.SPACY,
    TypeDataSet.all_results
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetTodos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#---------------------------------------LIBRERY--------------------------------------------
###########################################################################################
#########################################STANZA############################################
###########################################################################################
#------------------------------------------------------------------------------------------

#--------------------------------------------#POLITICO#-----------------------------------#
def politic_pos_svm_stanza_token(count_tweets):
    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.SVM,
        TaggingMethod.STANZA,
        TypeDataSet.politico
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetPoliticos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------------#ARTISTA#------------------------------------#
def artist_pos_svm_stanza_token(count_tweets):
    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.SVM,
        TaggingMethod.STANZA,
        TypeDataSet.artista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetArtistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------------#DEPORTISTA#---------------------------------#
def athlete_pos_svm_stanza_token(count_tweets):

    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.SVM,
        TaggingMethod.STANZA,
        TypeDataSet.deportista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetDeportistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------------#YOUTUBER#-----------------------------------#

def youtuber_pos_svm_stanza_token(count_tweets):

    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.SVM,
        TaggingMethod.STANZA,
        TypeDataSet.youtuber
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetYoutubers(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------------#TODOS#-----------------------------------#

def all_pos_svm_stanza_token(count_tweets):

    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.SVM,
        TaggingMethod.STANZA,
        TypeDataSet.all_results
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetTodos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#-------------------------------------CLASSIFIER-----ENCODING------------------------------
###########################################################################################
#########################################BAYES######POSTAGGING#############################
###########################################################################################
#------------------------------------------------------------------------------------------

#---------------------------------------#POLITICO#----------------------------------------#
def politic_pos_bayes_spacy_token(count_tweets):
    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.BAYES,
        TaggingMethod.SPACY,
        TypeDataSet.politico
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetPoliticos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------------#ARTISTA#------------------------------------#
def artist_pos_bayes_spacy_token(count_tweets):
    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.BAYES,
        TaggingMethod.SPACY,
        TypeDataSet.artista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetArtistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------------#DEPORTISTA#---------------------------------#
def athlete_pos_bayes_spacy_token(count_tweets):

    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.BAYES,
        TaggingMethod.SPACY,
        TypeDataSet.deportista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetDeportistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------------#YOUTUBER#-----------------------------------#

def youtuber_pos_bayes_spacy_token(count_tweets):
    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.BAYES,
        TaggingMethod.SPACY,
        TypeDataSet.youtuber
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetYoutubers(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------------#TODOS#-----------------------------------#

def all_pos_bayes_spacy_token(count_tweets):

    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.BAYES,
        TaggingMethod.SPACY,
        TypeDataSet.all_results
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetTodos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#-------------------------------------CLASSIFIER-----ENCODING------------------------------
###########################################################################################
#########################################BAYES########BIGRAM###############################
###########################################################################################
#------------------------------------------------------------------------------------------

#---------------------------------------#POLITICO#----------------------------------------#
def politic_big_bayes_spacy_token(count_tweets):

    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.BIGRAM,
        ClassificationMethod.BAYES,
        TaggingMethod.SPACY,
        TypeDataSet.politico
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetPoliticos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------------#ARTISTA#------------------------------------#
def artist_big_bayes_spacy_token(count_tweets):
    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.BIGRAM,
        ClassificationMethod.BAYES,
        TaggingMethod.SPACY,
        TypeDataSet.artista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetArtistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------------#DEPORTISTA#---------------------------------#
def athlete_big_bayes_spacy_token(count_tweets):
    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.BIGRAM,
        ClassificationMethod.BAYES,
        TaggingMethod.SPACY,
        TypeDataSet.deportista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetDeportistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------------#YOUTUBER#-----------------------------------#

def youtuber_big_bayes_spacy_token(count_tweets):
    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.BIGRAM,
        ClassificationMethod.BAYES,
        TaggingMethod.SPACY,
        TypeDataSet.youtuber
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetYoutubers(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------------#TODOS#-----------------------------------#

def all_big_bayes_spacy_token(count_tweets):
    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.BIGRAM,
        ClassificationMethod.BAYES,
        TaggingMethod.SPACY,
        TypeDataSet.all_results
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetTodos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#-------------------------------------CLASSIFIER-----------ENCODING------------------------
###########################################################################################
##################################LOGISTIC REGRESSION######POSTAGGING#######################
###########################################################################################
#------------------------------------------------------------------------------------------

#---------------------------------------#POLITICO#----------------------------------------#
def politic_pos_rl_spacy_token(count_tweets):
    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.LOGISTIC_REGRESSION,
        TaggingMethod.SPACY,
        TypeDataSet.politico
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetPoliticos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------------#ARTISTA#------------------------------------#
def artist_pos_rl_spacy_token(count_tweets):
    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.LOGISTIC_REGRESSION,
        TaggingMethod.SPACY,
        TypeDataSet.artista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetArtistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------------#DEPORTISTA#---------------------------------#
def athlete_pos_rl_spacy_token(count_tweets):
    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.LOGISTIC_REGRESSION,
        TaggingMethod.SPACY,
        TypeDataSet.deportista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetDeportistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------------#YOUTUBER#-----------------------------------#

def youtuber_pos_rl_spacy_token(count_tweets):

    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.LOGISTIC_REGRESSION,
        TaggingMethod.SPACY,
        TypeDataSet.youtuber
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetYoutubers(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------------#TODOS#-----------------------------------#

def all_pos_rl_spacy_token(count_tweets):

    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.POSTAGGING,
        ClassificationMethod.LOGISTIC_REGRESSION,
        TaggingMethod.SPACY,
        TypeDataSet.all_results
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetTodos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#-------------------------------------CLASSIFIER-----ENCODING------------------------------
###########################################################################################
##################################LOGISTC REGRESSION########BIGRAM###########################
###########################################################################################
#------------------------------------------------------------------------------------------

#---------------------------------------#POLITICO#----------------------------------------#
def politic_big_rl_spacy_token(count_tweets):

    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.BIGRAM,
        ClassificationMethod.LOGISTIC_REGRESSION,
        TaggingMethod.SPACY,
        TypeDataSet.politico
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetPoliticos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------------#ARTISTA#------------------------------------#
def artist_big_rl_spacy_token(count_tweets):

    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.BIGRAM,
        ClassificationMethod.LOGISTIC_REGRESSION,
        TaggingMethod.SPACY,
        TypeDataSet.artista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetArtistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------------#DEPORTISTA#---------------------------------#
def athlete_big_rl_spacy_token(count_tweets):

    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.BIGRAM,
        ClassificationMethod.LOGISTIC_REGRESSION,
        TaggingMethod.SPACY,
        TypeDataSet.deportista
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetDeportistas(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------------#YOUTUBER#-----------------------------------#

def youtuber_big_rl_spacy_token(count_tweets):

    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.BIGRAM,
        ClassificationMethod.LOGISTIC_REGRESSION,
        TaggingMethod.SPACY,
        TypeDataSet.youtuber
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetYoutubers(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

#--------------------------------------------#TODOS#-----------------------------------#

def all_big_rl_spacy_token(count_tweets):

    config = Configuration([
            Preprocessing.FIX_HASHTAG_TEXT
        ], 
        EncodingMethod.BIGRAM,
        ClassificationMethod.LOGISTIC_REGRESSION,
        TaggingMethod.SPACY,
        TypeDataSet.all_results
    )

    #-----------------VALIDATE_MODEL------------------------
    type_task = TypeTask.VALIDATE_MODEL
    #------------------------------------------------------

    data_set = DataSetTodos(count_tweets)

    classifier = Classification()
    pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)
    return pipeline.run(), config

# pipeline = TwitterPipeline(data_set, classifier, type_task, model = dict_config_model['model'], config = dict_config_model['config'], text = ['@pepe esta informando sobre covid'])

# pipeline = TwitterPipeline(config, classifier, type_task, dataset=data_set)

def save_(list_count_tweets_x_user_to_download):

    list_dataset = []
    list_prep = []
    list_encoding = []
    list_library = []
    list_classifier = []
    list_total_classes = []
    list_total_sentences = []
    list_count_tweets_x_user = []
    list_accuracy = []
    list_f1 = []
    list_recall = []

    # list_method = [politic_pos_svm_spacy_within_prep, politic_pos_svm_spacy_token, politic_pos_svm_token_stopword_lowercase_lemma, politic_pos_svm_rm_alphanumeric]

    list_method = [politic_pos_svm_spacy_within_prep, politic_pos_svm_spacy_token,
    politic_pos_svm_token_stopword_lowercase_lemma, politic_pos_svm_rm_alphanumeric,
    politic_pos_svm_emoji_link_num_punct, artist_pos_svm_spacy_within_prep,
    artist_pos_svm_spacy_token, artist_pos_svm_token_stopword_lowercase_lemma,
    artist_pos_svm_rm_alphanumeric, artist_pos_svm_emoji_link_num_punct, 
    athlete_pos_svm_spacy_within_prep, athlete_pos_svm_spacy_token, 
    athlete_pos_svm_token_stopword_lowercase_lemma,
    athlete_pos_svm_rm_alphanumeric, athlete_pos_svm_emoji_link_num_punct, 
    youtuber_pos_svm_spacy_within_prep, youtuber_pos_svm_spacy_token,
    youtuber_pos_svm_token_stopword_lowercase_lemma, youtuber_pos_svm_rm_alphanumeric,
    youtuber_pos_svm_emoji_link_num_punct, all_pos_svm_spacy_within_prep,
    all_pos_svm_spacy_token, all_pos_svm_token_stopword_lowercase_lemma,
    all_pos_svm_rm_alphanumeric, all_pos_svm_emoji_link_num_punct, politic_big_svm_spacy_within_prep,
    politic_big_svm_spacy_token, politic_big_svm_token_stopword_lowercase_lemma,
    politic_big_svm_rm_alphanumeric, politic_big_svm_emoji_link_num_punct,
    artist_big_svm_spacy_within_prep, artist_big_svm_spacy_token,
    artist_big_svm_token_stopword_lowercase_lemma, artist_big_svm_rm_alphanumeric,
    artist_big_svm_emoji_link_num_punct, athlete_big_svm_spacy_within_prep,
    athlete_big_svm_spacy_token, athlete_big_svm_token_stopword_lowercase_lemma,
    athlete_big_svm_rm_alphanumeric, athlete_big_svm_emoji_link_num_punct,
    youtuber_big_svm_spacy_within_prep, youtuber_big_svm_spacy_token,
    youtuber_big_svm_token_stopword_lowercase_lemma, youtuber_big_svm_rm_alphanumeric,
    youtuber_big_svm_emoji_link_num_punct, all_big_svm_spacy_within_prep,
    all_big_svm_spacy_token, all_big_svm_token_stopword_lowercase_lemma,
    all_big_svm_rm_alphanumeric, all_big_svm_emoji_link_num_punct,
    politic_pos_svm_stanza_token, artist_pos_svm_stanza_token, athlete_pos_svm_stanza_token, 
    youtuber_pos_svm_stanza_token, all_pos_svm_stanza_token,
    politic_pos_bayes_spacy_token, artist_pos_bayes_spacy_token,
    athlete_pos_bayes_spacy_token, youtuber_pos_bayes_spacy_token, all_pos_bayes_spacy_token,
    politic_big_bayes_spacy_token, artist_big_bayes_spacy_token, athlete_big_bayes_spacy_token,
    youtuber_big_bayes_spacy_token, all_big_bayes_spacy_token, politic_pos_rl_spacy_token, 
    artist_pos_rl_spacy_token, athlete_pos_rl_spacy_token, youtuber_pos_rl_spacy_token,
    all_pos_rl_spacy_token, politic_big_rl_spacy_token, artist_big_rl_spacy_token, 
    athlete_big_rl_spacy_token, youtuber_big_rl_spacy_token, all_big_rl_spacy_token,
    politic_bert_within_prep, politic_bert_spacy_token, politic_bert_token_stopword_lowercase_lemma,
    politic_bert_rm_alphanumeric, politic_bert_emoji_link_num_punct, artist_bert_within_prep,
    artist_bert_spacy_token, artist_bert_token_stopword_lowercase_lemma, artist_bert_rm_alphanumeric,
    artist_bert_emoji_link_num_punct, athlete_bert_within_prep, athlete_bert_spacy_token,
    athlete_bert_token_stopword_lowercase_lemma, athlete_bert_rm_alphanumeric,
    athlete_bert_emoji_link_num_punct, youtuber_bert_within_prep,
    youtuber_bert_spacy_token, youtuber_bert_token_stopword_lowercase_lemma,
    youtuber_bert_rm_alphanumeric, youtuber_bert_emoji_link_num_punct,
    all_bert_within_prep, all_bert_spacy_token, all_bert_token_stopword_lowercase_lemma,
    all_bert_rm_alphanumeric, all_bert_emoji_link_num_punct
    ]

    #values[0] -> accuracy, recall, f1score
    #values[1] -> config
    for index, count_tweets in enumerate(list_count_tweets_x_user_to_download):
        for idx, method in enumerate(list_method):
            # if idx == 26:
                # break
            values = method(count_tweets)
            tuple_data = values[0]
            config = values[1]

            if(len(tuple_data) == 5):
                accuracy = tuple_data[0]
                recall = tuple_data[1]
                f1 = tuple_data[2]
                total_classes = tuple_data[3]
                total_sentences = tuple_data[4]
                list_total_classes.append(total_classes)
                list_total_sentences.append(total_sentences)
                list_accuracy.append(int(float(accuracy)*100))
                list_recall.append(int(float(recall)*100))
                list_f1.append(int(float(f1)*100))
                list_library.append(config.tagging_method.name)

                if idx == len(list_method)-26 and index == len(list_count_tweets_x_user_to_download)-1:
                    dict_results = {
                        'ACCURACY': list_accuracy,
                        'RECALL': list_recall,
                        'F1': list_f1
                    }
                    dump(dict_results, f'DATA_FRAME/acc_rec_f1')
            else:
                X = tuple_data[0]
                y = tuple_data[1]
                dictionary_X_y = {'X': X, 'y': y}
                dump(dictionary_X_y, f'BERT_DATA_SET/{count_tweets}/{method.__name__}')
                
                if config.tagging_method != None:
                    list_library.append(config.tagging_method.name)
                else:
                    list_library.append('-')

            list_preprocessesing = config.preprocessing_methods

            if list_preprocessesing == []:
                list_prep.append('BASIC')
            elif list_preprocessesing[0] == Preprocessing.FIX_HASHTAG_TEXT and len(list_preprocessesing) == 1:
                list_prep.append('TOKENIZE')
            elif list_preprocessesing[0] == Preprocessing.FIX_HASHTAG_TEXT and list_preprocessesing[1]:
                list_prep.append('SYNTAX')
            elif list_preprocessesing[0] == Preprocessing.EMOTICONS:
                list_prep.append('ARTIFACT')
            elif list_preprocessesing[0] == Preprocessing.REMOVE_ALPHA_NUMERIC:
                list_prep.append('RM_ALPHA_NUM')

            list_encoding.append(config.encoding_method.name)

            list_classifier.append(config.classification_method.name)

            list_dataset.append(config.type_dataset)

            list_count_tweets_x_user.append(count_tweets)

    return list_dataset, list_prep, list_encoding, list_library, list_classifier, list_total_classes, list_total_sentences, list_count_tweets_x_user
    # return list_accuracy, list_recall, list_f1, list_dataset, list_prep, list_encoding, list_library, list_classifier, list_total_classes, list_total_sentences, list_count_tweets_x_user
    
    # dump(tuple_data, f'DATA_FRAME/tuple_data')

def fill_data_frame():

    list_dataset, list_prep, list_encoding, list_library, list_classifier, list_total_classes, list_total_sentences, list_count_tweets_x_user = save_(list_count_tweets_x_user_to_download)
    # list_accuracy, list_recall, list_f1, list_dataset, list_prep, list_encoding, list_library, list_classifier, list_total_classes, list_total_sentences, list_count_tweets_x_user = save_(list_count_tweets_x_user_to_download)
  
    df['total_classes'] = list_total_classes
    df['total_sentences'] = list_total_sentences
    df['count_tweets_x_user'] = list_count_tweets_x_user
    df['dataset'] = list_dataset
    df['preprocessing'] = list_prep
    df['encoding'] = list_encoding
    df['classifier'] = list_classifier
    df['library'] = list_library
    # df['accuracy'] = list_accuracy
    # df['recall'] = list_recall
    # df['f1'] = list_f1

    dump(df, f'DATA_FRAME/dataframe')

def modify_data_frame():

    df = load(f'DATA_FRAME/dataframe')

    # pass

if __name__ == "__main__":
    # save_(list_count_tweets_x_user_to_download)
    fill_data_frame()
    # pipeline.run()


configs = []
datasets = []
