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
import numpy as np
import logging

# list_count_tweets_x_user_to_download = [50, 100, 200, 500, 1000]
list_count_tweets_x_user_to_download = [50, 100, 200]


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

#--------------------------------------PREPROSSESING - BIGRAM------------------------------
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
def politic_pos_svm_stanza_within_prep(count_tweets):
    config = Configuration([
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

def artist_pos_svm_stanza_within_prep(count_tweets):
    config = Configuration([
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

#------------------------------------------------------------------------------------------

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
def athlete_pos_svm_stanza_within_prep(count_tweets):

    config = Configuration([
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
#------------------------------------------------------------------------------------------

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

def youtuber_pos_svm_stanza_within_prep(count_tweets):

    config = Configuration([
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

#------------------------------------------------------------------------------------------

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

def all_pos_svm_stanza_within_prep(count_tweets):

    config = Configuration([
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
#------------------------------------------------------------------------------------------

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
def politic_pos_bayes_spacy_within_prep(count_tweets):
    config = Configuration([
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
def artist_pos_bayes_spacy_within_prep(count_tweets):
    config = Configuration([
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
def athlete_pos_bayes_spacy_within_prep(count_tweets):

    config = Configuration([
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

def youtuber_pos_bayes_spacy_within_prep(count_tweets):
    config = Configuration([
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

def all_pos_bayes_spacy_within_prep(count_tweets):

    config = Configuration([
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
def politic_big_bayes_spacy_within_prep(count_tweets):

    config = Configuration([
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
def artist_big_bayes_spacy_within_prep(count_tweets):
    config = Configuration([
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
def athlete_big_bayes_spacy_within_prep(count_tweets):
    config = Configuration([
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

def youtuber_big_bayes_spacy_within_prep(count_tweets):
    config = Configuration([
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

def all_big_bayes_spacy_within_prep(count_tweets):
    config = Configuration([
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
def politic_pos_rl_spacy_within_prep(count_tweets):
    config = Configuration([
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
def artist_pos_rl_spacy_within_prep(count_tweets):
    config = Configuration([
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
def athlete_pos_rl_spacy_within_prep(count_tweets):
    config = Configuration([
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

def youtuber_pos_rl_spacy_within_prep(count_tweets):

    config = Configuration([
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

def all_pos_rl_spacy_within_prep(count_tweets):

    config = Configuration([
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
def politic_big_rl_spacy_within_prep(count_tweets):

    config = Configuration([
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
def artist_big_rl_spacy_within_prep(count_tweets):

    config = Configuration([
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
def athlete_big_rl_spacy_within_prep(count_tweets):

    config = Configuration([
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

def youtuber_big_rl_spacy_within_prep(count_tweets):

    config = Configuration([
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

def all_big_rl_spacy_within_prep(count_tweets):

    config = Configuration([
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


def save_(list_count_tweets_x_user_to_download):

    #DataFrame
    df = pd.DataFrame(columns=['total_classes', 'total_sentences', 'count_tweets_x_user', 
                           'dataset', 'preprocessing', 'encoding', 'classifier', 
                           'library', 'accuracy', 'recall', 'f1'])

    cont = 0
    list_method = [
    politic_big_bayes_spacy_within_prep, artist_big_bayes_spacy_within_prep,
    athlete_big_bayes_spacy_within_prep, youtuber_big_bayes_spacy_within_prep,
    all_big_bayes_spacy_within_prep, politic_big_bayes_spacy_token, 
    artist_big_bayes_spacy_token, athlete_big_bayes_spacy_token,
    youtuber_big_bayes_spacy_token, all_big_bayes_spacy_token
    # politic_bert_within_prep, politic_bert_spacy_token, politic_bert_token_stopword_lowercase_lemma,
    # politic_bert_rm_alphanumeric, politic_bert_emoji_link_num_punct, artist_bert_within_prep,
    # artist_bert_spacy_token, artist_bert_token_stopword_lowercase_lemma, artist_bert_rm_alphanumeric,
    # artist_bert_emoji_link_num_punct, athlete_bert_within_prep, athlete_bert_spacy_token,
    # athlete_bert_token_stopword_lowercase_lemma, athlete_bert_rm_alphanumeric,
    # athlete_bert_emoji_link_num_punct, youtuber_bert_within_prep,
    # youtuber_bert_spacy_token, youtuber_bert_token_stopword_lowercase_lemma,
    # youtuber_bert_rm_alphanumeric, youtuber_bert_emoji_link_num_punct,
    # all_bert_within_prep, all_bert_spacy_token, all_bert_token_stopword_lowercase_lemma,
    # all_bert_rm_alphanumeric, all_bert_emoji_link_num_punct
    ]

    for index, count_tweets in enumerate(list_count_tweets_x_user_to_download):
        for idx, method in enumerate(list_method):

            #Para reanudar en la siguiente posicion del metodo que se registró en los logs
            # if index != 3:
            #     break
            # if idx < 45:
            #   continue

            values = method(count_tweets)
            tuple_data = values[0]
            config = values[1]

            accuracy = tuple_data[0]
            recall = tuple_data[1]
            f1 = tuple_data[2]
            total_classes = tuple_data[3]
            total_sentences = tuple_data[4]
            library = config.tagging_method.name

            list_preprocessesing = config.preprocessing_methods

            if list_preprocessesing == []:
                prep = 'BASIC'
            elif list_preprocessesing[0] == Preprocessing.FIX_HASHTAG_TEXT and len(list_preprocessesing) == 1:
                prep = 'TOKENIZE'
            elif list_preprocessesing[0] == Preprocessing.FIX_HASHTAG_TEXT and list_preprocessesing[1]:
                prep = 'SYNTAX'
            elif list_preprocessesing[0] == Preprocessing.EMOTICONS:
                prep = 'ARTIFACT'
            elif list_preprocessesing[0] == Preprocessing.REMOVE_ALPHA_NUMERIC:
                prep = 'RM_ALPHA_NUM'

            encoding = config.encoding_method.name

            classifier = config.classification_method.name

            dataset = config.type_dataset

            #Filling dataframe
            df.loc[cont] = pd.Series({'total_classes':total_classes, 'total_sentences':total_sentences, 
                                            'count_tweets_x_user': count_tweets, 'dataset':dataset, 
                                            'preprocessing':prep, 'encoding':encoding, 
                                            'classifier':classifier, 'library':library, 'accuracy':int(float(accuracy)*100), 
                                            'recall':int(float(recall)*100), 'f1': int(float(f1)*100)})

            dump(df, 'DATA_FRAME/dataframe')
            
            #Record logs
            logging.basicConfig(filename='log/record.log',
                            filemode='a',format='%(asctime)s : %(levelname)s : %(message)s',
                            datefmt='%d/%m/%y %H:%M:%S',
                            level=logging.INFO
                            )
            logging.info(f'[Posicion_list_count_tweet_x_user: {index} count_tweet_x_user: {count_tweets}], [Posicion_method: {idx} Nombre del Método: {method.__name__}], [INDEX_DATAFRAME: {cont}]')

            cont += 1

if __name__ == "__main__":
    save_(list_count_tweets_x_user_to_download)
