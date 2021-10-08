import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# verificar si debo importar DataSet, creo que no es necesario
from tweetlib.data_set.data_set import DataSet
from tweetlib.pipeline.execute_pipeline import TwitterPipeline
from tweetlib.config.configuration import Configuration
from tweetlib.definitions import Preprocessing, EncodingMethod, ClassificationMethod, TaggingMethod, TypeTask, TypeDataSet
from tweetlib.classification.classification import Classification

from transformers import BertForSequenceClassification

from tweetlib.data_set.politicos import DataSetPoliticos

from joblib import dump, load

config = Configuration([
        # Preprocessing.TOKENIZE,
        Preprocessing.FIX_HASHTAG_TEXT, 
        # Preprocessing.REMOVE_ALPHA_NUMERIC,
        # Preprocessing.NUM,
        # Preprocessing.PUNCT,
        # Preprocessing.EMAILS,
        # Preprocessing.LINKS,
        # Preprocessing.LOWERCASE,
        # Preprocessing.LEMMATIZE,
        # Preprocessing.EMOTICONS, 
        # Preprocessing.REMOVE_STOP_WORDS,
        # Preprocessing.MENTIONS,
        # Preprocessing.HASHTAG
    ], 
    EncodingMethod.POSTAGGING,
    # EncodingMethod.BIGRAM, 
    # ClassificationMethod.LOGISTIC_REGRESSION
    ClassificationMethod.LOGISTIC_REGRESSION,
    TaggingMethod.SPACY,
    type_dataset = TypeDataSet.politico
)

#id_model, text, n_value, model y config se van a pasar como kwrds como parametro de la instancia, es decir como diccionario, ej: id_mode = 1
#-----------------PREDICTION---------------------------
# type_task = TypeTask.PREDICTION_BERT
# id_model = 'BERT_10_10'
# # Insertar esto en DEMO
# # load the model
# # dict_config_model = load(f'models/{id_model}')
# output_dir = f'models/BERT/{id_model}'
# model = BertForSequenceClassification.from_pretrained(output_dir)
# config = load(f'models/BERT/BERT_10_10/config_prep')
# len_labels = load(f'models/BERT/BERT_10_10/len_labels')
# text = load(f'models/X_test')

#------------------------------------------------------
#-----------------PCA------------------------
# type_task = TypeTask.PCA
#------------------------------------------------------
#-----------------VALIDATE_MODEL------------------------
type_task = TypeTask.VALIDATE_MODEL
#------------------------------------------------------

#-----------------MODEL_STORAGE------------------------
# type_task = TypeTask.MODEL_STORAGE_BERT
#------------------------------------------------------

data_set_politicos = DataSetPoliticos(10)

classifier = Classification()

# pipeline = TwitterPipeline(data_set_politicos, classifier, type_task, model = dict_config_model['model'], config = dict_config_model['config'], text = ['@pepe esta informando sobre covid'])

pipeline = TwitterPipeline(config, classifier, type_task, dataset = data_set_politicos)
# pipeline = TwitterPipeline(config, classifier, type_task, model = model, len_labels = len_labels, text = text)


if __name__ == "__main__":
     pipeline.run()
