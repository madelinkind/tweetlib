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

from tweetlib.data_set.artistas import DataSetArtistas

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
    ClassificationMethod.SVM,
    TaggingMethod.STANZA,
    type_dataset = TypeDataSet.artista
)

type_task = TypeTask.VALIDATE_MODEL

data_set_artista = DataSetArtistas(1000)

classifier = Classification()

pipeline = TwitterPipeline(config, classifier, type_task, dataset= data_set_artista)

if __name__ == "__main__":
    pipeline.run()


