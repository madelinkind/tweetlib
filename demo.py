import os
import sys
# from os import remove

from crawler.io import load_users_list_from_file
import crawler.config as conf
from crawler.engine import TwitterEngine
from crawler.storage import DBStorage
# from datetime import datetime, date, time, timedelta
from datetime import datetime
import shutil

# ----------------------------------------------------------

# Django specific settings
sys.path.append('./orm')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")

# import and setup django
import django
django.setup()

# Import your models for use in your script
# pylint: disable=import-error
from db.models import Tweet, TwitterUser

#save and load model and config
from joblib import load
# import models

from tweetlib.classification.classification import Classification
from tweetlib.config.configuration import Configuration
from tweetlib.data_set.data_set import DataSet
from tweetlib.definitions import TypeDataSet, TypeTask, EncodingMethod, ClassificationMethod, TaggingMethod
import dict_commands
from tweetlib.pipeline.execute_pipeline import TwitterPipeline
#--------------------------DATASET-------------------------------------
from tweetlib.data_set.politicos import DataSetPoliticos
from tweetlib.data_set.artistas import DataSetArtistas
from tweetlib.data_set.deportistas import DataSetDeportistas
from tweetlib.data_set.youtubers import DataSetYoutubers
from tweetlib.data_set.todos import DataSetTodos 
# ----------------------------------------------------------

# # load users list

def download_all_tweets_users():
    """[summary]
    """
    users_map = map(lambda item: item['screen_name'], TwitterUser.objects.order_by('id').values('screen_name'))
    users_list = list(users_map)
    
    dbs = DBStorage()
    te = TwitterEngine(
        access_token = conf.ACCESS_TOKEN,
        access_token_secret = conf.ACCESS_TOKEN_SECRET,
        consumer_key = conf.CONSUMER_KEY,
        consumer_key_secret = conf.CONSUMER_KEY_SECRET,
        usernames = users_list,
    
        storage = dbs
    )
    
    te.download_tweets()
# user = 'Ninelconde'
def download_all_tweets_user(user, typeuser=None):
    """[summary]

    Args:
        user ([type]): [description]
        typeuser ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
#Si esta en DB lo descargamos directamente, de lo contrario insertamos en DB y descargamos los tweets.
    if not TwitterUser.objects.filter(screen_name=user).exists():
        #Hacer test, agregamos typeuser
        TwitterUser.objects.create(screen_name=user)

    users_list = [user]

    dbs = DBStorage()
    te = TwitterEngine(
        access_token = conf.ACCESS_TOKEN,
        access_token_secret = conf.ACCESS_TOKEN_SECRET,
        consumer_key = conf.CONSUMER_KEY,
        consumer_key_secret = conf.CONSUMER_KEY_SECRET,
        usernames = users_list,

        storage = dbs
    )

    te.download_tweets()

def delete_user(user_name):
    """[summary]

    Args:
        user_name ([type]): [description]
    """
    TwitterUser.objects.filter(screen_name=user_name).delete()
    print('Usuario eliminado satisfactoriamente de BD')

#Add text user
def add_text_user(user_name, text):
    """[summary]

    Args:
        user_name ([type]): [description]
        text ([type]): [description]
    """
    if TwitterUser.objects.filter(screen_name=user_name).exists():
        user = TwitterUser.objects.get(screen_name=user_name)
        Tweet.objects.create(twitter_user=user, tweet_text=text, tweet_date=datetime.now(), tweet_lang='es', is_retweet=False)
        print(f"El texto del usuario '{user_name}', fue agregado satisfactoriamente en la DB")
    else:
        print("El usuario no existe en BD. Por favor inserte antes el usuario en BD")
#------------------------------------------TASKS--------------------------------------------
classifier = Classification()

# list_prep = ['TOKENIZE']
# encoding = 'POS'
# classifier_type = 'SVM'
# tagging = 'SPACY'
# type_dataset = 'politico'

#Validate
def validate_model(list_prep: list, encoding: EncodingMethod, classifier_type: ClassificationMethod, nlp_library: TaggingMethod, type_dataset: TypeDataSet, n_tweets_x_user: int):
    """[summary]

    Args:
        list_prep (list): [description]
        encoding (EncodingMethod): [description]
        classifier_type (ClassificationMethod): [description]
        nlp_library (TaggingMethod): [description]
        type_dataset (TypeDataSet): [description]
    """
    config = configur(list_prep, encoding, classifier_type, type_dataset, nlp_library)
    #Si existe alg??n error en la entrada de los datos config = 1
    if config == 'error':
        return
    data_set = type_data_set(type_dataset, n_tweets_x_user)
    if data_set == 'error':
        return
    if encoding == 'BERT':
        type_task = TypeTask.VALIDATE_MODEL_BERT
    else:
        type_task = TypeTask.VALIDATE_MODEL
    run_pipeline(config, classifier_type, type_task, data_set)
    print("El modelo ha sido validado correctamente.")

#Classification.
#guardar modelos
#save model train in the file
# dump(model, 'models/model_train.jb')
#load the model
# model1 = load('models/model_train.jb')

# list_prep = ['TOKENIZE']
# encoding = 'POS'
# classifier_type = 'SVM'
# tagging = 'SPACY'
# type_dataset = 'politico'
# id_model = '3'
#Create model
def create_model(id_model, list_prep: list, encoding: EncodingMethod, classifier_type: ClassificationMethod, nlp_library: TaggingMethod, type_dataset: TypeDataSet, n_tweets_x_user: int):
    """[summary]

    Args:
        id_model ([type]): [description]
        list_prep (list): [description]
        encoding (EncodingMethod): [description]
        classifier_type (ClassificationMethod): [description]
        nlp_library (TaggingMethod): [description]
        type_dataset (TypeDataSet): [description]

    Returns:
        [type]: [description]
    """
    dir_model = f"models/{id_model}"
    if os.path.exists(dir_model):
        print(f"El modelo {id_model} ya existe, por lo que no puede ser creado. Si desea actualizar el modelo debe correr el siguiente comando: update-model")
        return
    # try:
    #     load(dir_model)
    #     return print(f"El modelo {id_model} ya existe, por lo que no puede ser creado. Si desea actualizar el modelo debe correr el siguiente comando: update-model")
    # except:
    else:
        if encoding == 'BERT':
            type_task = TypeTask.MODEL_STORAGE_BERT
        else:
            type_task = TypeTask.MODEL_STORAGE

        data_set = type_data_set(type_dataset, n_tweets_x_user)
        if data_set == 'error':
            return
        config = configur(list_prep, encoding, classifier_type, type_dataset, nlp_library)
        #Si alg??n par??metro es incorrecto config = 'error'
        if config == 'error':
            return
        run_pipeline(config, classifier_type, type_task, data_set, id_model = id_model)
        print(f"El modelo {id_model}, ha sido creado y almacenado satisfactoriamente.")

#Update model
def update_model(id_model, list_prep: list, encoding: EncodingMethod, classifier_type: ClassificationMethod, nlp_library: TaggingMethod, type_dataset: TypeDataSet, n_tweets_x_user: int):
    """[summary]

    Args:
        id_model ([type]): [description]
        list_prep (list): [description]
        encoding (EncodingMethod): [description]
        classifier_type (ClassificationMethod): [description]
        nlp_library (TaggingMethod): [description]
        type_dataset (TypeDataSet): [description]
    """
    dir_model = f"models/{id_model}"
    if not os.path.exists(dir_model):
        print(f"El modelo {id_model} no existe, por lo que no puede ser actualizado. Si desea crear el modelo debe correr el siguiente comando: create-model")
        return
    # try:
    #     load(dir_model)
    if encoding == 'BERT':
        type_task = TypeTask.MODEL_STORAGE_BERT
        shutil.rmtree(dir_model)
    else:
        type_task = TypeTask.MODEL_STORAGE

    data_set = type_data_set(type_dataset, n_tweets_x_user)
    if data_set == 'error':
        return
    config = configur(list_prep, encoding, classifier_type, type_dataset, nlp_library)
    if config == 'error':
        return
    run_pipeline(config, classifier_type, type_task, data_set, id_model = id_model)

    print(f"El modelo {id_model} ha sido actualizado satisfactoriamente.")

    # except:
    #     print(f"El modelo {id_model} no existe, por lo que no puede ser actualizado. Si desea crear el modelo debe correr el siguiente comando: create-model")

#Pedict
# n_value = 5
# text = ['Los incendios han vuelto a golpearnos duramente un verano m??s. Toca ayudar a quienes han sufrido los efectos de esta y otras cat??strofes naturales y eso es lo que va a hacer el Gobierno: habilitamos las ayudas necesarias para asegurar a los afectados una recuperaci??n justa. #CMin', 'Respeto a la integridad de los derechos y libertades de las ni??as y mujeres afganas. Sus derechos son los de todos y todas los  dem??cratas.', 'Los pueblos que defendemos la causa de Palestina, conmemoramos los 92 a??os del natalicio de Yasser Arafat. Su lucha por la libertad, la soberan??a y la autodeterminaci??n del pueblo palestino, est?? m??s vigente que nunca. ??Venezuela alza la voz, Viva Palestina Libre!']
# file = '/home/madelinkind/b.txt'
# id_model = '2'
def find_author(id_model, file, n_value: int = None):
    """[summary]

    Args:
        id_model ([type]): [description]
        file ([type]): [description]
        n_value (int, optional): [description]. Defaults to None.
    """
    if os.path.exists(file):
        with open(file, 'r') as f:
            text = f.readlines()
            # print(text)
            # for tweet in text:
            #     print(tweet)
    else:
        print(f"El fichero con direcci??n {file} no existe.")
        return

    dir_model = f'models/{id_model}'
    if os.path.exists(dir_model):
        if os.path.isfile(dir_model):
            dict_model_config = load(dir_model)
            config = dict_model_config['config']
            model = dict_model_config['model']
            type_task = TypeTask.PREDICTION
            run_pipeline(config, classifier, type_task, text = text, n_value = n_value, model = model)
        else:
            config = load(f'{dir_model}/config_prep')
            type_task = TypeTask.PREDICTION_BERT
            model = dir_model
            len_labels = load(f'{dir_model}/len_labels')
            run_pipeline(config, classifier, type_task, text = text, n_value = n_value, model = model, len_labels = len_labels)
    else:
        print(f"El modelo {id_model} no existe, debe entrar un modelo v??lido")
        models_exist = os.listdir('models')
        print(f"Los modelos que existen actualmente son los siguientes: {models_exist}")
        return
    # type_dataset = config.type_dataset
    # data_set = type_data_set(type_dataset)
    print("La predicci??n se ha realizado correctamente.")

def type_data_set(type_dataset, n_tweets_x_user):
    """[summary]

    Args:
        type_dataset ([type]): [description]

    Returns:
        [type]: [description]
    """
    error = 'error'
    if TypeDataSet.politico == type_dataset:
        data_set = DataSetPoliticos(n_tweets_x_user)
    elif TypeDataSet.artista == type_dataset:
        data_set = DataSetArtistas(n_tweets_x_user)
    elif TypeDataSet.deportista == type_dataset:
        data_set = DataSetDeportistas(n_tweets_x_user)
    elif TypeDataSet.youtuber == type_dataset:
        data_set = DataSetYoutubers(n_tweets_x_user)
    elif TypeDataSet.all_results == type_dataset:
        data_set = DataSetTodos(n_tweets_x_user)
    else:
        print("Debe entrar un valor v??lido de DataSet. Chequee c??mo debe escribir el dataset que desea. Ej: nombre-comando --help")
        return error
    return data_set

def configur(list_prep: list, encoding: EncodingMethod, classifier_type: ClassificationMethod, type_dataset: TypeDataSet, nlp_library: TaggingMethod = None):
    """[summary]

    Args:
        list_prep (list): [description]
        encoding (EncodingMethod): [description]
        classifier_type (ClassificationMethod): [description]
        nlp_library (TaggingMethod): [description]
        type_dataset (TypeDataSet): [description]

    Returns:
        [type]: [description]
    """
    error = 'error'
    prep = []
    if len(list_prep) != 0:
        for p in list_prep:
            if p in dict_commands.dict_prep:
                prep.append(dict_commands.dict_prep.get(p))
            else:
                print(f"El preprocesamiento {p} es incorrecto. Chequee c??mo debe escribir los diferentes preprocesamientos que desea aplicar, con el siguiente comando: nombre-comando --help")
                return error
    if encoding in dict_commands.dict_encoding:
        encoding = dict_commands.dict_encoding.get(encoding)
    else:
        print(f"El encoding {encoding} es incorrecto. Chequee c??mo debe escribir el encoding que desea aplicar, con el siguiente comando: nombre-comando --help")
        return error

    if classifier_type in dict_commands.dict_classifier:
        classifier_type = dict_commands.dict_classifier.get(classifier_type)
    else:
        print(f"El clasificador {classifier_type} es incorrecto. Chequee c??mo debe escribir el clasificador que desea aplicar, con el siguiente comando: nombre-comando --help")
        return error
    if nlp_library in dict_commands.dict_tagging:
            nlp_library = dict_commands.dict_tagging.get(nlp_library)
    elif nlp_library != None and classifier_type.name != 'BERT':
        print(f"La librer??a de nlp {nlp_library} es incorrecta. Chequee c??mo debe escribir la librer??a que desea utilizar, con el siguiente comando: nombre-comando --help")
        return error
    config = Configuration(prep, encoding, classifier_type, nlp_library, type_dataset)

    return config

def run_pipeline(config: Configuration, classifier: Classification, task: TypeTask, dataset: DataSet = None, text: list = None, id_model: str = None, n_value: int = None, model = None, len_labels: int = None):
    """[summary]

    Args:
        config (Configuration): [description]
        dataset (DataSet): [description]
        classifier (Classification): [description]
        task (TypeTask): [description]
        text (list, optional): [description]. Defaults to None.
        id_model (str, optional): [description]. Defaults to None.
        n_value (int, optional): [description]. Defaults to None.
        model ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if dataset is None:
        pipeline = TwitterPipeline(config, classifier, task, text = text, n_value = n_value, id_model = id_model, model = model, len_labels = len_labels)
    else:
        pipeline = TwitterPipeline(config, classifier = classifier, dataset = dataset, task = task, text = text, n_value = n_value, id_model = id_model, model = model)
    return pipeline.run()
