from tweetlib.definitions import TypeTask
from tweetlib.classification.classification import Classification


dict_task = {
    TypeTask.VALIDATE_MODEL: Classification.validation_method,
    TypeTask.VALIDATE_MODEL_BERT: Classification.validate_kfold_bert,
    TypeTask.PREDICTION: Classification.predict_method,
    TypeTask.PREDICTION_BERT: Classification.bert_predict,
    TypeTask.MODEL_STORAGE: Classification.model_storage,
    TypeTask.MODEL_STORAGE_BERT: Classification.model_storage_bert,
    TypeTask.PCA: Classification.pca
}