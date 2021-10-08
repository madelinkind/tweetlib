import numpy as np
# from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score

import torch
import os
import sys

# Django specific settings
sys.path.append('./orm')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")

# import and setup django
import django
django.setup()

from joblib import dump
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig, EvalPrediction
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import random
import operator

#Python
# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# import matplotlib.colors as mcolors
import pandas as pd

# from sklearn.tree import DecisionTreeClassifier

from pylab import rcParams
from sklearn.metrics import precision_recall_fscore_support

# from imblearn.under_sampling import NearMiss
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.combine import SMOTETomek
# from imblearn.ensemble import BalancedBaggingClassifier

import os
import sys

FILE = __file__
DATA_SET_FOLDER = os.path.split(FILE)[0]
TWEET_LIB_FOLDER = os.path.split(DATA_SET_FOLDER)[0]
PROJECT_FOLDER = os.path.split(TWEET_LIB_FOLDER)[0]

sys.path.append(PROJECT_FOLDER)

# from tweetlib.data_set.data_set import DataSet
from tweetlib.config.configuration import Configuration
# from tweetlib.encoding import postagging
from tweetlib.definitions import ClassificationMethod
# import models
from joblib import dump, load
# from tweetlib.pipeline.execute_pipeline import TwitterPipeline

class Classification(object):

    @staticmethod
    def validate_kfold_bert(input_ids, attention_masks, labels, len_labels, len_sentences):
        kfold = range(5)
        accuracy_list = []
        recall_list = []
        f1_list = []
        for i in kfold:
            print(f'Iter {i+1}')
            accuracy, recall, f1 = Classification.run_model_bert(input_ids, attention_masks, labels, len_labels, True)
            accuracy_list.append(accuracy)
            recall_list.append(recall)
            f1_list.append(f1)

        accuracy_mean = np.array(accuracy_list).mean()
        accuracy_mean = f"{accuracy_mean:.2f}"

        recall_mean = np.array(recall_list).mean()
        recall_mean = f"{recall_mean:.2f}"

        f1_mean = np.array(f1_list).mean()
        f1_mean = f"{f1_mean:.2f}"

        print(f"Iter1_Accuracy: {accuracy_list[0]:.2f}, Iter2_Accuracy: {accuracy_list[1]:.2f}, Iter3_Accuracy: {accuracy_list[2]:.2f}, Iter4_Accuracy: {accuracy_list[3]:.2f}, Iter5_Accuracy: {accuracy_list[4]:.2f}")
        print(f"Accuracy Mean: {accuracy_mean}")
        print(f"Recall Mean: {recall_mean}")
        print(f"F1 Mean: {f1_mean}")
        return accuracy, recall, f1, len_labels, len_sentences

        #Validate
    @staticmethod
    def validation_method(X, y, method: ClassificationMethod):
        """[summary]

        Args:
            X ([type]): [description]
            y ([type]): [description]
            method (ClassificationMethod): [description]
        """
        print("Comenzamos validación...")
        accuracy=[]
        recall = []
        f1 = []
       #Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
        #dividimos en sets de entrenamiento y test
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        y = np.array(y)
        # skf = StratifiedKFold(n_splits=7, random_state=None)
        skf = StratifiedKFold(n_splits=5)

        skf.get_n_splits(X,y)
        # print(skf)
        # StratifiedKFold(n_splits=7, random_state=None, shuffle=False)
        for train_index, test_index in skf.split(X, y):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #ejecutamos el modelo
            print("Ejecutamos el modelo")
            model = Classification.run_model_balanced(X_train, y_train, method)

            # se realiza las predicciones en los datos de prueba usando predict()
            pred_y = model.predict(X_test)

            #Accuracy
            acc_score = accuracy_score(y_test, pred_y)
            accuracy.append(acc_score)

            #Recall
            rcll_score = recall_score(y_test, pred_y, average='micro')
            recall.append(rcll_score)

            #f1
            f_score = f1_score(y_test, pred_y, average='micro') 
            f1.append(f_score)

            #mostrar los resultados
            # conf_matrix = confusion_matrix(y_test, pred_y)
            # FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
            # FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
            # TP = np.diag(conf_matrix)
            # TN = conf_matrix.sum() - (FP + FN + TP)
            # ACC = (TP+TN)/(TP+FP+FN+TN)

            # recall = TP / (TP + FN)       # definition of recall
            # recall_all.append(recall)
            # precision = TP / (TP + FP)
            # f1 = 2 * ((precision + recall)/(precision*recall))
            # plt.figure(figsize=(12, 12))
            #Cambios recientes, borrar comentario luego
            # LABELS= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            #esta linea tenia una coma al final, se la he quitado
            # sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
            # plt.title("Confusion matrix")
            # plt.ylabel('True class')
            # plt.xlabel('Predicted class')
            # plt.show()
            print (classification_report(y_test, pred_y))
        # print(f"Accuracy: {accuracy}")
        # print(f"ACC: {ACC}")
        # print(f"RECALL: {recall}")
        # print(f"F1: {f1}")
        # print(f"PRECISION: {precision}")

        accuracy_mean = np.array(accuracy).mean()
        accuracy_mean = f"{accuracy_mean:.2f}"

        recall_mean = np.array(recall).mean()
        recall_mean = f"{recall_mean:.2f}"

        f1_mean = np.array(f1).mean()
        f1_mean = f"{f1_mean:.2f}"

        print(f"Iter1_Accuracy: {accuracy[0]:.2f}, Iter2_Accuracy: {accuracy[1]:.2f}, Iter3_Accuracy: {accuracy[2]:.2f}, Iter4_Accuracy: {accuracy[3]:.2f}, Iter5_Accuracy: {accuracy[4]:.2f}")
        print(f"Accuracy Mean: {accuracy_mean}")
        print(f"Recall Mean: {recall_mean}")
        print(f"F1 Mean: {f1_mean}")
        # return accuracy_mean, len(list(set(y))), len(X)
        return accuracy_mean, recall_mean, f1_mean, len(list(set(y))), len(X)
    
    #Save model bert
    @staticmethod
    def model_storage_bert(id_model: str, config: Configuration, method: ClassificationMethod, input_ids, attention_masks, labels, tokenizer, len_labels):

        model = Classification.run_model_bert(input_ids, attention_masks, labels, len_labels)

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        os.mkdir(f'models/BERT/{id_model}')
        output_dir = f'models/BERT/{id_model}'
        config_dir = f'{output_dir}/config_prep'
        len_label_dir = f'{output_dir}/len_labels'

        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
 
        print("Saving model to %s" % output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        #save config and len_labels
        dump(config, config_dir)
        dump(len_labels, len_label_dir)

    @staticmethod
    def load_model(id_model):

        device = Classification.cpu_or_gpu_availability()

        output_dir = f'/content/drive/MyDrive/Colab Notebooks/Models/{id_model}'
        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(output_dir)
        tokenizer = BertTokenizer.from_pretrained(output_dir)

        # Copy the model to the GPU. #Ver donde iria esta linea de codigo
        model.to(device)

        return model, tokenizer

    #Save model
    @staticmethod
    def model_storage(id_model: int, config: Configuration, X, y, method: ClassificationMethod):
        """[summary]

        Args:
            id_model (int): [description]
            config (Configuration): [description]
            X ([type]): [description]
            y ([type]): [description]
            method (ClassificationMethod): [description]
        """
        #Si el id_model existe -> actualizalo
        #Sino crealo 
        model = Classification.run_model_balanced(X, y, method)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        dict_config_model = {
            'config': config,
            'model': model
        }
        #guardar modelos
        #save model train in the file
        dump(dict_config_model, f'models/{id_model}')
        #load the model
        # model1 = load('models/id_model')

    @staticmethod
    def run_model_bert(input_ids, attention_masks, labels, len_labels, validate=None):

        device = Classification.cpu_or_gpu_availability()

        # Load BertForSequenceClassification, the pretrained BERT model with a single 
        # linear classification layer on top. 
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = len_labels, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )

        if device.type == 'cpu':
            # Tell pytorch to run this model on the GPU.Actualizar si el dispositivo = cpu
            model.cpu()
        else:
            model.cuda()

        #--------

        # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
        # I believe the 'W' stands for 'Weight Decay fix"
        optimizer = AdamW(model.parameters(),
                          lr = 5e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                        )

        # Combine the training inputs into a TensorDataset.
        dataset = TensorDataset(input_ids, attention_masks, labels)

        if validate != None:

            # Create a 90-10 train-validation split.

            # Calculate the number of samples to include in each set.
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size

            # Divide the dataset by randomly selecting samples.
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            print('{:>5,} training samples'.format(train_size))
            print('{:>5,} validation samples'.format(val_size))

            #-----
            # The DataLoader needs to know our batch size for training, so we specify it 
            # here. For fine-tuning BERT on a specific task, the authors recommend a batch 
            # size of 16 or 32.
            batch_size = 16

            # Create the DataLoaders for our training and validation sets.
            # We'll take training samples in random order. 
            train_dataloader = DataLoader(
                        train_dataset,  # The training samples.
                        sampler = RandomSampler(train_dataset), # Select batches randomly
                        batch_size = batch_size # Trains with this batch size.
                    )

            # For validation the order doesn't matter, so we'll just read them sequentially.
            validation_dataloader = DataLoader(
                        val_dataset, # The validation samples.
                        sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                        batch_size = batch_size # Evaluate with this batch size.
                    )
        else:
            batch_size = 16
            # Create the DataLoaders for our training and validation sets.
            # We'll take training samples in random order. 
            train_dataloader = DataLoader(
                        dataset,  # The training samples.
                        sampler = RandomSampler(dataset), # Select batches randomly
                        batch_size = batch_size # Trains with this batch size.
                    )

        # Number of training epochs. The BERT authors recommend between 2 and 4. 
        # We chose to run for 4, but we'll see later that this may be over-fitting the
        # training data.
        epochs = 1

        #---
        # Total number of training steps is [number of batches] x [number of epochs]. 
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)

        #--------------------

        # This training code is based on the `run_glue.py` script here:
        # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

        # Set the seed value all over the place to make this reproducible.
        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # We'll store a number of quantities such as training and validation loss, 
        # validation accuracy, and timings.

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        # For each epoch...
        for epoch_i in range(0, epochs):
        
            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            # Put the model into training mode. Don't be mislead--the call to 
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):
            
                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = Classification.format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                # Unpack this training batch from our dataloader. 
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the 
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: labels 
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because 
                # accumulating the gradients is "convenient while training RNNs". 
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                model.zero_grad()        

                # Perform a forward pass (evaluate the model on this training batch).
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # It returns different numbers of parameters depending on what arguments
                # arge given and what flags are set. For our useage here, it returns
                # the loss (because we provided labels) and the "logits"--the model
                # outputs prior to activation.
                output = model(b_input_ids, 
                                      token_type_ids=None, 
                                      attention_mask=b_input_mask, 
                                      labels=b_labels)
                loss = output.loss
                logits = output.logits 

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value 
                # from the tensor.
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)            

            # Measure how long this epoch took.
            training_time = Classification.format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))

            if validate != None:
              accuracy, recall, f1 = Classification.validation(model, validation_dataloader, epoch_i, avg_train_loss, training_time, total_t0)
              if epoch_i == 0:
                return accuracy, recall, f1
            else:
                return model

    @staticmethod
    def validation(model, validation_dataloader, epoch_i, avg_train_loss, training_time, total_t0): 

        device = Classification.cpu_or_gpu_availability()

        accuracy = 0 

        # We'll store a number of quantities such as training and validation loss, 
        # validation accuracy, and timings.
        training_stats = []
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
          
            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using 
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        
              
                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                output = model(b_input_ids, 
                                        token_type_ids=None, 
                                        attention_mask=b_input_mask,
                                        labels=b_labels)
                loss = output.loss
                logits = output.logits    
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            # preds.append(logits)
            # labels.append(label_ids)
            
            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += Classification.flat_accuracy(logits, label_ids)

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        if epoch_i == 0:
            accuracy = avg_val_accuracy
            y_pred = np.argmax(logits, axis=1).flatten()
            y_test = label_ids.flatten()
            _, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='micro')
            print(f'El recall es: {recall}')
            print(f'El f1 es: {f1}')
            print(classification_report(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = Classification.format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
        print("Validation complete!")
        print("Total training took {:} (h:mm:ss)".format(Classification.format_time(time.time()-total_t0)))
        return accuracy, recall, f1
    
    #Model
    @staticmethod
    def run_model_balanced(X_train, y_train, method: ClassificationMethod):
        """[summary]

        Args:
            X_train ([type]): [description]
            y_train ([type]): [description]
            method (ClassificationMethod): [description]

        Returns:
            [type]: [description]
        """
        #clasificador a utilizar
        if method == ClassificationMethod.LOGISTIC_REGRESSION:
            clf = LogisticRegression(C=1.0,penalty='l2',random_state=1,solver="newton-cg",class_weight="balanced")
        if method == ClassificationMethod.SVM:
            clf = SVC(kernel='linear', probability=True, class_weight="balanced")
        if method == ClassificationMethod.BAYES:
            #class_weight="balanced"
            clf = MultinomialNB()
        #Ajustamos los datos de entrenamiento en el clasificador usando fit(). Entrenar nuestro modelo
        print("Clasificando...")
        clf.fit(X_train, y_train)
        print("Modelo listo.")
        return clf

    #Predict
    @staticmethod
    def predict_method(model, X, n_value: int):
        """[summary]

        Args:
            model ([type]): [description]
            X ([type]): [description]
            n_value (int): [description]
        """
        results = model.predict_proba(X)
        prob_per_class_dictionary = dict(zip(model.classes_, results))
        
        for idx, _ in enumerate(prob_per_class_dictionary):
            results_ordered_by_probability = sorted(
                    zip(model.classes_, results[idx]),
                    key=lambda x: x[1],
                    reverse=True
                )
            if n_value != None and n_value <= len(results_ordered_by_probability):
                n_results_ordered_by_probability = results_ordered_by_probability[0:n_value]
                print(f"A continuación las {n_value} clases con mejor probabilidad para el texto {idx+1}: {n_results_ordered_by_probability}")

            else:
                n = len(results_ordered_by_probability)
                n_results_ordered_by_probability = results_ordered_by_probability[0:n]
                print(f"A continuación las {n} clases con sus probabilidades para el texto {idx+1}: {n_results_ordered_by_probability}")

    @staticmethod
    def take_predictions_bert(model, b_input_ids, b_attention_masks):

        device = Classification.cpu_or_gpu_availability()

        predictions = []

        batch_size = 16
        prediction_data = TensorDataset(b_input_ids, b_attention_masks)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

        # Prediction on test set

        print('Predicción de clases para {:,} oraciones de prueba...'.format(len(b_input_ids)))

        # Put model in evaluation mode
        model.eval()

        # Predict 
        for batch in prediction_dataloader:
          # Add batch to GPU
          batch = tuple(t.to(device) for t in batch)

          b_input_ids, b_input_mask = batch

          # Telling the model not to compute or store gradients, saving memory and 
          # speeding up prediction
          with torch.no_grad():
              # Forward pass, calculate logit predictions
              outputs = model(b_input_ids, token_type_ids=None, 
                              attention_mask=b_input_mask)

          logits = outputs[0]

          # Move logits and labels to CPU
          logits = logits.detach().cpu().numpy()

          predictions.append(logits)
        return predictions

    @staticmethod
    def bert_predict(model, input_ids, attention_masks, n_value, len_labels):
        # pred, true_labels = bert_predict(model, X_test, tokenizer)
        pred = Classification.take_predictions_bert(model, input_ids, attention_masks)
        dictionary_pred = {}

        print("A continuación se muestran las probabilidades para cada texto oredenadas por los mejores resultados")

        for arr_classes in pred:
            # print(pred)
            for idx_text, arr_class in enumerate(arr_classes):
                for idx, arr in enumerate(arr_class):
                    dictionary_pred[idx+1] = f"{arr:.2f}"
                print(f"Para el texto {idx_text+1}:")
                dictionary_pred_sort = sorted(dictionary_pred.items(), key=operator.itemgetter(1), reverse=True)
        
                if n_value != None and n_value <= len_labels:
                  for name in enumerate(dictionary_pred_sort[:n_value]):
                      print('La clase', name[1][0], 'con probabilidad:', dictionary_pred[name[1][0]])
        
                else:
                  for name in enumerate(dictionary_pred_sort):
                      print('La clase', name[1][0], 'con probabilidad:', dictionary_pred[name[1][0]])

    #PCA
    @staticmethod
    def pca(X, y):
        df = pd.DataFrame(X, columns=['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'LINK', 'EMTC', 'MENT', 'HASH'])
        df['CLASS'] = y
        print(df)
        features = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'LINK', 'EMTC', 'MENT', 'HASH']
        # Separating out the features
        x = df.loc[:, features].values
        # Separating out the target
        y = df.loc[:,['CLASS']].values
        # Standardizing the features
        x = StandardScaler().fit_transform(x)

        pca = PCA(n_components=2)
        data_components = pca.fit_transform(x)
        principalDf = pd.DataFrame(
            data = data_components,
            columns = ['pc1', 'pc2']
        )
        finalDf = pd.concat([principalDf, df[['CLASS']]], axis = 1)
        
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1, 1, 1) 
        ax.set_xlabel('PC 1', fontsize = 15)
        ax.set_ylabel('PC 2', fontsize = 15)
        ax.set_title('2 component PCA', fontsize = 20)
        targets = list(range(1, 11))
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', '#eaff00', '#fa14e3', '#79f711']
        for target, color in zip(targets,colors):
            indicesToKeep = finalDf['CLASS'] == target
            ax.scatter(
                finalDf.loc[indicesToKeep, 'pc1'],
                finalDf.loc[indicesToKeep, 'pc2'],
                c = color,
                s = 50
            )
        ax.legend(targets)
        ax.grid()

#####################Util BERT#####################
    @staticmethod
    def cpu_or_gpu_availability():
        if torch.cuda.is_available():    

                # Tell PyTorch to use the GPU.    
                device = torch.device("cuda")

                print('There are %d GPU(s) available.' % torch.cuda.device_count())

                print('We will use the GPU:', torch.cuda.get_device_name(0))

        # If not, CPU available...
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")
        return device
    
    @staticmethod
    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    # Function to calculate the accuracy of our predictions vs labels
    @staticmethod
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)


        # pd.DataFrame(pca.components_, columns=features, index=['PC1', 'PC2'])
        # pd.DataFrame(pca.components_, columns=features, index=['PC1', 'PC2']).to_csv('/home/madelinkind/Desktop/loadings.csv')
        # pd.DataFrame(pca.components_, columns=features, index=['PC1', 'PC2']).loc['PC2', :].abs().sort_values(ascending=False).to_csv('/home/madelinkind/Desktop/loadings.csv')
        #pd.DataFrame(pca.components_, columns=features, index=['PC1', 'PC2']).loc['PC2', :].abs().sort_values(ascending=False)
        #pd.DataFrame(pca.components_, columns=features, index=['PC1', 'PC2']).loc['PC1', :].abs().sort_values(ascending=False)
