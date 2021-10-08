import demo
import click

@click.group()
#función que agrupa
def main():
    pass

#Download all tweets from all DB users
@main.command()
def download_tweets_users():
    demo.download_all_tweets_users()

#Download all tweets of a user, if the user does not exist create it
@main.command()
@click.option('--username', '-u', help='Entre un usuario, para proceder a descargar los tweets de dicho usuario', type=str, required=True)
@click.option('--typeuser', '-tu', help='Especifique el tipo de usuario. Ej: artista, politico, youtuber, deportista...', type=str, default="")
def download_tweets_user(username, typeuser):
    demo.download_all_tweets_user(username, typeuser)

#Delete user
@main.command()
@click.option('--username', '-u', help='Entre el usuario que desee eliminar de la DB', type=str, required=True)
def delete_user(username):
    demo.delete_user(username)

#Add new text
@main.command()
@click.option('--username', '-u', prompt='Entre un usuario', help='Entre el usuario del que desea guardar un tweet', type=str, required=True)
@click.option('--text', '-t', prompt='Entre el texto que desea guardar', help='Entre el texto del usuario que desea guardar en BD', type=str, required=True)
def add_text_user(username, text):
    print('Starting the insertion of the text in DB...')
    demo.add_text_user(username, text)

#Validate model
@main.command()
@click.option('--preprocessing', '-p', help='Si desea, puede aplicar uno o varios preprocesamientos (Opcional): [Preprocesamiento] "Descripción" \n\n\n [TOKENIZE] "Separar por token" \n\n\n [ALPHA_NUMERIC] "Eliminar caracteres alpha numéricos" \n\n\n [NUM] "Eliminar números" \n\n\n [PUNCT] "Eliminar signos de puntuación" \n\n\n [EMAILS] "Eliminar emails" \n\n\n [LINKS] "Eliminar links" \n\n\n [LOWERCASE] "Setear tokens en minúscula" \n\n\n [LEMMATIZE] "Setear token a su lema base" \n\n\n [EMOTICONS] "Eliminar emoticons" \n\n\n [STOP_WORDS] "Eliminar palabras vacías" \n\n\n [MENTIONS] "Eliminar mentions" \n\n\n [HASHTAG] "Eliminar hashtags"', multiple=True, default=[str])
@click.option('--encoding', '-e', help='Debe entrar un encoding: [Encoding] "Descripción" \n\n\n [POS] "Características sintácticas del dicurso" \n\n\n [BIGRAM] "Bigrama nivel caracter" \n\n\n [TRIGRAM] "Trigrama nivel caracter" \n\n\n [POSALLGRAM] "POS + BIGRAM + TRIGRAM" \n\n\n [ALLGRAM] "BIGRAM + TRIGRAM"', type=str, required=True)
@click.option('--classifier', '-c', help='Debe entrar un clasificador: \n\n\n [Clasificador] "Descripción" \n\n\n [SVM] "Support vector machines" \n\n\n [LR] "Logistic Regression" \n\n\n [BAYES] "Naive Bayes"', type=str, required=True)
@click.option('--nlp_library', '-nlp', help='Debe entrar una librería: \n\n\n [Librería] "Descripción" \n\n\n [SPACY] \n\n\n [STANZA] \n\n\n "Librerías para el procesamiento del lenguaje natural"', type=str, required=True)
@click.option('--dataset', '-ds', help='Debe entrar un data set: [Data Set] "Descripción" [politico] \n\n\n [artista] \n\n\n [deportista] \n\n\n [youtuber] \n\n\n [todos] \n\n\n "Conjunto de datos"', type=str, required=True)
@click.option('--count_tweets_x_user', '-ctu', help='Debe entrar la cantidad de tweets por usuario a descargar', type=int, required=True)
def validate_model(preprocessing, encoding, classifier, nlp_library, dataset, count_tweets_x_user):
    print('Starting model validation...')
    demo.validate_model(preprocessing, encoding, classifier,nlp_library, dataset, count_tweets_x_user)

#Create model
@main.command()
@click.option('--id_model', '-id', help='Debe entrar el nombre del modelo que desea actualizar', type=str, required=True)
@click.option('--preprocessing', '-p', help='Si desea, puede aplicar uno o varios preprocesamientos (Opcional): [Preprocesamiento] "Descripción" \n\n\n [TOKENIZE] "Separar por token" \n\n\n [ALPHA_NUMERIC] "Eliminar caracteres alpha numéricos" \n\n\n [NUM] "Eliminar números" \n\n\n [PUNCT] "Eliminar signos de puntuación" \n\n\n [EMAILS] "Eliminar emails" \n\n\n [LINKS] "Eliminar links" \n\n\n [LOWERCASE] "Setear tokens en minúscula" \n\n\n [LEMMATIZE] "Setear token a su lema base" \n\n\n [EMOTICONS] "Eliminar emoticons" \n\n\n [STOP_WORDS] "Eliminar palabras vacías" \n\n\n [MENTIONS] "Eliminar mentions" \n\n\n [HASHTAG] "Eliminar hashtags"', multiple=True, default=[str])
@click.option('--encoding', '-e', help='Debe entrar un encoding: [Encoding] "Descripción" \n\n\n [POS] "Características sintácticas del dicurso" \n\n\n [BIGRAM] "Bigrama nivel caracter" \n\n\n [TRIGRAM] "Trigrama nivel caracter" \n\n\n [POSALLGRAM] "POS + BIGRAM + TRIGRAM" \n\n\n [ALLGRAM] "BIGRAM + TRIGRAM"', type=str, required=True)
@click.option('--classifier', '-c', help='Debe entrar un clasificador: \n\n\n [Clasificador] "Descripción" \n\n\n [SVM] "Support vector machines" \n\n\n [LR] "Logistic Regression" \n\n\n [BAYES] "Naive Bayes"', type=str, required=True)
@click.option('--nlp_library', '-nlp', help='Debe entrar una librería: \n\n\n [Librería] "Descripción" \n\n\n [SPACY] \n\n\n [STANZA] \n\n\n "Librerías para el procesamiento del lenguaje natural"', type=str, required=True)
@click.option('--dataset', '-ds', help='Debe entrar un data set: [Data Set] "Descripción" [politico] \n\n\n [artista] \n\n\n [deportista] \n\n\n [youtuber] \n\n\n [todos] \n\n\n "Conjunto de datos"', type=str, required=True)
@click.option('--count_tweets_x_user', '-ctu', help='Debe entrar la cantidad de tweets por usuario a descargar', type=int, required=True)
def create_model(id_model, preprocessing, encoding, classifier,nlp_library, dataset, count_tweets_x_user):
    print('Starting create model...')
    demo.create_model(id_model, preprocessing, encoding, classifier,nlp_library, dataset, count_tweets_x_user)

#Update model
@main.command()
@click.option('--id_model', '-id', help='Debe entrar el nombre del modelo que desea actualizar', type=str, required=True)
@click.option('--preprocessing', '-p', help='Si desea, puede aplicar uno o varios preprocesamientos (Opcional): [Preprocesamiento] "Descripción" \n\n\n [TOKENIZE] "Separar por token" \n\n\n [ALPHA_NUMERIC] "Eliminar caracteres alpha numéricos" \n\n\n [NUM] "Eliminar números" \n\n\n [PUNCT] "Eliminar signos de puntuación" \n\n\n [EMAILS] "Eliminar emails" \n\n\n [LINKS] "Eliminar links" \n\n\n [LOWERCASE] "Setear tokens en minúscula" \n\n\n [LEMMATIZE] "Setear token a su lema base" \n\n\n [EMOTICONS] "Eliminar emoticons" \n\n\n [STOP_WORDS] "Eliminar palabras vacías" \n\n\n [MENTIONS] "Eliminar mentions" \n\n\n [HASHTAG] "Eliminar hashtags"', multiple=True, default=[str])
@click.option('--encoding', '-e', help='Debe entrar un encoding: [Encoding] "Descripción" \n\n\n [POS] "Características sintácticas del dicurso" \n\n\n [BIGRAM] "Bigrama nivel caracter" \n\n\n [TRIGRAM] "Trigrama nivel caracter" \n\n\n [POSALLGRAM] "POS + BIGRAM + TRIGRAM" \n\n\n [ALLGRAM] "BIGRAM + TRIGRAM"', type=str, required=True)
@click.option('--classifier', '-c', help='Debe entrar un clasificador: \n\n\n [Clasificador] "Descripción" \n\n\n [SVM] "Support vector machines" \n\n\n [LR] "Logistic Regression" \n\n\n [BAYES] "Naive Bayes"', type=str, required=True)
@click.option('--nlp_library', '-nlp', help='Debe entrar una librería: \n\n\n [Librería] "Descripción" \n\n\n [SPACY] \n\n\n [STANZA] \n\n\n "Librerías para el procesamiento del lenguaje natural"', type=str, required=True)
@click.option('--dataset', '-ds', help='Debe entrar un data set: [Data Set] "Descripción" [politico] \n\n\n [artista] \n\n\n [deportista] \n\n\n [youtuber] \n\n\n [todos] \n\n\n "Conjunto de datos"', type=str, required=True)
@click.option('--count_tweets_x_user', '-ctu', help='Debe entrar la cantidad de tweets por usuario a descargar', type=int, required=True)
def update_model(id_model, preprocessing, encoding, classifier,nlp_library, dataset, count_tweets_x_user):
    print('Starting update model...')
    demo.update_model(id_model, preprocessing, encoding, classifier,nlp_library, dataset, count_tweets_x_user)

#Pedict
@main.command()
@click.option('--id_model', '-id', help='Debe entrar el nombre del modelo', type=str, required=True)
@click.option('--file', '-f', help='Debe entrar una dirección válida de fichero', type=str, required=True)
@click.option('--value', '-v', help='Si desea, puede entrar un valor (n) <= que el número de clases de tu data set, para obtener las (n) clases con mejor probabilidad (Opcional)', type=int)
def find_author(id_model, file, value):
    print('Starting find author...')
    demo.find_author(id_model, file, value)

if __name__ == '__main__':
    main()