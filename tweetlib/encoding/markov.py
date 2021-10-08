# import click
# from operator import mul
# from functools import reduce

# @click.command()
# @click.argument('name', default='guest')
# @click.argument('age', type=int, nargs=-1)
# @click.option("--name", prompt="Your name", help="Provide your name")
#Introducir en consola  "python markov.py --blue", para que te lo coja como verdadero.
# @click.option('--blue', is_flag=True, help='message in blue color')

# @click.argument('values', type=int)

# def hello(blue):
#     if blue:
#         click.secho('Hello there', fg='blue')
#     else:
#         click.secho('Hello there')

# def hello(name):
#     if name == 'PP':
#         click.secho(f'hello, my name is {name}', fg='yellow', bold=True)
#     else:
#         click.secho('hello there')
    # print(f'hello {name} {age}')
    # print(f'The sum is {sum(age)}')
    # print(f'The product is {reduce(mul,age,1)}')
# if __name__=='__main__':
#     hello()

def Markov():
    pass

# import os
# import sys

# # Django specific settings

# FILE = __file__
# DATA_SET_FOLDER = os.path.split(FILE)[0]
# TWEET_LIB_FOLDER = os.path.split(DATA_SET_FOLDER)[0]
# PROJECT_FOLDER = os.path.split(TWEET_LIB_FOLDER)[0]

# sys.path.append(PROJECT_FOLDER)
# sys.path.append(f"{PROJECT_FOLDER}/orm")
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")

# # import and setup django
# import django
# django.setup()

# from tweetlib.singleton import Utils
# from tweetlib.definitions import TaggingMethod
# nlp = Utils.load_nlp(TaggingMethod.SPACY)

# text = ['prueba', 'de', 'variable', 'global', 'made@gmail.com']

# def Markov(text):
#     text_within_emails = []

#     if type(text) == str:
#         doc = nlp(text)
#     else:
#         doc = nlp(" ".join(text))

#     text_within_emails = [str(word) for word in doc if not word.like_email]

#     print(text_within_emails)

# if __name__ == '__main__':
#     Markov(text)


#
# @click.command()
# @click.argument('word')
# @click.option('--p/--no-p', default=False)
# def output(word, p):
#     if p:
#         click.echo(word.upper())
#     else:
#         click.echo(word)

# if __name__ == '__main__':
#     output()