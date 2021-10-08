from spacymoji import Emoji
from tweetlib.singleton import Utils
from tweetlib.definitions import TaggingMethod
# import es_core_news_md
# from collections import Counter
# from spacy.tokens import retokenize

# TODO: use Singleton of nlp
# nlp = Singleton.get_nlp()
nlp = Utils.load_nlp(TaggingMethod.SPACY)

# text = ['🤾', '🏻\u200d', '♀', '️', 'La', 'selección', 'española', 'vence', 'a', 'Noruega', 'y', 'se', 'coloca', 'a', 'un', 'paso','de','ganar','el','Mundial','femenino','de','balonmano','👏','🏻','\n','¡','Muc','…','https://t.co/IbVGXUz8Lb']
# text = [ '\n', '🤾','♀', '👏', '🏻\u200d', '️', 'La', 'selección', 'española', 'vence', 'a', 'Noruega', 'y', '', '🏻','\n','¡','Muc','…', 'https://t.co/IbVGXUz8Lb']
# text = ['🤾', '🏻\u200d', '♀', '️', 'La', 'selección', 'española']
# text = ['🤾', '♀', '️👏', '️👏', 'La', 'selección', 'española']


# TODO: use snake case (snake_case vs. CamlCase) for function identifiers
def rm_emoticons(text: list):
    """Elimina los emoji de un corpus

    Args:
        text (list[str]): Lista de tokens.
        nlp ([type]): Biblioteca de spacy, para utilizar sus atributos, en este caso "emoji".

    Returns:
        list[str]: Lista sin emoji.
    """
    # nlp = es_core_news_lg.load()
    # list_within_emoji = []
    # emoji = Emoji(nlp)
    # if not nlp.has_pipe("emoji"):
    #     nlp.add_pipe(emoji, first=True)
    # doc = nlp(" ".join(text))
    # if doc._.has_emoji:
    #     list_within_emoji = [str(word) for word in doc if not word._.is_emoji]
    list_within_emoji = []

    # TODO: move config to library initialization location
    if type(text) == str:
        doc = nlp(text)
    else:
        doc = nlp(" ".join(text))

    list_within_emoji = [str(word) for word in doc if not word._.is_emoji]
    return list_within_emoji

def count_emoticons(text: list):
    """[summary]

    Args:
        text (list): [description]
        nlp ([type]): [description]
    """
    # nlp = es_core_news_md.load()
    # if not nlp.has_pipe("emoji"):
    #     nlp.add_pipe("emoji", first=True)
    # doc = nlp(text)
    if type(text) == str:
        doc = nlp(text)
    else:
        doc = nlp(" ".join(text))
    list_emoticons = [word for word in doc if word._.is_emoji]
    return len(list_emoticons)

# if __name__ == '__main__':
#     count_emoticons(text)

# if __name__ == '__main__':
#     CleanEmoticons(text)

    # if doc._.has_emoji == True:
    #         for i in doc:
    #             if i._.is_emoji == False:
    #                 list_within_emoji.append(i)
    #         return list_within_emoji
    #     else:
    #         return text