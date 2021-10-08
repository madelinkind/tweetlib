
doc = ["prueba", "ngram", "check"]

def Bigram(doc):
    # create a list for the result
    result = list()
    # create a list that contains no punctuation
    # sentence = list()
    # parse through the document to add all tokens that are words to the sentence list
    # for token in doc:
    #     if token.is_alpha:
    #         sentence.append(token)
    # parse through the sentence while adding words in groups of two to the result
    for word in range(len(doc) - 1):
        first_word = doc[word]
        second_word = doc[word + 1]
        element = [first_word, second_word]
        result.append(element)

    print(result)

if __name__ == '__main__':
    Bigram(doc)
    #https://www.youtube.com/watch?v=-GBgUy6ufUk
    #https://spacy.io/usage/rule-based-matching