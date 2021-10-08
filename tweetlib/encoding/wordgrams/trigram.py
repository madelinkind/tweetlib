
doc = ["To", "be", "or", "not", "to", "be", "that", "is", "the", "question"]

def Trigram(doc):
    # create a list for the result
    result = list()
    # create a list that contains no punctuation
    # sentence = list()
    # parse through the document to add all tokens that are words to the sentence list
    # for token in doc:
    #     if token.is_alpha:
    #         sentence.append(token)
    # parse through the sentence while adding words in groups of two to the result
    for word in range(len(doc) - 2):
        first_word = doc[word]
        second_word = doc[word + 1]
        third_word = doc[word + 2]
        element = [first_word, second_word, third_word]
        result.append(element)

    print(result)

if __name__ == '__main__':
    Trigram(doc)
    #https://www.youtube.com/watch?v=-GBgUy6ufUk
    #https://spacy.io/usage/rule-based-matching