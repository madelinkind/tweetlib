
doc = ["To", "be", "or", "not", "to", "be", "that", "is", "the", "question"]

def Unigram(doc):
    # create a list for the result
    result = list()
    # create a list that contains no punctuation
    # sentence = list()
    # parse through the document to add all tokens that are words to the sentence list
    # for token in doc:
    #     if token.is_alpha:
    #         sentence.append(token)
    # parse through the sentence while adding words in groups of two to the result
    for word in range(len(doc)):
        first_word = doc[word]
        element = [first_word]
        result.append(element)

    print(result)

if __name__ == '__main__':
    Unigram(doc)
    #https://www.youtube.com/watch?v=-GBgUy6ufUk
    #https://spacy.io/usage/rule-based-matching