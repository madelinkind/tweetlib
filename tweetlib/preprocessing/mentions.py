# text = ["@made", "probando", "#vic", "@#h", "#d", "s@", "df#"]

def rm_mentions(text: list):
    # MENT = 0
    within_ment = []
    if type(text) == str:
        text = text.split(" ")
    for word in text:
        if word != '':
            if word[0] == '@':
                within_ment.append(word)
        else:
            within_ment.append(word)
        
    # within_ment = [word for word in text if word != '' and not '@' in word[0]]
        # if '@' in t[0]:
            # MENT = MENT + 1
    # print(within_ment)
    return within_ment

def count_mentions(text: list):
    # tx = text.split()
    # list_mentions = []
    if type(text) == str:
        text = text.split(" ")
        # for word in text:
        #     if word != '':
        #         if word[0] == '@':
        #             list_mentions.append(word)
    list_mentions = [word for word in text if word != '' and '@' in word[0]]
    return len(list_mentions)
# if __name__ == '__main__':
#     rm_mentions(text)