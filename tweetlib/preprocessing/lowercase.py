#Establecer todas las palabras a minúscula
def lowercase(text: list):
    """[summary]

    Args:
        text (list[str]): [description]

    Returns:
        [type]: [description]
    """
    lowercase_text = []
    
    lowercase_text = [str(word.lower().strip()) for word in text]
    # for word in text:
    #     lowercase_text.append(word.lower().strip())
    return lowercase_text