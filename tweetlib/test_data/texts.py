
class SpanishTextSamples(object):

    TEXT_NAVARROS = """
        Los navarros y navarras celebran hoy su gran día, conscientes del momento que vivimos.
        Con su enorme espíritu de superación y entrega ganarán el futuro que espera a nuestro país:
        un futuro próspero, igualitario, lleno de oportunidades para toda la
        sociedad, 75 $: 86$; 234 € ? 234€ "ojo", :) ;) :( :* @ # ! & ^ } [] { \ \ | ~ ` , etc
    """

    TEXT_COVID = """
        El estado de alarma funciona ⌛. Los jiennenses y malagueños apoyan con fuerza las
        medidas, con una recaudación de 45000$ y 50000€. Estamos :).
    """

    TEXT_CARACTERES_EMOTICONS_BASICO = """
        :) ;) :( :* [ ~ ! @ # $ % ^ & * ( ) _ + = { \ \ | / ? > < " ' ` €] 1 2 3 4 5 6 7 8 9 0 https://www.blablacar.es/
    """

    TEXT_CUBA = """
        La isla bella, pequeña y acojedora, la mejor.
    """

    TEXT_ANIMALS = """
        Más de un billón de personas dedican su vida a proteger y cuidar a sus 
        mascotas 🐶 🐵 🐱 🐔 🦎, alimentándolas 🥘🍗🍖🦴 y ofreciéndoles un 
        cálido cariño 😊😋🥰😚🥳🤗 :) ❥ ⚤ ✱ ♬.
    """

    @staticmethod
    def get_all_texts():
        return [
            SpanishTextSamples.TEXT_CARACTERES_EMOTICONS_BASICO,
            SpanishTextSamples.TEXT_NAVARROS,
            SpanishTextSamples.TEXT_COVID,
            SpanishTextSamples.TEXT_CUBA,
            SpanishTextSamples.TEXT_ANIMALS
        ]
