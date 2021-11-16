# TweetLib

TweetLib es una herramienta diseñada para la detección de suplantación en redes sociales. Su propósito, es brindar al usuario un conjunto de funcionalidades y casos para tareas del Procesamiento de Lenguaje Natural (PLN), específicamente la suplantación de identidad en redes sociales.

## Contribuyendo a TweetLib

La implementación de referencia de la herramienta, es este repositorio (https://github.com/madelinkind/tweetlib). Se recomienda que para cualquier problema experimentado, se levante un [Issue](https://github.com/madelinkind/tweetlib/issues).

## Instalación

TweetLib es una herramienta multiplataforma, soportada en los principales Sistemas Operativos (Windows, MacOS y Linux). Para su uso, es necesario tener Python instalado en el sistema. Los pasos para su instalación y uso son muy sencillos:

```console
$ cd tweetlib
$ pip install -r requirements.txt
```


> **NOTA:** Es necesario configurar el acceso a la fuente de datos donde será almacenada la información.
>
> Para ello, es necesario proporcionar los datos de conexión en: `tweetlib/orm/settings.py`
>
> ```python
> DATABASES = {
>     'default': {
>         'ENGINE': 'django.db.backends.ADAPTER',
>         'NAME': 'DB_NAME',
>         'USER': 'USER_NAME',
>         'PASSWORD':'PASSWORD',
>         'HOST':'HOST',
>         'PORT':'PORT',
>     }
> }
> ```

## Modos de uso

En este apartado, se recomienda consultar la memoria asociada a esta herramienta.