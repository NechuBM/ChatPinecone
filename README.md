
# 🧮🗨️  Pinecone y ChatGPT | Tutorial Base de Datos Vectorial desde 0
### Convierte a ChatGPT en tu Asistente Personal de Búsqueda de Documentos

🚀 [Tutorial en Español | Youtube](https://youtu.be/adq0BFxQ4C0)

¿Qué es una base de datos vectorial? En este taller, exploraremos Pinecone, una de las bases de datos vectoriales líderes en la nube. Este tipo de bases de datos ha ganado una gran popularidad en los últimos meses. ¿Son realmente útiles? Lo comprobaremos en el tutorial paso a paso.

Además, ¿sabías que ChatGPT puede mantener conversaciones con documentos? En este taller de Python, descubriremos cómo hacerlo posible gracias a Pinecone. Olvídate de las limitaciones, ahora podrás conversar y explorar tus documentos de una manera completamente nueva.


Para desarrollar esta aplicación necesitaremos:
* Cuenta en Pinecone
* ChatGPT API
* Streamlit


## ¿Cómo funciona?
1. Divide documento en cachos (o chunks)
2. Crea los embeddings de los cachos de texto
3. Guarda los cachos y los embeddings en Pinecone
4. Busca los cachos más similares a la pregunta del usuario gracias a los embeddings.
5. Pasa los cachos más similares junto a la pregunta a ChatGPT que genera la respuesta


## Instalación
¡Usar este código es fácil! Aquí están los pasos:
1. Clone o descargue el repositorio en su máquina local.
2. Instale las bibliotecas requeridas ejecutando el siguiente comando en su terminal:
```console
pip install -r requirements.txt
```
3. Obtenga una clave API de OpenAI para usar su API ChatGPT.
4. Obtenga una clave API de Pinecone.
5. Ejecute la aplicación con el siguiente comando:
```console
streamlit run app.py
```
6. Suba un documento a la aplicación.
7. Escriba su pregunta y disfrute de la magia.


 🎥 [Más vídeos en mi canal de Youtube](https://www.youtube.com/@NechuBM)