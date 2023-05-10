# %%
from tensorflow import keras

# %%
import pathlib
import random
import string
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.layers import Bidirectional,GRU,LSTM,Embedding
from tensorflow.keras.layers import Dense,MultiHeadAttention,LayerNormalization,Embedding,Dropout,Layer
from tensorflow.keras import Sequential,Input
from tensorflow.keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import sentence_bleu


file_path = pathlib.Path("spa.txt")  # Reemplaza "ruta/al/archivo" con la ruta real a tu archivo spa.txt
text_file = file_path.absolute().as_posix()


# Leer el archivo y almacenar los pares de oraciones en una lista 
with open(text_file, encoding='utf-8') as f:
    lines = f.read().split("\n")[:-1]
    text_pairs=[]
    for line in lines:
        english, spanish = line.split("\t")
        spanish = "[start] " + spanish + "[end]"
        text_pairs.append((english,spanish))
  
# Revolver aleatoriamente la lista de pares de oraciones y dividirla en conjuntos de entrenamiento, validación y prueba 
import random
random.shuffle(text_pairs)
num_val_samples=int(0.15*len(text_pairs))
num_train_samples=len(text_pairs)-2*num_val_samples
train_pairs=text_pairs[:num_train_samples]
val_pairs=text_pairs[num_train_samples:num_train_samples+num_val_samples]
test_pairs=text_pairs[num_train_samples+num_val_samples:]

# Establecer una cadena de caracteres para quitar los signos de puntuación del texto 
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[","")
strip_chars = strip_chars.replace("]","")

# Definir una función para estandarizar el texto eliminando los signos de puntuación y convirtiéndolo a minúsculas 
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(strip_chars)}]",""
    )

# Establecer el tamaño del vocabulario y la longitud máxima de las secuencias 
vocab_size = 15000
sequence_length = 20

# Crear objetos TextVectorization para vectorizar el texto de entrada y salida 
source_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
target_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)

# Crear listas de texto para entrenamiento y prueba en inglés y español 
train_english_texts = [pair[0] for pair in train_pairs]
train_spanish_texts = [pair[1] for pair in train_pairs]
test_eng_texts = [pair[0] for pair in test_pairs]
test_spa_texts = [pair[1] for pair in test_pairs]

# Adaptar los objetos TextVectorization a los datos de entrenamiento 
source_vectorization.adapt(train_english_texts)
target_vectorization.adapt(train_spanish_texts)

# Establecer el tamaño del lote para el entrenamiento 
batch_size = 64

# Definir una función para formatear los datos de entrada y salida en tensores para el entrenamiento 

def format_dataset(eng, spa):
    eng = source_vectorization(eng)
    spa = target_vectorization(spa)
    return ({
        "english": eng,
        "spanish": spa[:, :-1],
    }, spa[:, 1:])

# Función para crear un conjunto de datos a partir de un par de textos en inglés y español
def make_dataset(pairs):
    # Se extraen los textos en inglés y español de cada par y se convierten a listas
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    
    # Se crea un objeto Dataset de TensorFlow a partir de los textos en inglés y español
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    
    # Se divide el conjunto de datos en lotes del tamaño especificado por batch_size
    dataset = dataset.batch(batch_size)
    
    # Se aplica la función format_dataset a cada lote de datos en paralelo
    dataset = dataset.map(format_dataset, num_parallel_calls=4)
    
    # Se mezcla el conjunto de datos y se crea una caché para acelerar el acceso a los datos
    return dataset.shuffle(2048).prefetch(16).cache()

# Se crean los conjuntos de datos de entrenamiento y validación a partir de los pares de textos correspondientes
train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)


from tensorflow.keras.layers import Input, Embedding, GRU, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy


# Definir la dimensión de los vectores de incrustación
embedding_dim = 256

# Definir la dimensión de la capa GRU (Capa de procesamiento)
gru_units = 1024

# Definir la capa de entrada
encoder_input = Input(shape=(sequence_length,), name='english')

# Definir la capa de incrustación
encoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='encoder_embedding')(encoder_input)

# Definir la capa GRU del codificador 
encoder_gru = GRU(units=gru_units, return_state=True, name='encoder_gru')
encoder_outputs, encoder_state = encoder_gru(encoder_embedding)

# Definir la capa de entrada del decodificador
decoder_input = Input(shape=(None,), name='spanish')

# Definir la capa de incrustación del decodificador
decoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='decoder_embedding')(decoder_input)

# Definir la capa GRU del decodificador
decoder_gru = GRU(units=gru_units, return_sequences=True, return_state=True, name='decoder_gru')
decoder_outputs, _ = decoder_gru(decoder_embedding, initial_state=encoder_state)

# Definir la capa densa del decodificador
decoder_dense = Dense(units=vocab_size, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

# Definir el modelo completo
model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_outputs)

# Compilar el modelo
model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(),metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])

# Entrenar el modelo
model.fit(train_ds, epochs=10, validation_data=val_ds)

# Seleccionar el primer elemento de val_ds
input_data, _ = next(val_ds.take(1).as_numpy_iterator())

# Obtener la predicción del modelo
prediction = model.predict(input_data)

# Imprimir la predicción
print(prediction[0])