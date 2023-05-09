from tensorflow import keras
import pathlib
import random
import string
import re
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.layers import GRU,LSTM
from tensorflow.keras.layers import Dense,Embedding
from tensorflow.keras import Input

import pathlib

file_path = pathlib.Path("ruta/al/archivo/spa.txt")  # Reemplaza "ruta/al/archivo" con la ruta real a tu archivo spa.txt
text_file = file_path.absolute().as_posix()


with open(text_file, encoding="utf-8") as f:
    lines = f.read().split("\n")[:-1]
    text_pairs = []
    for line in lines:
        english, spanish = line.split("\t")
        spanish = "[start] " + spanish + " [end]"
        text_pairs.append((english, spanish))


import random
print(random.choice(text_pairs))

random.shuffle(text_pairs)
num_val_samples=int(0.15*len(text_pairs))
num_train_samples=len(text_pairs)-2*num_val_samples
train_pairs=text_pairs[:num_train_samples]
val_pairs=text_pairs[num_train_samples:num_train_samples+num_val_samples]
test_pairs=text_pairs[num_train_samples+num_val_samples:]

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[","")
strip_chars = strip_chars.replace("]","")

# 
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(strip_chars)}]",""
    )

# 
vocab_size = 15000
sequence_length = 20