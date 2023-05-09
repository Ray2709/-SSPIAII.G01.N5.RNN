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