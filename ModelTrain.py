# DataFrame
import pandas as pd

# Matplot
import matplotlib.pyplot as plt
import numpy as np

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Tensorflow
import tensorflow as tf
from tensorflow import keras

# nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Word2vec
from gensim.models import Word2Vec

# Utility
import re
import os
from collections import Counter
import logging
import time
import pickle
import itertools

# Set log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Log GPU
tf.debugging.set_log_device_placement(True)

nltk.download('stopwords')

# SETTINGS

# DATASET
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
TRAIN_SIZE = 0.8

# TEXT CLEANING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# WORD2VEC
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 8
BATCH_SIZE = 1024

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# EXPORT
KERAS_MODEL = "model.h5"
WORD2VEC_MODEL = "model.w2v"
TOKENIZER_MODEL = "tokenizer.pkl"
ENCODER_MODEL = "encoder.pkl"

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
print(checkpoint_dir)
print(os.getcwd())
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# READ DATASET

# Dataset details
# target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
# ids: The id of the tweet ( 2087)
# date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
# flag: The query (lyx). If there is no query, then this value is NO_QUERY.
# user: the user that tweeted (robotickilldozr)
# text: the text of the tweet (Lyx is cool)


dataset_path = "training.csv"
print("Open file:", dataset_path)
df = pd.read_csv(dataset_path, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

print("Dataset size:", len(df))

df.head(5)

# Map target label to String
# 0 -> NEGATIVE
# 2 -> NEUTRAL
# 4 -> POSITIVE

decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}


def decode_sentiment(label):
    return decode_map[int(label)]


df.target = df.target.apply(lambda x: decode_sentiment(x))

target_cnt = Counter(df.target)

plt.figure(figsize=(16, 8))
plt.bar(target_cnt.keys(), target_cnt.values())
plt.title("Dataset labels distribution")

# Pre-Process dataset

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")


def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)


df.text = df.text.apply(lambda x: preprocess(x))

# Split train and test

df_train, df_test = train_test_split(df, test_size=1 - TRAIN_SIZE, random_state=42)
print("TRAIN size:", len(df_train))
print("TEST size:", len(df_test))

documents = [_text.split() for _text in df_train.text]

# Train and Save w2v model
# w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE,
#                                            window=W2V_WINDOW,
#                                            min_count=W2V_MIN_COUNT,
#                                            workers=8)
# w2v_model.build_vocab(documents)
# words = w2v_model.wv.vocab.keys()
# vocab_size = len(words)
# print("Vocab size", vocab_size)
# w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)
# w2v_model.save(WORD2VEC_MODEL)

# Load w2v Model

w2v_model = Word2Vec.load(WORD2VEC_MODEL)

# TOKENIZE TEXT

tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df_train.text)

vocab_size = len(tokenizer.word_index) + 1
print("Total words", vocab_size)

x_train = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(df_train.text),
                                                        maxlen=SEQUENCE_LENGTH)
x_test = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(df_test.text),
                                                       maxlen=SEQUENCE_LENGTH)

# LABEL ENCODER

labels = df_train.target.unique().tolist()
labels.append(NEUTRAL)
labels

encoder = LabelEncoder()
encoder.fit(df_train.target.tolist())

y_train = encoder.transform(df_train.target.tolist())
y_test = encoder.transform(df_test.target.tolist())

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print("y_train", y_train.shape)
print("y_test", y_test.shape)

# EMBEDDING LAYER

embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
print(embedding_matrix.shape)

embedding_layer = tf.keras.layers.Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix],
                                            input_length=SEQUENCE_LENGTH, trainable=False)

model = keras.models.load_model(KERAS_MODEL)
model.summary()


# DECODE


def decode_sentiment(score, include_neutral=True):
    if include_neutral:
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE


def predict(text, include_neutral=True):
    start_at = time.time()
    # Tokenize text
    x_test = keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score),
            "elapsed_time": time.time() - start_at}


# Confusion Matrix


y_pred_1d = []
y_test_1d = list(df_test.target)
scores = model.predict(x_test, verbose=1, batch_size=8000)
y_pred_1d = [decode_sentiment(score, include_neutral=False) for score in scores]


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=22)
    plt.yticks(tick_marks, classes, fontsize=22)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)


cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
plt.figure(figsize=(12, 12))
plot_confusion_matrix(cnf_matrix, classes=df_train.target.unique(), title="Confusion matrix")
plt.show()

print(classification_report(y_test_1d, y_pred_1d))

accuracy_score(y_test_1d, y_pred_1d)

pickle.dump(tokenizer, open(TOKENIZER_MODEL, "wb"), protocol=0)
pickle.dump(encoder, open(ENCODER_MODEL, "wb"), protocol=0)
