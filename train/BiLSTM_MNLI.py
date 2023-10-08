# !wget http://nlp.stanford.edu/data/glove.6B.zip
# !unzip -q glove.6B.zip

#PARAMS
OUTPUT_SEQUENCE_LENGTH = 50
EMBEDDING_DIMENSION = 50
ALPHA = 0
BATCH_SIZE = 512

import re
import gc
import spacy
import numpy as np
import pandas as pd
# import en_core_web_sm
import tensorflow as tf
import tensorflow.keras as keras

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from spacy.lang.en.stop_words import STOP_WORDS
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from datasets import load_from_disk, Dataset

from tqdm import tqdm

mnli = load_from_disk("./MNLI")

len(mnli['train'])

sentences = mnli['train']['premise']+mnli['train']['hypothesis']
combined_logits = mnli['train']['combined_logits']
vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=20)
text_ds = tf.data.Dataset.from_tensor_slices(sentences).batch(128)
vectorizer.adapt(text_ds)
voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

import os
path_to_glove_file = "./glove.6B.50d.txt"

embeddings_index = {}
with open(path_to_glove_file, encoding='utf-8') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))
num_tokens = len(voc) + 2
embedding_dim = 50
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

logits = np.array(mnli['train']['combined_logits'])
x, y = logits[:, 0], logits[: ,1]
p = (x.argmax(axis=-1) == y.argmax(axis=-1)).sum()
print(p/len(logits))

print(len(logits))

import torch
embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
)
padded_sequences = embedding_layer(vectorizer(np.array([['Hi how are you?'],['Hello']])).numpy())
print(padded_sequences)
padded_sequences = np.concatenate((padded_sequences[0], padded_sequences[1], padded_sequences[0]*padded_sequences[1], tf.math.abs(padded_sequences[0] - padded_sequences[1])), axis=0)
# print(padded_sequences[0][0],padded_sequences[50][0],padded_sequences[100][0], padded_sequences[150][0])

cur_batch_size = 2048

# Define a function to preprocess a single batch of data
def preprocess_batch(batch_data):
  # print(batch_data)
  premises, hypotheses, combined_logits = batch_data['premise'], batch_data['hypothesis'], batch_data['combined_logits']
  print('converting to tf tensor')
  premises_data = embedding_layer(vectorizer(premises)).numpy()
  hypotheses_data = embedding_layer(vectorizer(hypotheses)).numpy()
  
  padded_sequences = np.concatenate((premises_data, hypotheses_data, premises_data*hypotheses_data, tf.math.abs(premises_data - hypotheses_data)), axis=1)
  # print(padded_sequences)
  # print(padded_sequences[1][0][0],padded_sequences[1][50][0],padded_sequences[1][100][0], padded_sequences[1][150][0])
  # print(combined_logits)
  combined_logits = np.array(combined_logits)
  # print(combined_logits)
  print('completed tf tensor')
  return padded_sequences, combined_logits

a, b = preprocess_batch(mnli['train'])

# alpha = 0
# del embedding_layer
# del sentences
# del combined_logits 
# del vectorizer 
# del text_ds 
# del voc 
# del word_index 
# del embedding_matrix
# del embeddings_index
# gc.collect()

class TotalLoss(tf.keras.losses.Loss):
    def __init__(self):
      super().__init__()
    def call(self, y_true, y_pred):
      # print(y_true)
      true_labels, teacher_logits= tf.split(y_true, num_or_size_splits=2, axis=1)
      # print(teacher_logits.shape, teacher_logits.shape)
      teacher_logits = teacher_logits[:, 0]
      # print(teacher_logits.shape, teacher_logits.shape)
      loss_wrt_teacher = tf.math.reduce_mean(tf.square(teacher_logits - y_pred))
      loss_wrt_true = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true_labels, logits=y_pred))
      # print("loss_wrt_teacher", loss_wrt_teacher,"loss_wrt_true", loss_wrt_true)
      return alpha*loss_wrt_true+(1-alpha)*loss_wrt_teacher

class CustomAccuracy(tf.keras.metrics.Metric):
  def __init__(self, name='accuracy', **kwargs):
    super(CustomAccuracy, self).__init__(name=name, **kwargs)
    self.total_correct = self.add_weight(name='total_correct', initializer='zeros')
    self.total_samples = self.add_weight(name='total_samples', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    true_labels, teacher_logits= tf.split(y_true, num_or_size_splits=2, axis=1)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(true_labels, axis=-1)
    y_true = tf.squeeze(y_true)
    cur_correct = tf.reduce_sum(tf.cast(tf.equal(y_pred, y_true), tf.float32))
    cur_smaples = tf.cast(tf.size(y_pred), tf.float32)

    self.total_correct.assign_add(cur_correct)
    self.total_samples.assign_add(cur_smaples)

  def result(self):
    return self.total_correct / self.total_samples

  def reset_state(self):
    self.total_correct.assign(0.0)
    self.total_samples.assign(0.0)


for alpha in tqdm([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
  try:
    del model
  except Exception as e:
    pass
  model = Sequential()
  model.add(Bidirectional(LSTM(300, return_sequences=False, input_shape=(None, 1))))  #Check what does input shape mean #Remove this 50 X 300(glove) & 50X64 
  model.add(Dropout(0.2)) # Avoid Overfitting
  model.add(Dense(400, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(3)) #OutputClasses
  # optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-07)
  optimizer = tf.keras.optimizers.Adam()
  model.compile(
      loss=TotalLoss(), 
      optimizer=optimizer, 
      metrics=[CustomAccuracy()],
      run_eagerly = True
  )

  # model.build(input_shape=(cur_batch_size, 200, 300))
  # model.summary()

  history = model.fit(
      a, b, 
      epochs=5,
      verbose=1,
      batch_size=cur_batch_size,
      validation_split = 0.1
  )

  print('alpha', alpha)
  print(history.history['val_accuracy'])
  print(history.history['accuracy'])



alpha = 0#best value
try:
    del model
except:
    pass

model = Sequential()
model.add(Bidirectional(LSTM(300, return_sequences=False, input_shape=(None, 1))))  #Check what does input shape mean #Remove this 50 X 300(glove) & 50X64 
model.add(Dropout(0.2)) # Avoid Overfitting
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3)) #OutputClasses
# optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-07)
optimizer = tf.keras.optimizers.Adam()
model.compile(
    loss=TotalLoss(), 
    optimizer=optimizer, 
    metrics=[CustomAccuracy()],
    run_eagerly = True
)

# model.build(input_shape=(cur_batch_size, 200, 300))
# model.summary()

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "mnli/my_checkpoint.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

history = model.fit(
    a, b, 
    epochs=15,
    verbose=1,
    batch_size=cur_batch_size,
    validation_split = 0.1
)

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path)

print(history.history['val_accuracy'])
print(history.history['accuracy'])


def get_test_data(data):
  sentences1, sentences2, labels = data['premise'], data['hypothesis'], data['label']
  
  sentences1_data = embedding_layer(vectorizer(sentences1)).numpy()
  sentences2_data = embedding_layer(vectorizer(sentences2)).numpy()

  padded_sequences = np.concatenate((sentences1_data, sentences2_data, sentences1_data*sentences2_data, tf.math.abs(sentences1_data - sentences2_data)), axis=1)
  labels = np.array(labels)
  return padded_sequences, labels

#Code for f1 score, accuracy and confusion matrix

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

test_padded_sequences, true_lables = get_test_data(mnli['validation_matched'])

predictions = model.predict(test_padded_sequences)
class_labels = np.argmax(predictions, axis=-1)

cm = confusion_matrix(true_lables, class_labels, labels=[0,1,2])
f1 = f1_score(true_lables, class_labels,  labels=[0,1,2], average = 'weighted')
accuracy = accuracy_score(true_lables, class_labels)

print("Confusion Matrix: \n", cm)
print("F1 Score:", f1)
print('Accuracy:', accuracy)


flag = np.array(class_labels == true_lables)
missed_values = []
identified_values = []
for i, f in enumerate(flag):
  if f:
    identified_values.append((mnli['validation_matched'][i]['premise'], mnli['validation_matched'][i]['hypothesis'], mnli['validation_matched'][i]['label']))

for i, f in enumerate(flag):
  if not f:
    missed_values.append((mnli['validation_matched'][i]['premise'], mnli['validation_matched'][i]['hypothesis'], mnli['validation_matched'][i]['label']))


print(missed_values)



test_padded_sequences, true_lables = get_test_data(mnli['validation_mismatched'])

predictions = model.predict(test_padded_sequences)
class_labels = np.argmax(predictions, axis=-1)

cm = confusion_matrix(true_lables, class_labels, labels=[0,1,2])
f1 = f1_score(true_lables, class_labels,  labels=[0,1,2], average = 'weighted')
accuracy = accuracy_score(true_lables, class_labels)

print("Confusion Matrix: \n", cm)
print("F1 Score:", f1)
print('Accuracy:', accuracy)


flag = np.array(class_labels == true_lables)
missed_values = []
identified_values = []
for i, f in enumerate(flag):
  if f:
    identified_values.append((mnli['validation_mismatched'][i]['premise'], mnli['validation_mismatched'][i]['hypothesis'], mnli['validation_mismatched'][i]['label']))

for i, f in enumerate(flag):
  if not f:
    missed_values.append((mnli['validation_mismatched'][i]['premise'], mnli['validation_mismatched'][i]['hypothesis'], mnli['validation_mismatched'][i]['label']))


print(missed_values)