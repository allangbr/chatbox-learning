
# import numpy as np
# #Passo 2
# import nltk 
# from nltk.chat import Chat
# from nltk.chat.util import reflections

# #Passo 4
# nltk.download('punkt')
# nltk.download('wordnet')

# import json
# from nltk.stem import WordNetLemmatizer

# with open('intents.json') as file:
#     data = json.load(file)

# lemmatizer = WordNetLemmatizer()

# #Passo 5
# words = []
# classes = []
# documents = []
# ignore_letters = ['?', '!', '.', ',']

# for intent in data['intents']:
#     for pattern in intent['patterns']:
#         word_list = nltk.word_tokenize(pattern)
#         words.extend(word_list)
#         documents.append((word_list, intent['tag']))
#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])

# words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
# words = sorted(set(words))

# classes = sorted(set(classes))

# #Passo 6
# training = []
# output_empty = [0] * len(classes)

# for document in documents:
#     bag = []
#     word_patterns = document[0]
#     word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
#     for word in words:
#         bag.append(1) if word in word_patterns else bag.append(0)

#     output_row = list(output_empty)
#     output_row[classes.index(document[1])] = 1
#     training.append([bag, output_row])

# from tensorflow.keras.preprocessing import sequence

# training_sequences = [x[0] for x in training]
# max_len = max([len(x[0]) for x in training])
# padded_training_sequences = sequence.pad_sequences(training_sequences, maxlen=max_len, padding='post', truncating='post', value=0)
# training = np.array(padded_training_sequences)


# import random
# random.shuffle(training)
# training = np.array(training)

# train_x = [x.tolist() for x in training[:, 0]]
# train_y = [y.tolist() for y in training[:, 1]]

# #Passo 7
# import tensorflow as tf

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(128, input_shape=(len(words),), activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(len(classes), activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# hist = model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=5, verbose=1)

# #Passo 8
# model.save('chatbot_model.h5', hist)
# print('Model trained and saved')

import nltk
from nltk.stem import WordNetLemmatizer
import json
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

intents_file = open('intents.json').read()
intents = json.loads(intents_file)

words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique stemmed words", words)

# create our training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# shuffle our features and turn into np.array
np.random.shuffle(training)
training = np.array(training)

# create train and test lists
train_x = list(training[:, 0])
train_y = list(training[:, 1])
# train_y = keras.utils.to_categorical(train_y, num_classes)

model = Sequential()
model.add(keras.layers.Embedding(input_dim=len(words), output_dim=128, input_length=len(train_x[0])))
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
# compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("model created")
