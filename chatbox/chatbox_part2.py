#Passo 9
import tensorflow as tf
model = tf.keras.models.load_model('chatbot_model.h5')

#Passo 10
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# def predict_class(sentence):
#     bow = bag_of_words(sentence)
#     res = model.predict(np.array([bow]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

#     results.sort(key=lambda x: x[1], reverse=True)
#     return_list = []
#     for r in results:
#         return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
#     return return_list

def predict_class(input_text):
    # preprocess the input text
    input_text = input_text.lower()
    tokens = word_tokenize(input_text)
    tokens = [stemmer.stem(token) for token in tokens]
    # convert the input text to a numeric sequence of tokens
    sequence = []
    for token in tokens:
        if token in word_index:
            sequence.append(word_index[token])
        else:
            sequence.append(0)
    # pad the sequence to a fixed length
    sequence = pad_sequences([sequence], maxlen=max_len, padding='post')
    # predict the class of the input text
    prediction = model.predict(sequence)[0]
    predicted_class_index = np.argmax(prediction)
    predicted_class = classes[predicted_class_index]
    return predicted_class


# def get_response(intents_list, intents_json):
#     tag = intents_list[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if i['tag'] == tag:
#             result = random.choice(i['responses'])
#             break
#     return result
def predict_class(input_text):
    # preprocess the input text
    input_text = input_text.lower()
    tokens = word_tokenize(input_text)
    tokens = [stemmer.stem(token) for token in tokens]
    # convert the input text to a numeric sequence of tokens
    sequence = []
    for token in tokens:
        if token in word_index:
            sequence.append(word_index[token])
        else:
            sequence.append(0)
    # pad the sequence to a fixed length
    sequence = pad_sequences([sequence], maxlen=max_len, padding='post')
    # predict the class of the input text
    prediction = model.predict(sequence)[0]
    predicted_class_index = np.argmax(prediction)
    predicted_class = classes[predicted_class_index]
    return predicted_class


print('Chatbot is ready to talk!')

# while True:
#     message = input('')
#     if message == 'quit':
#         break
#     intents = predict_class(message)
#     response = get_response(intents, intents_json)
#     print(response)
while True:
    input_text = input("You: ")
    if input_text.lower() == 'bye':
        break
    response_text = get_response(input_text)
    print("Bot:", response_text)