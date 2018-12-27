import joblib
from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.optimizers import Adam, SGD
from keras.models import Model
from gensim import models
from preprocess import embed_matrix, train_model

LOAD_VOCAB_PATH = 'vocab_dic.p'
LOAD_MAX_LEN_PATH = 'max_len.p'
LOAD_NUM_BATCHES = 'num_batches.p'
x_load = './qns/x_'
y_load = './ans/y_'
WORD2VEC = 'GoogleNews-vectors-negative300.bin.gz'
SAVE_MODEL_PATH ='model.h5'

BATCH_SAVE = 10
EPOCHS = 10
LSTM_UNITS = 100
EMBEDDING_DIM = 300

# Load Previously Saved Objects
vocab_idx = joblib.load(open(LOAD_VOCAB_PATH, 'rb'))
max_len = joblib.load(open(LOAD_MAX_LEN_PATH, 'rb'))
num_batches = joblib.load(open(LOAD_NUM_BATCHES, 'rb'))

# Load Word2Vec
w = models.KeyedVectors.load_word2vec_format(WORD2VEC, binary=True)
embed_dic = {word: vec for word, vec in zip(list(w.vocab), list(w.vectors))}
init_mat = embed_matrix(vocab_idx, embed_dic)

# Model Architecture
ad = Adam(lr=0.00005)
input_context = Input(shape=(max_len,), name='input_context')
input_answer = Input(shape=(max_len,), name='input_answer')
LSTM_encoder = LSTM(LSTM_UNITS, kernel_initializer='lecun_uniform')
LSTM_decoder = LSTM(LSTM_UNITS, kernel_initializer='lecun_uniform')
Shared_Embedding = Embedding(output_dim=EMBEDDING_DIM, input_dim=len(vocab_idx), weights=[init_mat], input_length=max_len)
word_embedding_context = Shared_Embedding(input_context)
context_embedding = LSTM_encoder(word_embedding_context)

word_embedding_answer = Shared_Embedding(input_answer)
answer_embedding = LSTM_decoder(word_embedding_answer)

merge_layer = concatenate([context_embedding, answer_embedding])
out = Dense(int(len(vocab_idx)/2), activation="relu", name='relu_dense')(merge_layer)
out = Dense(len(vocab_idx), activation="softmax", name='soft_dense')(out)

model = Model([input_context, input_answer], out)

model.compile(loss='categorical_crossentropy', optimizer=ad)

print(model.summary())

# Train Model
for epoch in range(EPOCHS):
    for batch in range(num_batches):
        X = joblib.load(open(x_load + str(batch) + '.p', 'rb'))
        Y = joblib.load(open(y_load + str(batch) + '.p', 'rb'))
        train_model(model, X, Y, vocab_idx, max_len)
        model.save(SAVE_MODEL_PATH)