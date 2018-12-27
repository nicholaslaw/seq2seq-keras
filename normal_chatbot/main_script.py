import pandas as pd
import numpy as np
import joblib
from preprocess import clean_text, vocab_idx_build, build_features, embed_matrix
from gensim import models

from keras.layers import Embedding
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping

EMBEDDING_DIM = 300
latent_dim = 100
batch_size = 32
epochs = 2

# Import Data
df = pd.read_csv('movie_dialogue.csv')

# Preprocess
preproc_q = clean_text(list(df['questions'])[:100])
preproc_a = clean_text(list(df['answers'])[:100])

# Build Vocab Idx Dictionary
vocab_idx = vocab_idx_build(preproc_q, preproc_a)

# Build Features
x1, x2, y = build_features(preproc_q, preproc_a, vocab_idx)

# Import Word2Vec Vectors
w = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
embed_dic = {word: vec for word, vec in zip(list(w.vocab), list(w.vectors))}

# Embeddings Matrix
init_mat = embed_matrix(vocab_idx, embed_dic)

# Model Architecture
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,))
x = Embedding(input_dim=len(vocab_idx),output_dim=EMBEDDING_DIM,
                            weights=[init_mat],
                            trainable=False, mask_zero=True)(encoder_inputs)
x, state_h, state_c = LSTM(latent_dim,
                           return_state=True, activation='softmax')(x)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
x = Embedding(input_dim=len(vocab_idx),output_dim=EMBEDDING_DIM,
                            weights=[init_mat],
                            trainable=False, mask_zero=True)(decoder_inputs)
x = LSTM(latent_dim, return_sequences=True)(x, initial_state=encoder_states)
decoder_outputs = Dense(len(vocab_idx), activation='softmax')(x)

# Define the model that will turn
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile & run training
model.compile(optimizer='sgd', loss='categorical_crossentropy')

print(model.summary())

filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.h5"
checkpointer = ModelCheckpoint(filepath='weights.h5', verbose=1, save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=1, mode='auto')
callbacks_list = [checkpointer, earlystop]

history = model.fit([np.array(x1), np.array(x2)], np.array(y),
          batch_size=batch_size,
          callbacks=callbacks_list,
          epochs=epochs,
          validation_split=0.2)
model.save('nmt.h5')
joblib.dump(history.history, open('model_hist.p', 'wb'))