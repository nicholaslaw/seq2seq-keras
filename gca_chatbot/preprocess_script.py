import pandas as pd
import joblib
from preprocess import clean_text, vocab_idx_build, prepare_qn_ans

BATCH_SAVE = 10
DATA_PATH = 'movie_dialogue.csv'
SAVE_VOCAB_PATH = 'vocab_dic.p'
SAVE_MAX_LEN_PATH = 'max_len.p'
SAVE_NUM_BATCHES = 'num_batches.p'
x_save = './qns/x_'
y_save = './ans/y_'
QUESTIONS_COL = 'questions'
ANSWERS_COL = 'answers'
LOWER_CASE_ONLY = True

# Import Data
df = pd.read_csv(DATA_PATH)

# Preprocess
preproc_q = clean_text(list(df[QUESTIONS_COL]), lower_case=LOWER_CASE_ONLY)
preproc_a = clean_text(list(df[ANSWERS_COL]), lower_case=LOWER_CASE_ONLY)

# Build Vocab-IDX Dictionary
vocab_idx = vocab_idx_build(preproc_q, preproc_a)

# Build Qn, ANS Features (Not one hot)
X, Y, max_len = prepare_qn_ans(preproc_q, preproc_a, vocab_idx)

# Save Batches of Features
chunks_X = [X[i:i+BATCH_SAVE] for i in range(0, len(X), BATCH_SAVE)]
chunks_Y = [Y[i:i+BATCH_SAVE] for i in range(0, len(Y), BATCH_SAVE)]
joblib.dump(len(chunks_X), open(SAVE_NUM_BATCHES, 'wb'))

counter = 0
for i, j in zip(chunks_X, chunks_Y):
    joblib.dump(i, open(x_save+str(counter)+'.p', 'wb'))
    joblib.dump(j, open(y_save+str(counter)+'.p', 'wb'))
    counter += 1
joblib.dump(vocab_idx, open(SAVE_VOCAB_PATH, 'wb'))
joblib.dump(max_len, open(SAVE_MAX_LEN_PATH, 'wb'))