import nltk, re
import pandas as pd
import numpy as np

def clean_text(sentences):
    '''
    Requires packages: re, nltk
    
    sentences: a list of sentences (strings)
    
    returns a list of lists where each list contains tokens in a sentence
    '''
    print('Preprocessing Begins...')
    result = []
    symbols = '~ ! @ # $ % ^ & * < > ? /'
    symbols = symbols.split()
    for text in sentences:
        text = str(text)
        try:
            text = re.sub(r"\'m", " am", text)
            text = re.sub(r"\'ll", " will", text)
            text = re.sub(r"\'ve", " have", text)
            text = re.sub(r"\'re", " are", text)
            text = re.sub(r"won\'t", "will not", text)
            text = re.sub(r"can\'t", "cannot", text)
            text = re.sub(r"n\'t", " not", text)
            text = re.sub(r"n\'", "ng", text)
            text = re.sub(r"\'bout", "about", text)
            text = re.sub(r"\'til", "until", text)
            text = re.sub(r'\.+', ".", text)
            text = re.sub(r'\!+', "!", text)
            text = re.sub(r'\?+', "?", text)
            text = re.sub(r'\-', ' ', text)
            text = re.sub("[^a-zA-Z]"," ",text)
        except:
            print(text)
        if len(text.split()) == 0:
            continue
        if text.split()[0] in symbols:
            text = text[1:]
        tokenized = nltk.word_tokenize(text)
        then = []
        for word in tokenized:
            if word.isupper():
                word = word.capitalize()
#             if not word.islower() and not word.isupper():
#                 print('not upper not lower')
#                 word = word.lower()
            then.append(word)
        result.append(then)
    print('A Sneak Peek at Preprocessed Items...\n')
    print(result[:3], '\n')
    print('Preprocessing Done...\n')
    return result

def vocab_idx_build(tokenized_qns, tokenized_ans):
    '''
    Requires packages: 
    
    tokenized_sentences: list of lists, where each list contain tokens of preprocessed sentences
    
    returns dictionary of tokens and their respective indices
    '''
    print('Building Vocab Idx Dictionary...')
    tokenized_sentences = []
    tokenized_sentences.extend(tokenized_qns)
    tokenized_sentences.extend(tokenized_ans)
    result = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
#     result = {'<UNK>': 0, '<START>': 1, '<END>': 2}
    almost = list(set([i for sent in tokenized_sentences for i in sent]))
    for idx, j in enumerate(almost):
        result[j] = idx + 3
    print('Found {} Distinct Words..'.format(len(result) - 4))
    print('Done...\n')
    return result

def pad_truncate(sent, max_len, vocab_idx_dic):
    '''
    Requires packages:
    
    sent: a list of tokens belonging to a preprocessed sentence
    max_len: max length of a sentence
    
    returns a padded or truncated list with <START> or <END> tokens
    '''
    if len(sent) >= max_len:
        result = sent[:max_len]
        return result
    elif len(sent) < max_len:
        sent.extend([vocab_idx_dic['<PAD>']]*(max_len - len(sent)))
        return sent
    raise Exception('Pad or Truncation Error')
    
def swap_token_idx(sent, vocab_idx_dic):
    '''
    Requires packages:
    
    sent: a list of tokens
    vocab_idx_dic: dictionary of tokens and their respective indices
    
    returns a list of tokens' indices
    '''
    result = [vocab_idx_dic[token] for token in sent]
    return result
def build_features(tokenized_qns, tokenized_ans, vocab_idx_dic):
    '''
    Requires packages: numpy
    
    tokenized_sentences: list of lists, where each list contain tokens of preprocessed sentences
    vocab_idx_dic: dictionary of tokens and their respective indices
    
    returns X1, X2 and Y
    '''
    print('Building Features...')
    max_qn_len = [len(i) for i in tokenized_qns]
    max_ans_len = [len(i) for i in tokenized_ans]
    # max_sent_len = int(np.mean(max_qn_len + max_ans_len) + (1.5 * np.std(max_qn_len + max_ans_len)))
    max_sent_len = max(max(max_ans_len), max(max_qn_len))
    print('Sentence Length: ', max_sent_len)
    print('Vocab Size: ', len(vocab_idx_dic))
    X1 = []
    X2 = []
    Y = []
    for qn, ans in zip(tokenized_qns, tokenized_ans):
        idx_qn = swap_token_idx(qn, vocab_idx_dic)
        idx_ans = swap_token_idx(ans, vocab_idx_dic)
        
        temp_x1 = idx_qn[:max_sent_len]
        temp_x2 = [vocab_idx_dic['<EOS>']] + idx_ans
        
        temp_y = idx_ans[:(max_sent_len-1)] + [vocab_idx_dic['<EOS>']]
        
        preproc_x1 = pad_truncate(temp_x1, max_sent_len, vocab_idx_dic)
        preproc_x2 = pad_truncate(temp_x2, max_sent_len, vocab_idx_dic)
        preproc_y = pad_truncate(temp_y, max_sent_len, vocab_idx_dic)
#         assert len(preproc_x1) == max_sent_len
#         assert len(preproc_x2) == max_sent_len
#         assert len(preproc_y) == max_sent_len
        x1_final = []
        x2_final = []
        y_final = []
        for i, xx1, xx2 in zip(preproc_y, preproc_x1, preproc_x2):
#             xx1 = np.zeros(len(vocab_idx_dic))
#             xx1[i] = 1
#             xx2 = np.zeros(len(vocab_idx_dic))
#             xx2[j] = 1
            yy = np.zeros(len(vocab_idx_dic))
            yy[i] = 1
            x1_final.append(xx1)
            x2_final.append(xx2)
            y_final.append(np.array(yy))
        X1.append(np.array(x1_final))
        X2.append(np.array(x2_final))
        Y.append(np.array(y_final))
    print('Done...\n')
    return np.array(X1), np.array(X2), np.array(Y)

def embed_matrix(vocab_idx, word_embed_dic):
    mat = np.zeros((len(vocab_idx), 300))
    for word, idx in vocab_idx.items():
        mat[idx, :] = word_embed_dic.get(word, np.zeros(300))
    return mat