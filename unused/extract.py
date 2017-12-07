import gensim as gs
import datetime as dt
import numpy as np

w2v_filepath = 'GoogleNews-vectors-negative300.bin.gz'
TR_filepath = 'TREC_training.txt'
TE_filepath = 'TREC_test.txt'

labels_text = ["DESC","ENTY","ABBR","HUM","NUM","LOC"]
UNK_SYMBOL = '*UNK*'

TREC_vocab_filepath = "TREC_embeddings_pretrained.npy"
TREC_train_sentences_filepath = "TREC_train_sentences.npy"
TREC_train_labels_filepath = "TREC_train_labels.npy"
TREC_test_sentences_filepath = "TREC_test_sentences.npy"
TREC_test_labels_filepath = "TREC_test_labels.npy"

def preprocess_TREC_train(do_unk=True):
    labels = []
    sublabels = []
    sentences = []
    labels_in_nums = []
    sublabels_in_nums = []
    sentences_in_nums = []
    num_to_word = []
    word_to_num = {}
    word_count = 0
    word_occurrences = {}
    with open(TR_filepath) as f:
        for line in f.readlines():
            split_labels = line.split(':', 1)
            label = split_labels[0]
            labels.append(label)
            labels_in_nums.append(labels_text.index(label))

            split_sublabels = split_labels[1].split(' ', 1)
            sublabel = split_sublabels[0]
            sublabels.append(sublabels) # not used anywhere yet

            sentence = split_sublabels[1].split()
            sentences.append(sentence)

            # just count the word occurences
            for word in sentence:
                word_count += 1
                if word not in word_occurrences:
                    word_occurrences[word] = 0
                word_occurrences[word] += 1

    # now can unk the 1 word occurrences and add to dict
    # possibly extract this to another function
    for sentence in sentences:
        sentence_in_nums = []
        for word in sentence:
            if do_unk and word_occurrences[word] == 1:
                # print(word)
                sentence[sentence.index(word)] = UNK_SYMBOL
                word = UNK_SYMBOL
            if word not in word_to_num:
                word_to_num[word] = len(num_to_word)
                num_to_word.append(word)
            sentence_in_nums.append(word_to_num[word])
        # print(sentence_in_nums)
        # raw_input()
        sentences_in_nums.append(sentences_in_nums)

    print(len(sentences_in_nums))
    print(word_count)
    print(len(num_to_word))
    result = {}
    result["labels"] = labels
    result["sublabels"] = sublabels
    result["sentences"] = sentences
    result["labels_in_nums"] = labels_in_nums
    result["sublabels_in_nums"] = sublabels_in_nums
    result["sentences_in_nums"] = sentences_in_nums
    result["num_to_word"] = num_to_word
    result["word_to_num"] = word_to_num
    result["word_count"] = word_count
    result["word_occurrences"] = word_occurrences
    return result

def preprocess_TREC_test(train_vocab, do_unk=True):
    labels = []
    sublabels = []
    sentences = []
    labels_in_nums = []
    sublabels_in_nums = []
    sentences_in_nums = []
    num_to_word = []
    word_to_num = {}
    word_count = 0
    word_occurrences = {}
    words_in_train = 0
    with open(TE_filepath) as f:
        for line in f.readlines():
            split_labels = line.split(':', 1)
            label = split_labels[0]
            labels.append(label)
            labels_in_nums.append(labels_text.index(label))

            split_sublabels = split_labels[1].split(' ', 1)
            sublabel = split_sublabels[0]
            sublabels.append(sublabels) # not used anywhere yet

            sentence = split_sublabels[1].split()
            # unk the words not found in training set
            for i in range(len(sentence)):
                word_count += 1
                word = sentence[i]
                if do_unk and word not in train_vocab:
                    sentence[i] = UNK_SYMBOL
                else:
                    words_in_train += 1
            sentences.append(sentence)

    # make sentences in nums
    for sentence in sentences:
        sentence_in_nums = []
        for word in sentence:
            if word not in train_vocab:
                sentence_in_nums.append(0)
            else:
                sentence_in_nums.append(train_vocab[word])
        sentences_in_nums.append(sentence_in_nums)

    result = {}
    result["labels"] = labels
    result["sublabels"] = sublabels
    result["sentences"] = sentences
    result["labels_in_nums"] = labels_in_nums
    result["sublabels_in_nums"] = sublabels_in_nums
    result["sentences_in_nums"] = sentences_in_nums
    result["num_to_word"] = num_to_word
    result["word_to_num"] = word_to_num
    result["word_count"] = word_count
    result["word_occurrences"] = word_occurrences
    result["words_in_train"] = words_in_train
    return result

def ev(model, vocab_list):
    vectors = []
    in_model = 0
    for word in vocab_list:
        vector = np.random.normal(0, 0.01, 300)
        if word in model:
            in_model += 1
            vector = model[word]
        vectors.append(vector)
    # vectors = np.array(vectors)
    np.save('vocab_store.txt', vectors)
    print(in_model)

def timenow():
    return dt.datetime.now().strftime("%H:%M:%S")
def timeat(t):
    return t.strftime("%H:%M:%S")

start_time = dt.datetime.now()
print('Start time: %s' % timeat(start_time))
print('Preprocessing train')
print('Time now: %s' % timenow())
result = preprocess_TREC_train(do_unk=True)
print('Saving train sentences')
print('Time now: %s' % timenow())
# np.save(TREC_train_sentences_filepath, result["sentences_in_nums"])
print('Saving train labels')
print('Time now: %s' % timenow())
np.save(TREC_train_labels_filepath, result["labels_in_nums"])
print('Preprocessing test')
print('Time now: %s' % timenow())
result_test = preprocess_TREC_test(result["word_to_num"], do_unk=True)
np.save(TREC_test_sentences_filepath, result_test["sentences_in_nums"])
np.save(TREC_test_labels_filepath, result_test["labels_in_nums"])
print 'loading model'
# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
model = gs.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)
print 'loaded model'
print('Time now: %s' % timenow())
ev(model, result["num_to_word"])
print('Time now: %s' % timenow())
loaded = np.load('vocab_store.txt.npy')
print(loaded.shape)
print('Time now: %s' % timenow())

'''
word vectors - load in as embed
training sentences in nums
training sentence labels
testing sentences in nums
testing sentence labels
'''
TREC_train_sentences_filepath = "TREC_train_sentences.npy"
TREC_train_labels_filepath = "TREC_train_labels.npy"
TREC_test_sentences_filepath = "TREC_test_sentences.npy"
TREC_test_labels_filepath = "TREC_test_labels.npy"
