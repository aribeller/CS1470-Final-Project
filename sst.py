def massage_sst_1(corpus_filepath, phrases_filepath, sentiments_filepath, set_mappings_filepath):
	# corpus_filepath: datasetSentences.txt
	# phrases_filepath: dictionary.txt
	# sentiments_filepath: sentiment_labels.txt
	# set_mappings_filepath: datasetSplit.txt

    # output:
    #     <label>:<numeric> <sentence>

    # read the files
    corpus = file_to_array(corpus_filepath, "\t")
    phrases = file_to_dict(phrases_filepath, "|", skip_first_line=False)
    sentiments = file_to_dict(sentiments_filepath, "|")
    set_mappings = file_to_array(set_mappings_filepath, ",")
    print(len(corpus))
    print(len(phrases))
    print(len(set_mappings))
    print(len(sentiments))

    # get the labels for each sentence and organise them into t/t/d sets
    train = []
    test = []
    dev = []
    sentence_id = 0
    for sentence in corpus:
        phrase_id = phrases[sentence]
        sentiment = float(sentiments[phrase_id])
        label = get_label(sentiment)
        destination_set = int(set_mappings[sentence_id])
        dataset_sentence = "%s:%s %s" % (label, sentiment, sentence)
        if destination_set == 1:
            train.append(dataset_sentence)
        if destination_set == 2:
            test.append(dataset_sentence)
        if destination_set == 3:
            dev.append(dataset_sentence)
        sentence_id += 1
        # print(sentence_id)
    print(len(train))
    print(len(test))
    print(len(dev))
    return train, test, dev

def write_datasets(train, dev, test, train_filepath, test_filepath, dev_filepath):
    # write the datasets to disk
    file_from_array(train_filepath, train)
    file_from_array(test_filepath, test)
    file_from_array(dev_filepath, dev)

def get_label(sentiment):
    if 0 <= sentiment and sentiment <= 0.2:
        return "VERY_NEGATIVE"
    if 0.2 < sentiment and sentiment <= 0.4:
        return "NEGATIVE"
    if 0.4 <= sentiment and sentiment <= 0.6:
        return "NEUTRAL"
    if 0.6 < sentiment and sentiment <= 0.8:
        return "POSITIVE"
    if 0.8 <= sentiment and sentiment <= 1:
        return "VERY_POSITIVE"

def file_to_dict(filepath, delimiter, skip_first_line=True):
    d = {}
    with open(filepath) as f:
        for line in f.read().splitlines():
            if skip_first_line:
                skip_first_line = False
                continue
            line = line.split(delimiter, 1)
            d[line[0]] = line[1]
    return d

def file_to_array(filepath, delimiter, keep_position=1, skip_first_line=True):
    a = []
    with open(filepath) as f:
        for line in f.read().splitlines():
            if skip_first_line:
                skip_first_line = False
                continue
            line = line.split(delimiter, 1)
            a.append(line[keep_position])
    return a

def file_from_array(filepath, a):
    with open(filepath, 'w') as f:
        for line in a:
            f.write("%s\n" % (line))

train, test, dev = massage_sst_1("stanfordSentimentTreebank/datasetSentences.txt", "stanfordSentimentTreebank/dictionary.txt", "stanfordSentimentTreebank/sentiment_labels.txt", "stanfordSentimentTreebank/datasetSplit.txt")
write_datasets(train, test, dev, "sst_1_train.txt", "sst_1_test.txt", "sst_1_dev.txt")
