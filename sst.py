def massage_sst_1(corpus_filepath, phrases_filepath, sentiments_filepath, set_mappings_filepath):
	# corpus_filepath: datasetSentences.txt
	# phrases_filepath: dictionary.txt
	# sentiments_filepath: sentiment_labels.txt
	# set_mappings_filepath: datasetSplit.txt

    # output:
    #     <label>:<numeric> <sentence>

    # read the corpus
    with open(corpus_filepath) as f:
        corpus = [line.split("\t", 1) for line in f.read().splitlines()]
        # exclude the first line, which is just headings
        corpus = corpus[1:]

    corpus = file_to_array(corpus_filepath, "\t")
    phrases = file_to_dict(phrases_filepath, "|", skip_first_line=False)
    sentiments = file_to_dict(sentiments_filepath, "|")
    set_mappings = file_to_dict(set_mappings_filepath, ",")
    print(len(corpus))
    print(len(phrases))
    print(len(set_mappings))
    print(len(sentiments))

    sentence_id = 0
    for sentence in corpus:
        sentence_id += 1
        phrase_id = phrases[sentence]
        sentiment = float(sentiments[phrase_id])
        label = get_label(sentiment)
        print(sentence_id)
        # print(sentence)
        # print(sentiment)
        # print(label)
        # raw_input()

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
        # lines = [line.split(delimiter, 1) for line in f.read().splitlines()]
        # for line in lines:
        #     key = line[0]
        #     value = line[1]
        #     d[key] = value



massage_sst_1("stanfordSentimentTreebank/datasetSentences.txt", "stanfordSentimentTreebank/dictionary.txt", "stanfordSentimentTreebank/sentiment_labels.txt", "stanfordSentimentTreebank/datasetSplit.txt")
