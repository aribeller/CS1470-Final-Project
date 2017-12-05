import math
import random

def make_mr(positive_filepath, negative_filepath):
    corpus = []
    with open(positive_filepath) as f:
        positive_lines = [line for line in f.read().splitlines()]
    with open(negative_filepath) as f:
        negative_lines = [line for line in f.read().splitlines()]

    for line in negative_lines:
        output = "NEGATIVE:0 %s" % (line)
        corpus.append(output)

    for line in positive_lines:
        output = "POSITIVE:1 %s" % (line)
        corpus.append(output)

    total = len(corpus)
    random.shuffle(corpus)
    total_train = int(math.ceil(total * 0.7))
    total_test = int(math.ceil(total * 0.2))
    total_dev = total - total_train - total_test
    print(total_train)
    print(total_test)
    print(total_dev)
    train = [corpus[i] for i in range(total_train)]
    test = [corpus[i + total_train] for i in range(total_test)]
    dev = [corpus[i + total_train + total_test] for i in range(total_dev)]
    print(len(train))
    print(len(test))
    print(len(dev))
    return train, test, dev

def write_datasets(train, test, dev, train_filepath, test_filepath, dev_filepath):
    # write the datasets to disk
    file_from_array(train_filepath, train)
    file_from_array(test_filepath, test)
    file_from_array(dev_filepath, dev)

def file_from_array(filepath, a):
    with open(filepath, 'w') as f:
        for line in a:
            f.write("%s\n" % (line))

train, test, dev = make_mr("movie/rt-polaritydata/rt-polarity.pos", "movie/rt-polaritydata/rt-polarity.neg")
write_datasets(train, test, dev, "MR_train.txt", "MR_test.txt", "MR_dev.txt")
