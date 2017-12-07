import math
import random

def make_mr(objective_filepath, subjective_filepath):
    corpus = []
    with open(objective_filepath) as f:
        objective_lines = [line for line in f.read().splitlines()]
    with open(subjective_filepath) as f:
        subjective_lines = [line for line in f.read().splitlines()]

    for line in subjective_lines:
        output = "SUBJECTIVE:0 %s" % (line)
        corpus.append(output)

    for line in objective_lines:
        output = "OBJECTIVE:1 %s" % (line)
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

train, test, dev = make_mr("subjectivity/obj.txt", "subjectivity/subj.txt")
write_datasets(train, test, dev, "SUBJ_train.txt", "SUBJ_test.txt", "SUBJ_dev.txt")
