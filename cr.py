import math
import random

def make_cr(corpora_filepaths):
    corpus = []
    for filepath in corpora_filepaths:
        with open(filepath) as f:
            lines = [line for line in f.read().splitlines()]
            if lines[0].find("***") != -1:
                lines = lines[11:]
        for line in lines:
            # print(line)
            if line[:3] == "[t]":
                continue
            ratings_text, sentence = line.split("##", 1)
            rating = 0
            # print(ratings_text)
            # print(sentence)
            for rating_text in ratings_text.split(","):
                minus_index = rating_text.find("[-")
                if minus_index != -1:
                    next_char = rating_text[minus_index + 2]
                    if next_char == "]":
                        next_char = "1"
                    rating -= int(next_char)

                plus_index = rating_text.find("[+")
                if plus_index != -1:
                    next_char = rating_text[plus_index + 2]
                    if next_char == "]":
                        next_char = "1"
                    rating += int(next_char)
            # print(rating)
            if rating == 0:
                continue
            label = get_label(rating)
            # print(label)
            output = "%s:%s %s" % (label, rating, sentence)
            # print(output)
            # raw_input()
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

def get_label(rating):
    if rating < 0:
        return "NEGATIVE"
    if rating == 0:
        return "NEUTRAL"
    if rating > 0:
        return "POSITIVE"

def write_datasets(train, test, dev, train_filepath, test_filepath, dev_filepath):
    # write the datasets to disk
    file_from_array(train_filepath, train)
    file_from_array(test_filepath, test)
    file_from_array(dev_filepath, dev)

def file_from_array(filepath, a):
    with open(filepath, 'w') as f:
        for line in a:
            f.write("%s\n" % (line))

files = []
files.append("customer review data/Apex AD2600 Progressive-scan DVD player.txt")
files.append("customer review data/Canon G3.txt")
files.append("customer review data/Creative Labs Nomad Jukebox Zen Xtra 40GB.txt")
files.append("customer review data/Nikon coolpix 4300.txt")
files.append("customer review data/Nokia 6610.txt")
files.append("customer review data/pt 2/Canon PowerShot SD500.txt")
files.append("customer review data/pt 2/Canon S100.txt")
files.append("customer review data/pt 2/Diaper Champ.txt")
files.append("customer review data/pt 2/Hitachi router.txt")
files.append("customer review data/pt 2/ipod.txt")
files.append("customer review data/pt 2/Linksys Router.txt")
files.append("customer review data/pt 2/MicroMP3.txt")
files.append("customer review data/pt 2/Nokia 6600.txt")
files.append("customer review data/pt 2/norton.txt")
files.append("customer review data/pt 3/Router.txt")
files.append("customer review data/pt 3/Computer.txt")
files.append("customer review data/pt 3/Speaker.txt")
train, test, dev = make_cr(files)
write_datasets(train, test, dev, "CR_train.txt", "CR_test.txt", "CR_dev.txt")
