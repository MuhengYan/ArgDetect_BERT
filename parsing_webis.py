from nltk.tokenize import word_tokenize
import torch

from glob import glob
import csv

from tqdm import tqdm
import pickle



loc = 'data/corpus-webis-editorials-16/annotated-txt/split-for-evaluation-final'
train = f'{loc}/training'
dev = f'{loc}/validation'
test = f'{loc}/test'

#read as list
sets = {'train': train, 'dev': dev, 'test': test}
for name in sets:
    train_list = glob(f'{sets[name]}/*')
    sents = []
    tags = []


    for f in tqdm(train_list):
        reader = csv.reader(open(f, 'r'), delimiter='\t')
        num = []
        label = []
        text = []
        for row in reader:
            num.append(row[0])
            if len(row) == 3:
                label.append(row[1])
            else:
                label.append('no-unit')
            text.append(row[2])


        curr = []
        c_tag = []
        stopped = True
        #parse as sentence
        for l, t in zip(label, text):
            if l == 'title':
                stopped = True
                continue
            elif l == 'par-sep':
                if len(curr) > 0:
                    assert len(curr) == len(c_tag)
                    sents.append(curr)
                    tags.append(c_tag)
                curr = []
                c_tag = []
                stopped = True
            else:
                tks = word_tokenize(t)
                if stopped:
                    if l != 'no-unit':
                        c_tag += ['B']
                        c_tag += ['I'] * (len(tks) - 1)
                        stopped = False
                    else:
                        c_tag += ['O'] * len(tks)
                        stopped = True
                else:
                    if l != 'no-unit':
                        c_tag += ['I'] * (len(tks))
                        stopped = False
                    else:
                        c_tag += ['O'] * len(tks)
                        stopped = True

                curr += tks


    pickle.dump((sents, tags), open(f'data/parsed_webis_{name}.pkl', 'wb'))         