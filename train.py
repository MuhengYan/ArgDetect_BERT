from transformers import DistilBertTokenizerFast
from transformers import DistilBertForTokenClassification
from transformers import EarlyStoppingCallback

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

import torch

from glob import glob
import pandas as pd
import csv
import pickle
import numpy as np
import collections


train_texts, train_tags = pickle.load(open('data/parsed_webis_train.pkl', 'rb'))
val_texts, val_tags = pickle.load(open('data/parsed_webis_dev.pkl', 'rb'))
test_texts, test_tags = pickle.load(open('data/parsed_webis_test.pkl', 'rb'))

#drop the one long doc
train_texts = [txt for txt in train_texts if len(txt) <= 512]
train_tags = [t for t in train_tags if len(t) <= 512]

#report stats here
def report_stats(tags):
    print('==============')
    length = [len(t) for t in tags]
    print(f'#chunks: {len(tags)}')
    print(f'average length: {np.mean(length):.1f}')
    print(f'99% length: {np.percentile(length, [99])[0]:.1f}')
    print(f'max length: {np.percentile(length, [100])[0]:.1f}')
    counter = collections.defaultdict(int)
    
    for t in tags:
        for key in ['B', 'I', 'O']:
            counter[key] += t.count(key)
    for key in ['B', 'I', 'O']:
        print(f'{key}: {counter[key]}')

report_stats(train_tags)
report_stats(val_tags)
report_stats(test_tags)

#encode token tags
tags = train_tags + val_tags + test_tags
unique_tags = set(tag for doc in tags for tag in doc)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}


#tokenize the texts
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
test_encodings = tokenizer(test_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)

def encode_tags(tgs, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tgs]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)
        
        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        
        encoded_labels.append(doc_enc_labels.tolist())
        
        
    return encoded_labels

train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)
test_labels = encode_tags(test_tags, test_encodings)



#wrap in dataset class
class WebisDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_encodings.pop("offset_mapping") 
val_encodings.pop("offset_mapping")
test_encodings.pop("offset_mapping")

train_dataset = WebisDataset(train_encodings, train_labels)
val_dataset = WebisDataset(val_encodings, val_labels)
test_dataset = WebisDataset(test_encodings, test_labels)

#load pretrained model

model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=len(unique_tags))

#training settings

early_stopping = EarlyStoppingCallback(early_stopping_patience=10)
# def compute_metrics(pred):
#     labels = pred.label_ids.flatten()
#     preds = pred.predictions.argmax(-1).flatten()
#     z = zip(labels, preds)
#     z = [item for item in z if item[0] != -100]
#     labels = np.array([item[0] for item in z])
#     preds = np.array([item[1] for item in z])
    
    
    
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
#     acc = accuracy_score(labels, preds)
#     return {
#         'acc_macro': acc,
#         'f1_macro': f1,
#         'p_macro': precision,
#         'r_macro': recall
#     }

def compute_metrics(pred):
    labels = pred.label_ids.flatten()
    preds = pred.predictions.argmax(-1).flatten()
    z = zip(labels, preds)
    z = [item for item in z if item[0] != -100]
    labels = np.array([item[0] for item in z])
    preds = np.array([item[1] for item in z])
    
    l_0 = np.array([1 if item[0]==0 else 0 for item in z])
    p_0 = np.array([1 if item[1]==0 else 0 for item in z])
    
    l_1 = np.array([1 if item[0]==1 else 0 for item in z])
    p_1 = np.array([1 if item[1]==1 else 0 for item in z])
    
    l_2 = np.array([1 if item[0]==2 else 0 for item in z])
    p_2 = np.array([1 if item[1]==2 else 0 for item in z])
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    p_0, r_0, f1_0, _ = precision_recall_fscore_support(l_0, p_0, average='binary')
    p_1, r_1, f1_1, _ = precision_recall_fscore_support(l_1, p_1, average='binary')
    p_2, r_2, f1_2, _ = precision_recall_fscore_support(l_2, p_2, average='binary')
    
    acc = accuracy_score(labels, preds)
    return {
        'acc': acc,
        'f1_macro': f1,
        'p_macro': precision,
        'r_macro': recall,
        'f1_0': f1_0,
        'p_0': p_0,
        'r_0': r_0,
        'f1_1': f1_1,
        'p_1': p_1,
        'r_1': r_1,
        'f1_2': f1_2,
        'p_2': p_2,
        'r_2': r_2,
    }


training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=30,              
    per_device_train_batch_size=32,  
    per_device_eval_batch_size=64,   
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',            
    logging_steps=5,
    evaluation_strategy='steps',
    eval_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1_macro'
)


#train model
trainer = Trainer(
    model=model,                         
    args=training_args,                 
    train_dataset=train_dataset,        
    eval_dataset=val_dataset, 
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)

trainer.train()

#eval and test the best model
print(trainer.evaluate(val_dataset))

print(trainer.evaluate(test_dataset))
