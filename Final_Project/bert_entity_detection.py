## IMPORT

import torch
import torch.nn as nn
import torch.utils.data as data
import os
import numpy as np
import pandas as pd
import random
import sys
from args import get_args
import utils
from transformers import BertTokenizer, AdamW, BertConfig,  get_linear_schedule_with_warmup
from entity_detection_model import EntityDetectionModel
from entity_detection_evaluation import evaluation
import time

## general prerequisites: args, cuda ...
np.set_printoptions( threshold=sys.maxsize)
args = get_args()
print (args)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministoc = True

device = "cpu"
if args.cuda and torch.cuda.is_available():
    #args.gpu is set to 0 (first device) by default
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
    device = "cuda"

## BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

## DATA
# field names in the file train.txt, valid.txt, and test.txt
field_names = [ "id", "subject", "entity", "relation", "object", "question", "entity_label" ]

# use PANDAs to load TRAIN and VALIDATION files
pd_train = pd.read_csv( os.path.join(args.data_dir,"train.txt"), delimiter="\t", header=None,
                        names= field_names, usecols=["question", "entity_label"])
pd_dev = pd.read_csv( os.path.join(args.data_dir,"valid.txt"), delimiter="\t", header=None,
                        names= field_names, usecols=["question", "entity_label"])


# tokenized IDs, attention masks, and annotated lables for TRAIN questions
train_input_ids, train_attention_masks = utils.get_tokenized_sentences(tokenizer, pd_train["question"])
max_train_question_length = train_input_ids.size()[1]
train_labels = utils.get_ed_labels(pd_train["entity_label"], max_train_question_length)

# tokenized IDs, attention masks, and annotated lables for VALIDATION questions
dev_input_ids, dev_attention_masks = utils.get_tokenized_sentences(tokenizer, pd_dev["question"])
max_dev_question_length = dev_input_ids.size()[1]
dev_labels = utils.get_ed_labels(pd_dev["entity_label"], max_dev_question_length)

# TRAIN data wrapper ( input_ids, masks, and labels )
train_dataset = data.TensorDataset( train_input_ids, train_attention_masks, train_labels)
train_dataloader = data.DataLoader( train_dataset, sampler = data.RandomSampler(train_dataset), batch_size=args.batch_size)

# VALIDATION data wrapper ( input_ids, masks, and labels )
dev_dataset = data.TensorDataset( dev_input_ids, dev_attention_masks, dev_labels)
dev_dataloader = data.DataLoader( dev_dataset, sampler = data.SequentialSampler(dev_dataset), batch_size=args.batch_size)


# directory to save the check-point model
save_path = os.path.join(args.save_path, args.entity_detection_path.lower())
os.makedirs(save_path, exist_ok=True)

# initialize the model and related variables for training
model = EntityDetectionModel(n_classes = 3, dropout= args.dropout)
model.to(device)
optimizer = AdamW(model.parameters(), lr = args.lr )
total_steps = len(train_dataloader)*args.epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= 0, num_training_steps= total_steps )
loss_fn = nn.NLLLoss()
total_train_loss = 0
early_stop = False
best_dev_f1, best_dev_precision, best_dev_recall = 0, 0, 0
iterations = 0
iters_not_improved = 0 # record the number of iterations that the model is not improved
num_dev_in_epoch = (len(train_dataloader) // args.batch_size // args.dev_every) + 1
patience = args.patience * num_dev_in_epoch # for early stopping
log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{},{}'.split(','))
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
print(header)
index2tag = np.array(['X','O','I'])
# start training
start = time.time()

for epoch in range (0, args.epochs):
    if early_stop:
        print("Early Stopping. Epoch: {}, Best Dev F1: {}".format(epoch, best_dev_f1))
        break

    n_total = 0
    n_correct = 0
    n_correct_ed, n_correct_ner, n_correct_rel = 0,0,0
    model.train()
    for step,batch in enumerate(train_dataloader):
        iterations += 1
        batch_input_ids = batch[0].to(device)
        batch_input_masks = batch[1].to(device)
        batch_labels = batch[2].to(device)
        model.zero_grad()
        scores = model(batch_input_ids, batch_input_masks)
        ## scores: (batch, length,n_classes)
        ## batch_level: ( batch, length)
        loss = loss_fn(scores.permute(0,2,1), batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_gradient)
        optimizer.step()
        n_total += len(batch_input_ids)
        n_correct += utils.calculate_correct_scores(scores, batch_labels)

        #evaluate performance every args.dev_every
        if iterations % args.dev_every == 0:
            model.eval()
            n_dev_correct = 0
            b_dev_correct_rel = 0
            gold_list = []
            predict_list = []
            for dev_step,dev_batch in enumerate(dev_dataloader):
                batch_dev_input_ids = dev_batch[0].to(device)
                batch_dev_input_masks = dev_batch[1].to(device)
                batch_dev_labels = dev_batch[2].to(device)
                dev_scores = model(batch_dev_input_ids, batch_dev_input_masks)
                n_dev_correct += utils.calculate_correct_scores(dev_scores,batch_dev_labels)
                index_tag = torch.max(dev_scores,dim=2)[1]
                gold_list.append(batch_dev_labels.cpu().data.numpy())
                predict_list.append(index_tag)

            precision, recall, f1 = evaluation( gold_list, predict_list, index2tag)
            print("{} Precision: {:10.6f}% Recall: {:10.6f}% F1 Score: {:10.6f}%".format(
                    "Dev", 100. * precision, 100. * recall, 100. * f1))

                #update model if we have a better f1
            if f1 > best_dev_f1:
                best_dev_precision = precision
                best_dev_recall = recall
                best_dev_f1 = f1
                iters_not_improved = 0 # reset the counter
                snapshot_path = os.path.join(args.save_path, args.specify_prefix + '_best_model.pt')
                # save model, delete previous 'best_snapshot' files
                torch.save(model, snapshot_path)
            else: # this model is not good enough
                iters_not_improved += 1
                if iters_not_improved >  patience:
                    early_stop = True
                    break


        if iterations % args.log_every == 1:
            # print progress message
            print(log_template.format(time.time() - start,  epoch, iterations,
                                      1 + step, len(train_dataloader), # progress
                                      100. * (1 + step) / len(train_dataloader), # % done per epoch
                                      loss.item(),  #loss
                                      ' ' * 8,      # dev/loss
                                      100. * n_correct / n_total, # accuracy
                                      ' ' * 12))    # dev/accuracy

# '  Time    Epoch     Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
#('{:>6.0f},{:>5.0f},  {:>9.0f}, {:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{},{}'.split(','))
