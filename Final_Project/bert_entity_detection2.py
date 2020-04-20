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

## general prerequisites
np.set_printoptions( threshold=sys.maxsize)

args = get_args()
print (args)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True

device = "cpu"
if args.cuda and torch.cuda.is_available():
    #args.gpu is set to 0 (first device) by default
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
    device = "cuda"

## BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# field names in the file train.txt, valid.txt, and test.txt
field_names = [ "id", "subject", "entity", "relation", "object", "question", "entity_label" ]

# use PANDAs to load TRAIN and VALIDATION files
pd_train = pd.read_csv( os.path.join(args.data_dir,"train.txt"), delimiter="\t", header=None,
                        names= field_names, usecols=["question", "entity_label"])
train = pd_train.values.tolist()

pd_dev = pd.read_csv( os.path.join(args.data_dir,"valid.txt"), delimiter="\t", header=None,
                        names= field_names, usecols=["question", "entity_label"])
dev = pd_dev.values.tolist();

train_dataloader = data.DataLoader( train, sampler = data.RandomSampler(train),   batch_size=args.batch_size)
dev_dataloader   = data.DataLoader( dev,   sampler = data.SequentialSampler(dev), batch_size=args.batch_size)

# directory to save the check-point model
save_path = os.path.join(args.save_path, args.entity_detection_path.lower())
os.makedirs(save_path, exist_ok=True)

# initialize the model and related variables for training
# n_classes = 3 : 0 for padding, 1 for O and 2 for I
model = EntityDetectionModel(config = 2, n_classes = 3, dropout= args.dropout)
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
print( "Model Config: ", model.config)
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
        batch_questions = batch[0]
        batch_labels = batch[1]
        max_question_length = utils.max_len(batch_questions)
        batch_train_ids, batch_train_masks =  \
            utils.get_tokenized_sentences(tokenizer, batch_questions)
        # batch_train_labels have to have the same length with the input_ids in order to calculate loss
        batch_train_labels = utils.get_ed_labels(batch_labels, batch_train_ids.size()[1])

        batch_train_ids    = batch_train_ids.to(device)
        batch_train_masks  = batch_train_masks.to(device)
        batch_train_labels = batch_train_labels.to(device)
        model.zero_grad()
        scores = model(batch_train_ids, batch_train_masks)
        ## scores: (batch, length,n_classes)
        ## batch_level: ( batch, length)
        loss = loss_fn(scores.permute(0,2,1), batch_train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()
        n_total += len(batch_train_ids)
        n_correct += utils.calculate_correct_scores(scores, batch_train_labels)
        ##print( "total {}, correct {}".format(n_total, n_correct))

        if iterations % args.log_every == 0:
            # print progress message
            print(log_template.format(time.time() - start,  epoch, iterations,
                                      1 + step, len(train_dataloader), # progress
                                      100. * (1 + step) / len(train_dataloader), # % done per epoch
                                      loss.item(),  #loss
                                      ' ' * 8,      # dev/loss
                                      100. * n_correct / n_total, # accuracy
                                      ' ' * 12))    # dev/accuracy

        #evaluate performance every args.dev_every
        if iterations % args.dev_every == 0:
            model.eval()
            n_dev_correct = 0
            b_dev_correct_rel = 0
            gold_list = []
            predict_list = []
            for dev_step,dev_batch in enumerate(dev_dataloader):
                batch_questions = dev_batch[0]
                batch_labels = dev_batch[1]
                #max_question_length = utils.max_len(batch_questions)
                batch_dev_ids, batch_dev_masks = \
                    utils.get_tokenized_sentences(tokenizer, batch_questions)
                batch_dev_labels = utils.get_ed_labels(batch_labels, batch_dev_ids.size()[1])
                batch_dev_ids    = batch_dev_ids.to(device)
                batch_dev_masks  = batch_dev_masks.to(device)
                batch_dev_labels = batch_dev_labels.to(device)
                dev_scores = model(batch_dev_ids, batch_dev_masks)
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



# the end, print the best performance score
print("{} Precision: {:10.6f}% Recall: {:10.6f}% F1 Score: {:10.6f}%".format(
             "Best Dev", 100. * best_dev_precision, 100. * best_dev_recall, 100. * best_dev_f1))

            
# '  Time    Epoch     Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
#('{:>6.0f},{:>5.0f},  {:>9.0f}, {:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{},{}'.split(','))
