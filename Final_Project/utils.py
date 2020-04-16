import torch
import numpy as np

def max_len( *args ):
    """
    find the biggest number of words in a list of strings
    :param args: array-like of strings
    :return: integer
    """
    max_len = 0
    for data in args:
        max_data_length =  max(map(lambda x :len(x.split()), data))
        if max_len < max_data_length:
            max_len = max_data_length
    return max_len


def get_ed_labels( eds, max_sentence_length ):
    """
    convert a list of string to a 2D tensor where the 2nd dimension is of length max_sentence_length
    :param eds:
    :param max_sentence_length:
    :return:
    """
    ed_labels = []
    for ed in eds:
        ed = ed.split()
        ed_arr = np.asarray(ed)
        ed_arr =  (ed_arr == 'I') + 1
        if max_sentence_length > len(ed):
            extra_length = max_sentence_length - len(ed)
            ed_arr = np.pad(ed_arr, [(0, extra_length)])
        else:
            ed_arr = ed_arr[:max_sentence_length]
        ed_labels.append(ed_arr)

    return torch.tensor(ed_labels)

def get_tokenized_sentences( tokenizer, sentences ):
    """
    :param tokenizer: a Bert tokenizer
    :param sentences: a list of sentences that need to be tokenized and padded to a length max_question_length
    :param max_question_length:
    :return:
    """
    input_ids = []
    attention_masks = []
    encoded_dict = tokenizer.batch_encode_plus(
            sentences,                       # Sentence to encode.
            add_special_tokens=True,         # Add '[CLS]' and '[SEP]'
            pad_to_max_length=True,
            return_attention_mask=True,      # Construct attn. masks.
            return_tensors='pt',            # Return pytorch tensors.
            )
    input_ids =encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids,attention_masks

def calculate_correct_scores( scores, labels):
    """
    :param scores: output from the model(X) (batch, length, n_classes)
    :param labels: ( batch, length )
    :return: number of correct outputs in compare to the labels
    """
    ## the original calculation is below
    ## n_correct += torch.sum((torch.sum((torch.max(scores, 1)[1].view(batch.ed.size()).data == batch.ed.data), dim=0) \
    ##                        == batch.ed.size()[0])).item()

    score_index_array = torch.max(scores, dim=2)[1]   #(batch, length)
    batch_of_correct_elements_on_each_sentence = torch.sum(score_index_array == labels,dim=1) #(batch)
    ## each output is considered correct if all of its elements are one
    n_correct = torch.sum( batch_of_correct_elements_on_each_sentence == labels.size()[1]).item()
    return n_correct
