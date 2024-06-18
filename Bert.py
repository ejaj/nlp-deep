from datasets import load_dataset, Split
from transformers import BertTokenizer

from preprocess import sentence_tokenize

tokenizer = BertTokenizer('our_vocab/vocab.txt')
# print(tokenizer)
new_sentence = 'follow the white rabbit neo'
new_tokens = tokenizer.tokenize(new_sentence)
print(new_tokens)
new_ids = tokenizer.convert_tokens_to_ids(new_tokens)
print(new_ids)
new_ids = tokenizer.encode(new_sentence)
print(new_ids)
print(tokenizer.convert_ids_to_tokens(new_ids))
tokenized_output = tokenizer(new_sentence, add_special_tokens=False, return_tensors='pt')
print(tokenized_output)

sentence1 = 'follow the white rabbit neo'
sentence2 = 'no one can be told what the matrix is'
batch_of_pairs = tokenizer([sentence1, sentence2], padding=True, return_tensors='pt', max_length=50, truncation=True)
print(batch_of_pairs)
separate_sentences = tokenizer([sentence1, sentence2], padding=True)
print(separate_sentences)
print(tokenizer.convert_ids_to_tokens(separate_sentences['input_ids'][0]))
print(separate_sentences['attention_mask'][0])
first_sentences = [sentence1, 'another first sentence']
second_sentences = [sentence2, 'a second sentence here']
batch_of_pairs = tokenizer(first_sentences, second_sentences)
first_input = tokenizer.convert_ids_to_tokens(batch_of_pairs['input_ids'][0])
second_input = tokenizer.convert_ids_to_tokens(batch_of_pairs['input_ids'][1])
print(first_input)
print(second_input)

localfolder = "texts"
new_fnames = sentence_tokenize(localfolder)
print(new_fnames)

dataset = load_dataset(path='csv', data_files=new_fnames, quotechar='\\', split=Split.TRAIN)
# print(dataset.features, dataset.num_columns, dataset.shape)
#
# print(dataset[2])
#
# print(dataset['source'][:3])
print(dataset.unique('source'))


def is_alice_label(row):
    is_alice = int(row['source'] == 'alice28-1476.txt')
    return {'labels': is_alice}


dataset = dataset.map(is_alice_label)


tokenized_dataset = tokenizer(dataset['sentence'],
                              padding=True,
                              return_tensors='pt',
                              max_length=50,
                              truncation=True)
print(tokenized_dataset['input_ids'])

