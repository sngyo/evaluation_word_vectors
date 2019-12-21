import argparse
import os

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MODEL_LS = ['bert-base-uncased']  # TODO loading data/bert_model/*


# make an argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", "-m", metavar='MODEL', choices=MODEL_LS, default=MODEL_LS[0],
    help='loading local model among: ' + ' | '.join(MODEL_LS)\
    + ' (default: ' + MODEL_LS[0]
)

args = parser.parse_args()
MODEL_DIR_PATH = 'data/bert_model/'
# mkdir
os.makedirs(MODEL_DIR_PATH, exist_ok=True)

MODEL_NAME = MODEL_DIR_PATH + args.model

print("Try to load BERT model from local checkpoint...")
try:
    model = torch.load(MODEL_NAME + ".pt")
    print("BERT model is loaded.")
except:
    print("Cannot model locally, downloading... (it may takes few minutes)")
    model = BertModel.from_pretrained('bert-base-uncased')
    print("Download finised, then save it locally ./data/bert_model/")
    torch.save(model, MODEL_NAME + '.pt')
    print("Model is locally saved !")

# print(model)
model.eval()



def text2token(text):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(tokenized_text)
    tokens_ts = torch.tensor([indexed_tokens])
    segments_ts = torch.tensor([segments_ids])
    return tokens_ts, segments_ts


def get_token_embeddings(encoded_layers):
    # convert [layers, batches, tokens, features] to [tokens, layers, features]
    token_embeddings = torch.stack(encoded_layers, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1, 0, 2)
    return token_embeddings


def get_sum_word_vecs(token_embeddings):
    token_vecs_sum = []
    for token in token_embeddings:
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum.append(sum_vec)
    return token_vecs_sum


def get_sentence_vec(encoded_layers):
    # this sentence vec is just the average of all embedded tokens
    # TODO remove [CLS], [SEP] or '.' etc.
    token_vecs = encoded_layers[-1][0]  # last_layer, batch_id:0
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding


def main():
    text = input("text: ")
    tokens_ts, segments_ts = text2token(text)

    with torch.no_grad():
        encoded_layers, _ = model(tokens_ts, segments_ts)

    print("Number of layers: {}".format(len(encoded_layers)))
    # number of sentences == number of batches
    print("Number of batches: {}".format(len(encoded_layers[0])))  
    print("Number of tokens: {}".format(len(encoded_layers[0][0])))
    print("Number of hidden units: {}".format(len(encoded_layers[0][0][0])))

    token_embeddings = get_token_embeddings(encoded_layers)
    sum_word_vecs = get_sum_word_vecs(token_embeddings)

    sentence_vec = get_sentence_vec(encoded_layers)

    
if __name__ == "__main__":
    main()
    
