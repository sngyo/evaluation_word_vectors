import os

import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

from bert_utils import *


class Bert:
    def __init__(self, model_name):
        self.csv_fname = 'data/bert_model/bert_w2v.csv'
        self.model_name = model_name
        self.arch_name = os.path.basename(model_name)
        self.model = self._load_model()
        self.tokenizer = BertTokenizer.from_pretrained(self.arch_name)
        self.word_embeddings = self._wes_load_from_cache()

    def _load_model(self):
        # Model Loading
        print("Try to load BERT model from local checkpoint...")
        # mkdir
        
        os.makedirs(os.path.dirname(self.csv_fname), exist_ok=True)
        try:
            model = torch.load(self.model_name + ".pt")
            print("BERT model is loaded.")
        except:
            print("Cannot model locally, downloading...")
            model = BertModel.from_pretrained(self.arch_name)
            print("Download finised, then save it locally ./data/bert_model/")
            torch.save(model, self.model_name + '.pt')
            print("Model is locally saved !")
        # print(model)
        model.eval()
        return model

    def _wes_load_from_cache(self):
        # load Word EmbeddingS from csv file
        if os.path.isfile(self.csv_fname):
            df = pd.read_csv()
            data_dict = df.to_dict()
        else:
            data_dict = {
                "word": [],
                "bert_embedding": []
            }
        return data_dict

    def save_wes_as_csv(self):
        df = pd.DataFrame.from_dict(self.word_embeddings)
        if os.path.isfile(self.csv_fname):
            os.remove(self.csv_fname)  # remove old version
        df.to_csv(self.csv_fname, index=False)
        print("Word Embeddings are successfully saved!")

    def text2token(self,text):
        # convert a text to tokens which can be interpreted in BERT model
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * len(tokenized_text)
        tokens_ts = torch.tensor([indexed_tokens])
        segments_ts = torch.tensor([segments_ids])
        return tokens_ts, segments_ts

    def get_bert_w2v(self, word, sentence=False):
        try:
            index = self.word_embeddings["word"].index(word)
            return self.word_embeddings["bert_embedding"][index]
        except:
            # if "word" does not exist in self.word_embeddings
            tokens_ts, segments_ts = self.text2token(word)
            with torch.no_grad():
                encoded_layers, _ = self.model(tokens_ts, segments_ts)
            
            if not sentence:
                # OPTION 1
                # sum of three last layers
                # 0: [CLS], 1: [input word], 2:[SEP]
                token_embeddings = get_token_embeddings(encoded_layers)
                bert_w2v = get_sum_word_vecs(token_embeddings)[1]
            else:
                # OPTION 2
                # regard a sentece vec as a word embedding
                bert_w2v = get_sentence_vec(encoded_layers)
            
            # add "word" & corresponding word_embedding in self.word_embeddings
            self.word_embeddings["word"].append(word)
            self.word_embeddings["bert_embedding"].append(bert_w2v)
            return bert_w2v
