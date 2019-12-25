import os
import sys
import pickle
import argparse

import gensim
import gensim.downloader as api
import numpy as np

# load available model names
MODEL_NAMES = []
with open("data/word2vec/model_names.txt") as f:
    for line in f:
        MODEL_NAMES.append(line.replace('\n', ''))


class Word2Vec():
    def __init__(self, modelname=MODEL_NAMES[0]):
        self.modelname = modelname
        self.model = self._load_word2vec()

    def __call__(self, word_label):
        return self._get_vec(word_label)
        
    def _load_word2vec(self):
        os.makedirs('data/word2vec', exist_ok=True)
        
        # load cache file if exists
        cache_filename = 'data/word2vec/' + self.modelname + '.pickle'
        if os.path.exists(cache_filename):
            with open(cache_filename, 'rb') as f:
                print('loading from cache')
                return pickle.load(f)

        print('Downloading word2vec model file')
        model = api.load(self.modelname)
            
        print('Creating cache of word2vec model')
        with open(cache_filename, 'wb') as f:
            pickle.dump(model, f)
        return model

    def _get_vec(self, word_label):
        if word_label in self.model:
            return self.model[word_label]
        # TODO is it okay [return None]?
        else:
            return None

    def most_similar(self, word_label):
        # get a list of 10 most similar words of word_label
        # similarity is based on cos_similarity
        return self.model.most_similar(word_label)

    def algebraic_operation(self, pos=[], neg=[]):
        # return pos[0] + pos[1] + ... - neg[0] - neg[1]...
        return self.model.most_similar(positive=pos, negative=neg)
    
    def get_cosine_similarity(self, word_label_1, word_label_2):
        vec1 = self._get_vec(word_label_1)
        vec2 = self._get_vec(word_label_2)
        if vec1 is None or vec2 is None:
            print(word_label_1, word_label_2)
            return None
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        cos = dot / (norm1 * norm2)
        return cos


# for debug
if __name__ == "__main__":

    # argument [--model]    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', '-m', metavar='model_name',
        default=MODEL_NAMES[0], choices=MODEL_NAMES,
        help='model name: ' + ' | '.join(MODEL_NAMES)\
        + ' (default:' + MODEL_NAMES[0]+ ')'
    )
    args = parser.parse_args()
    
    vec_loader = Word2Vec(modelname=args.model)
    word_label = input('test word label: ')
    
    # show word vector itself
    # print(type(vec_loader(word_label)))

    # show 10 most similar words
    sim_lis = vec_loader.most_similar(word_label)
    for word in sim_lis:
        print(word)

    # show 10 most similar words after algebraic operation
    pos = ['man', 'queen']
    neg = ['woman']
    sim_lis = vec_loader.algebraic_operation(pos, neg)
    print('\n Exapmple of algebraci_operation')
    print('(+) ' + str(pos))
    print('(-) ' + str(neg))
    for word in sim_lis:
        print(word)
    
    # show cosine similarity between 2 words
    print(vec_loader.get_cosine_similarity("coast", "shore"))

