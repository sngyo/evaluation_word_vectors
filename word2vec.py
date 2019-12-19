import os
import sys
import pickle
import gensim
import gensim.downloader as api
import argparse


# FNAME = 'data/glove-wiki-gigawaord-100.txt'
MODEL_NAMES = ['glove-wiki-gigaword-100', ]


class Word2Vec():
    def __init__(self, modelname=MODEL_NAMES[0]):
        self.modelname = modelname
        self.model = self._load_word2vec()

    def __call__(self, word_label):
        return self._get_vec(word_label)
        
    def _load_word2vec(self):
        # load cache file if exists
        cache_filename = 'data/' + self.modelname + '.pickle'
        if os.path.exists(cache_filename):
            with open(cache_filename, 'rb') as f:
                print('load from cache')
                return pickle.load(f)

        print('Load word2vec model file')
        model = api.load(self.modelname)
            
        # print('Create cache of word2vec model')
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
        return self.model.most_similar(word_label)


# for debug
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', '-m', default=MODEL_NAMES[0], choices=MODEL_NAMES,
        help='model name: ' + ' | '.join(MODEL_NAMES)\
        + ' (default:' + MODEL_NAMES[0]+ ')'
    )
    args = parser.parse_args()
    
    vec_loader = Word2Vec(modelname=args.model)
    word_label = input('test word label: ')

    # show word vector itself
    # print(vec_loader(word_label))

    # show 10 most similar words
    sim_lis = vec_loader.most_similar(word_label)
    for word in sim_lis:
        print(word)

    
