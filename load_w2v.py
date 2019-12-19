import os
import sys
import pickle
import gensim
import gensim.downloader as api


FNAME = ('data/glove-wiki-gigawaord-100.txt')


class VectorLoader():
    def __init__(self, fname=FNAME):
        self.fname = fname
        self.model = self._load_word2vec()

    def __call__(self, word_label):
        return self._get_vec(word_label)
        
    def _load_word2vec(self):
        # load cache file if exists
        cache_filename = self.fname.replace('.txt', '.pickle')
        if os.path.exists(cache_filename):
            with open(cache_filename, 'rb') as f:
                return pickle.load(f)

        # check model file
        try:
            os.path.exists(self.fname)
        except FileNotFoundError:
            print('Wrong file or file path')
            sys.exit(1)

        print('Load word2vec model file')

        # TODO adapt for another fname
        if self.fname == FNAME:
            model = api.load('glove-wiki-gigaword-100')
            
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


# for debug
if __name__ == "__main__":
    vec_loader = VectorLoader()
    print(vec_loader('apple'))

    
