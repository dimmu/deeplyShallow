from fuzzywuzzy import fuzz

metric = fuzz.ratio

class Vocab(object):
    def __init__(self, words):
        self.word2index = {}
        self.index2word = []
        self.size = 0
        for word in words.split():
            word = word.lower()
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.index2word.append(word)
                self.size += 1

    def getindex(self, word):
        word = word.lower()
        if word in self.word2index:
            return self.word2index[word]
        # fuzzy compare
        else:
            max_score = 0
            closest_word = self.index2word[0]
            for w in self.word2index:
                score = metric(word, w)
                if score > max_score:
                    max_score = score
                    closest_word = w
            # we have these 'found' words point to
            # their closet match
            self.word2index[word] = self.word2index[closest_word]
            return self.word2index[word]

    def getword(self, index):
        if index < len(self.index2word):
            return self.index2word[index]
        else:
            raise "IndexError"



