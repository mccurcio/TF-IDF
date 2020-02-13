from mrjob.job import MRJob
from mrjob.protocol import RawValueProtocol
import re
import math
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

WORD_RE = re.compile(r"[\w']+")


# The following code is an example on computing the most frequent word of a text document.
# Please learn from this example and change it to compute TFIDF of a txt file

class LineNumberProtocol(RawValueProtocol):
    def __init__(self):
        self.counter = 0

    def read(self, line):
        key = self.counter
        self.counter += 1
        return (key, line)

class MRTFIDF(MRJob):

    INPUT_PROTOCOL = LineNumberProtocol

    def mapper_get_words(self, key, line):
        # emit total number of documents
        yield None, 1
        words = set()
        # yield each word in the line
        for word in WORD_RE.findall(line):
            if not word.lower() in ENGLISH_STOP_WORDS:
                # emit frequency on review
                yield key, word.lower()
                words.add(word.lower())
        for word in words:
            # emit once for each word in a review
            yield word, 1

    def reducer_count_words(self, key, values):
        if key == None:
            # emit total number of documents
            yield None, sum(values)
        elif type(key) is int:
            # group repeated words in a review together
            words = {}
            for value in values:
                if value in words.keys():
                    words[value] += 1
                else:
                    words[value] = 1
            # emit total count in review
            yield None, {str(key): words}
        else:
            # emit number of documents a word appears in
            yield None, (key, sum(values))

    # discard the key; it is just None
    def reducer_tfidf(self, _, values):
        N = 0
        word_counts = {}
        review_counts = {}
        for value in values:
            if type(value) is int:
                N = value
            elif type(value) is list:
                word_counts[value[0]] = value[1]
            else:
                for key in value.keys():
                    review_counts[key] = value[key]
        idf = {}
        for word in word_counts.keys():
            idf[word] = math.log(float(N) / word_counts[word])
        for i in range(N):
            try:
                review_words = review_counts[str(i)]
                for word in review_words.keys():
                    tf = review_words[word]
                    yield i, (word, tf*idf[word])
            except KeyError:
                pass

    def steps(self):
        return [
            self.mr(mapper=self.mapper_get_words,
                    reducer=self.reducer_count_words),
            self.mr(reducer=self.reducer_tfidf)
        ]


if __name__ == '__main__':
    MRTFIDF.run()
