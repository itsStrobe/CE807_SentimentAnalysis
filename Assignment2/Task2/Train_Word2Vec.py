from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from Util import ReadTrainData, ExtractCorpus

# path = get_tmpfile("word2vec.model")
sentences, targets = ReadTrainData()
corpus = ExtractCorpus(sentences)
model = Word2Vec(corpus, size=100, window=5, min_count=1, workers=4)
model.train(corpus, total_examples=len(corpus), epochs=10)
model.save("./models/word2vec_corpus.model")

print(model.wv.most_similar(positive="good"))
