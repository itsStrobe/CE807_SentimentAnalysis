from gensim.models import Word2Vec
from Util import ReadTrainData

# CONSTANTS
MODEL_DIR = "./models/word2vec_corpus.model"

# Load Word2Vec Model
model = Word2Vec.load(MODEL_DIR)


