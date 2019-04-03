import numpy as np
from gensim.models import Word2Vec
from Util import ReadTrainData, ReadTestData, ExtractCorpus

# CONSTANTS
MODEL_DIR = "./models/word2vec_corpus.model"
TRAIN_DIR = "./data/train_features.csv"
TEST_DIR  = "./data/test_features.csv"
TARG_DIR  = "./data/targets.csv"
RETRAIN   = False
WND_SIZE  = 100

def GenerateFeatVector(sentence, wnd_size = WND_SIZE, model=None):
    if(model is None):
        model = Word2Vec.load(MODEL_DIR)
    
    featVector = np.empty((0, WND_SIZE))

    for word in sentence:
        featVector = np.append(featVector, [model.wv[word]], axis=0)

    return np.average(featVector, axis=0).reshape(1, WND_SIZE)

def GenerateFeatMatrix(sentences, wnd_size = WND_SIZE, model=None):
    it = 0
    featMatrix = np.empty((sentences.shape[0], WND_SIZE))

    for sentence in sentences:
        sentence = sentence.split()
        featMatrix[it, :] = GenerateFeatVector(sentence, wnd_size=wnd_size, model=model)
        it += 1

    return featMatrix

def main():
    # Read datasets
    train, targets = ReadTrainData()
    test = ReadTestData()
    
    # Retrain Word2Vec if needed
    if(RETRAIN):
        full_dataset = np.append(train, test, axis=0)
        corpus = ExtractCorpus(full_dataset)
        model = Word2Vec(corpus, size=WND_SIZE, window=5, min_count=1, workers=4)
        model.train(corpus, total_examples=len(corpus), epochs=10)
        model.save(MODEL_DIR)

    # Load Model
    model = Word2Vec.load(MODEL_DIR)

    # Check the model was loaded succesfully
    print(model.wv["Happy"])

    train_sent = train[:, 2]
    targets    = targets
    test_sent  = test[:, 2]

    train_feat = GenerateFeatMatrix(train_sent, wnd_size = WND_SIZE, model=model)
    print("Finished Training Matrix")
    test_feat = GenerateFeatMatrix(test_sent, wnd_size = WND_SIZE, model=model)
    print("Finished Test Matrix")

    np.savetxt(TRAIN_DIR, train_feat, delimiter=',')
    np.savetxt(TARG_DIR, targets, delimiter=',')
    np.savetxt(TEST_DIR, test_feat, delimiter=',')



if __name__ == "__main__":
    main()
