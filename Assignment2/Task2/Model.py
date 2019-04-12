import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split

# CONSTANTS
MODEL_DIR = "./models/sentiment_classifier_v1.model"
TRAIN_DIR = "./data/train_features.csv"
TEST_DIR  = "./data/test_features.csv"
TARG_DIR  = "./data/targets.csv"
PRED_DIR  = "./data/predictions.csv"
LABELS    = 5
NODES     = 200
DROPOUT   = 0.5
WND_SIZE  = 100

# Read pre-processed data
X = np.genfromtxt(TRAIN_DIR, delimiter=',')
y = keras.utils.to_categorical(np.genfromtxt(TARG_DIR, delimiter=','), num_classes=LABELS)

# Data separation for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Model Arquitecture
model = Sequential()
model.add(Dense(NODES, input_dim=WND_SIZE, activation='relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NODES, activation='relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(LABELS, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train and Evaluate Model
model.fit(X_train, y_train, epochs=100, batch_size=100)
score = model.evaluate(X_test, y_test, batch_size=16)
print(score)

# Generate predictions for contest dataset
X_test = np.genfromtxt(TEST_DIR, delimiter=',')
pred_encoded = model.predict(X_test)
pred = np.zeros(pred_encoded.shape[0])

for it in range(pred_encoded.shape[0]):
    pred[it] = np.argmax(pred_encoded[it])

np.savetxt(PRED_DIR, pred, delimiter=',')
