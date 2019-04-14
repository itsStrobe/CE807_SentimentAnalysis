import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
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
ACT_LAYER = 'relu'
OPTIMIZER = 'rmsprop'

def create_model(num_nodes=NODES, input_dim=WND_SIZE, activation=ACT_LAYER, dropout_rate=DROPOUT, optimizer=OPTIMIZER):

    # Model Arquitecture
    model = Sequential()
    model.add(Dense(num_nodes, input_dim=input_dim, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_nodes, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_nodes, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(LABELS, activation='softmax'))

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model

# Read pre-processed data
X = np.genfromtxt(TRAIN_DIR, delimiter=',')
y = keras.utils.to_categorical(np.genfromtxt(TARG_DIR, delimiter=','), num_classes=LABELS)

# Data separation for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Grid Search Using 10-Fold Cross Validation
num_nodes    = [50, 100, 150, 200, 250, 300]
activation   = ['relu', 'sigmoid', 'tanh', 'linear']
dropout_rate = [0.0, 0.3, 0.5, 0.7, 0.9]
optimizer    = ['rmsprop', 'adam']
epochs       = [1, 10, 100]
batch_size   = [1, 10, 100]

model       = KerasClassifier(build_fn=create_model)
param_grid  = dict(num_nodes=num_nodes, dropout_rate=dropout_rate, activation=activation, optimizer=optimizer, epochs=epochs, batch_size=batch_size)
grid        = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10)
best_model  = grid.fit(X, y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Train and Evaluate Model
# best_model.fit(X_train, y_train, epochs=100, batch_size=100)
score = best_model.evaluate(X_test, y_test, batch_size=16)
print(score)

# Generate predictions for contest dataset
X_test = np.genfromtxt(TEST_DIR, delimiter=',')
pred_encoded = best_model.predict(X_test)
pred = np.zeros(pred_encoded.shape[0])

for it in range(pred_encoded.shape[0]):
    pred[it] = np.argmax(pred_encoded[it])

np.savetxt(PRED_DIR, pred, delimiter=',')
