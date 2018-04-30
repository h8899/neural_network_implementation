from utils import datasets
from applications import SentimentNet
from loss import SoftmaxCrossEntropy, L2
from optimizers import Adam
import numpy as np
np.random.seed(5242)

dataset = datasets.Sentiment()
model = SentimentNet(dataset.dictionary)
loss = SoftmaxCrossEntropy(num_class=2)

def schedule_function(lr, it, cycle):
    if(it % cycle == 0):
        return 0.00485
    else:
        return lr

adam = Adam(lr=0.00485, decay=0.000121,
            scheduler_func=schedule_function, cycle=10)
model.compile(optimizer=adam, loss=loss, regularization=L2(w=0.001))
train_results, val_results, test_results = model.train(
        dataset, 
        train_batch=128, val_batch=100, test_batch=100, 
        epochs=60, 
        val_intervals=100, test_intervals=300, print_intervals=5)