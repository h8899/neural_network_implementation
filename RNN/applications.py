from layers import *
from rnn_layers import *
from models import Model

def SentimentNet(word_to_idx):
    """Construct a RNN model for sentiment analysis

    # Arguments:
        word_to_idx: A dictionary giving the vocabulary. It contains V entries,
            and maps each string to a unique integer in the range [0, V).
    # Returns
        model: the constructed model
    """
    vocab_size = len(word_to_idx)

    model = Model()
    model.add(FCLayer(vocab_size, 300, name='embedding', initializer=Guassian(std=0.01)))
    model.add(ReLU(name='relu1'))
    model.add(BidirectionalRNN(RNNCell(in_features=300, units=100, initializer=Guassian(std=0.01))))
    model.add(Dropout(ratio=0.5, name='dropout1'))

    model.add(FCLayer(200, 32, name='fclayer1', initializer=Guassian(std=0.01)))
    model.add(ReLU(name='relu2'))
    model.add(TemporalPooling()) # defined in layers.py
    model.add(Dropout(ratio=0.5, name='dropout2'))
    model.add(FCLayer(32, 2, name='fclayer2', initializer=Guassian(std=0.01)))
    
    return model