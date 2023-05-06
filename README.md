# Music-Genre-Classification-using-Deep-Learning
## Data

For a project, I compared multiple Deep Learning models to classify and eventually predict music genres based on 30-seconds audio segments. The data used for this project is the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). For the purposes of this project, the original data was reduced to 8 genres (800 songs) and transformed such that for each song, 15 log-transformed Mel spectrograms are obtained. Each Mel spectrogram is an image file represented by a tensor of shape (80, 80, 1) which describes time, frequency and intensity of a song segment. The training data represent 80% of the total number of data points. I implemented a Parallel CNN, CNN-RNN and a stylised model, which are briefly discussed below:

## Parallel CNN

The first model is a parallel CNN and implemented as follows:

- First parallel branch:
  1. one convolutional layer processing the input data with 3 square filters of size 8, padding and leaky ReLU activation function with slope 0.3.
  2. one pooling layer which implements Max Pooling over the output of the convolutional layer, with pooling size 4.
  3. a layer flattening the output of the pooling.

- Second parallel branch:
  1. one convolutional layer processing the input data with 4 square filters of size 4, padding and leaky ReLU activation function with slope 0.3.
  2. one pooling layer which implements Max Pooling over the output of the convolutional layer, with size 2.
  3. a layer flattening the output of the pooling..

- Merging branch:
  1. a layer concatenating the outputs of the two parallel branches.
  2. a dense layer which performs the classification of the music genres using the approppriate activation function.

We use `tf.keras.losses.CategoricalCrossentropy()` as a loss function and mini-batch stochastic gradient descent as optimiser. The epoch size is set to 50.

## CNN-RNN

To implement a CNN-RNN model, we reduced the dimensionality of the dataset through `reduce_dimension` function to (80,80). The model's architecture is structured as follows: 

1. a convolutional layer with 8 square filters of size 4.
2. a max pooling layer that halves the dimensionality of the output.
3. a convolutional layer with 6 square filters of size 3.
4. a max pooling layer that halves the dimensionality of the output.
5. an LSTM layer with 128 units, returning the full sequence of hidden states as output.
6. an LSTM layer with 32 units, returning only the last hidden state as output.
7. a dense layer with 200 neurons and ReLU activation function.
8. a layer dropping out 20% of the neurons during training.
9. a dense layer which outputs the probabilities associated to each genre.

The training procedure is identical to before.

## Newly proposed model

In the final part I propose two models, `final_model_eff()` and `final_model_acc()`, which both significantly increase the classification accuracy. While `final_model_eff()` is more efficient, `final_model_acc()` attains the highest accuracies overall. The intuition behind both models and their implementation is further explained in the `main.ipynb` file. 
