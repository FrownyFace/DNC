# DNC in Tensorflow
A Differentiable neural computer is a neural network coupled with some memory. The memory is written to and read from using write and read heads whose values are derived from the neural network. You can learn more about it from DeepMind's paper [here](http://www.nature.com/articles/nature20101.epdf?author_access_token=ImTXBI8aWbYxYQ51Plys8NRgN0jAjWel9jnR3ZoTv0MggmpDmwljGswxVdeocYSurJ3hxupzWuRNeGvvXnoO8o4jTJcnAyhGuZzXJ1GEaD-Z7E6X_a9R-xqJ9TfJWBqz).

# About this Implementation
It's a straightforward implementation from the paper. The neural network used here is a two layer feedforward net but can be substituted for basically anything else - DeepMind used a recurrent neural net. 

The task that it is trying to accomplish is to repeat a one-hot sequence padded with zeros from its memory. For example, for an input 

[[1 0]
[0 1]
[0 0]
[0 0]] 

the targeted output is

[[0 0]
[0 0]
[1 0]
[0 1]]

With the sequence length = 4 and sequence width = 4, a DNC(num_words=10, words_size=4, num_heads=1) converges in less than 1000 iterations.

# TODO
As you increase num_words or word_size, the DNC's output goes to positive infinity as does the loss. Need to figure out why.
