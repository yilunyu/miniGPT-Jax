# Build a mini GPT using Jax from scratch.

## Background
This repository contains a Coding Challenge for building a Mini GPT model using Jax and Flax. You will use Jax to solve a sequence of tasks that guide you towards creating a functional mini GPT model.

This guide is heavily inspired by Andrej Karpathy's [Let's Build GPT from Scratch video](https://www.youtube.com/watch?v=kCc8FmEb1nY). The flow of the Challenge closely matches Karpathy's video, so you can refer to the video if you get stuck. The major differences between the video and this Challenge are:
* The video is in PyTorch while this Challenge uses Jax.
* This Challenge forces you to reimplement all the code yourself by checking your code against a sequence of unit tests, whereas the video only contains the solution.
* This Challenge contains some additional tasks like implementing Attention in Einsum which are not covered by the video.

### Intended Audience
* You're curious about Jax and would like to gain hands-on experience working with it.
* You have a high-level understanding of how transformers work and would like to try implementing it in code.

### Prerequisites
* You have a high level understanding of the following concepts:
    * Backpropagation
    * Multi Layer Perceptron
    * Cross Entropy Loss
    * Softmax Function
    * Multi-head Attention
    * Tokenization
* You can understand numpy operations including:
    * Stacking / concatenating / slicing / reshaping arrays.
* You are comfortable coding with Python in Colab.
 
### What is Jax?
Jax is Google's new framework for building neural networks. At its core, Jax is a fast numerical processing library that can leverage hardware accelerators, so think Numpy on TPUs and GPUs. On top of that, Jax also has the following core features:
* Autograd: automatically calculate derivatives of Python functions with respect to the functions' inputs.
* Just-In-Time Compilation: compile your Python function consisting of a sequence of Jax operations into optimized machine code.
* pmap & shard_map: Easily parallelize training over multiple hardware accelerators using data parallelism or model parallelism.
* Eager execution: By default, Jax operations executes eagerly. So you can quickly understand the effects of your code changes. This is useful for interactive development and debugging.

### What is Flax?
Flax is the layering library built on top of Jax. Flax comes with a large set of off-the-shelf layers. It also manages keeping track of and updating model parameters. 

## Challenge Overview
We will build a transformer model shown in the illustration below. Our goal is to build a model that predicts the next `token` given a sequence of `previous tokens`. Check [here](https://seantrott.substack.com/p/tokenization-in-large-language-models) if you would like a refresher on tokenization.

![Diagram 1 - miniGPT bird's eye view (training time)](./pictures/image.png)

### Dataset and tokenization (already implemented for you)
The dataset we will be working with is the [tiny Shakespeare dataset](https://www.tensorflow.org/datasets/catalog/tiny_shakespeare), which contains 40k lines of Shakespeare's plays. The dataset contains 1.1M characters in total. We split 90% of the dataset into training split and the remaining into eval split.

In this challenge, we chose to tokenize the data using character-level tokenization. This means that the string "hello world" maps to the array ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"]. Then we feed the array to the model. We picked character-level tokenization because it is easy to implement and has a small vocabulary size, which will require fewer parameters to train.
### Warm up - Check the performance of a simple text decoder model
We implemented a SimpleDecoder Flax Module for you. This Module is a really simple decoder that predicts the next token given only the previous token.

The SimpleDecoder Flax module is defined as follows:
```
class SimpleDecoder(nn.Module):
  vocab_size: int

  def setup(self):
    self.token_embedding = nn.Embed(
        num_embeddings=self.vocab_size,
        features=self.vocab_size)

  def __call__(self, x):
    B, T = x.shape
    return self.token_embedding(x) # B, T, vocab_size
```
Let's break down the three methods.
* setup: The setup method is similar to the __init__ method in a normal Python class. This is where you declare the Submodules of a Flax Module. In this case we used Flax's built-in nn.Embed module.
  This module takes in an integer (which represents a token) as input, and spits out a vector of dimension `features`.
* __call__: This method defines the behavior of the forward pass through your Flax Module. It uses the submodules declared in `setup` to calculate how to generate the output of the current Module given the input.
In our case, we will simply call self.token_embedding Module, which turns our input of shape batch_size, sequence_length to batch_size, sequence_length, vocab_size.

At training time, we will pass a integer tensor 

Here's how we leverage the SimpleDecoder Module to define the training loop:
```
@jax.jit
def train_step(state, x, y):
  def _loss(params):
    predictions = state.apply_fn(params, x)
    loss = optax.softmax_cross_entropy_with_integer_labels(predictions, y)
    return loss.mean()
  loss, grads = jax.value_and_grad(_loss)(state.params)
  state = state.apply_gradients(grads=grads)
  return state, loss

# Initialize the decoder instance. Importantly, the decoder object doesn't contain any state.
# All state in Flax is stored separately in a dictionary when we make the decoder.init call below.
decoder = SimpleDecoder(vocab_size=vocab_size)

# Some boilerplate code.
random_key = jax.random.PRNGKey(0)
x = jnp.ones((batch_size, block_size), dtype=jnp.int16)
random_key, random_subkey = jax.random.split(random_key)

# This initializes the parameters in `model`. The parameters are stored in a nested dictionary in `params`.
# You can print `params` to see what it looks like.
params = decoder.init(random_subkey, x)
# This creates a state object that stores params and optimizer states.
state = init_train_state(
    decoder, params, learning_rate=learning_rate)

# x, y are both integer tensors (where each integer represents a token) of shape (batch_size, block_size). They represent a batch of training data.
x, y = get_batch(random_subkey, train_data, batch_size=batch_size, block_size=block_size)
train_step(state, x, y)
```

You should observe that before training, the model decodes a bunch of gibberish. Here is what I see:
```
Generated sequence: KplEzUplEzUplEzUplEz
```

After training until evaluation loss doesn't go down anymore, the model repeats itself like so:
```
Generated sequence: KI the the the the t
```
What is your intuition for why the model keeps repeating the word "the" over and over?
### Task 1 - Implement MiniGPT as shown in Diagram 1.

### Task 2 - 
