# Build a mini GPT using Jax from scratch.

## Background
This repository contains a Coding Challenge for building a Mini GPT model using Jax and Flax. You will use Jax to solve a sequence of tasks that guide you towards creating a functional mini GPT model.

This guide is heavily inspired by Andrej Karpathy's [Let's Build GPT from Scratch video](https://www.youtube.com/watch?v=kCc8FmEb1nY). The flow of the Challenge closely matches Karpathy's video, so you can refer to the video if you get stuck, but watching the video is not required to finish the Challenge. The major differences between the video and this Challenge are:
* The video is in PyTorch while this Challenge uses Jax.
* This Challenge forces you to reimplement the code yourself by checking your code against unit tests, whereas the video only contains the solution.
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
The dataset we will be working with is the [tiny Shakespeare dataset](https://www.tensorflow.org/datasets/catalog/tiny_shakespeare), which contains a corpus of Shakespeare's plays. The dataset contains 1.1M characters in total. The entire dataset is contained in a single string with 1.1M characters. We split the first 90% of the string into training split and the remaining into eval split. Check the `Load and tokenize dataset` section for more detail.

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
    emb = self.token_embedding(x) # B, T, vocab_size
    return emb
```
Let's break down the three methods.
* setup: The setup method is similar to the __init__ method in a normal Python class. This is where you declare the Submodules of a Flax Module. In this case we used Flax's built-in nn.Embed module.
  This module takes in an integer (which represents a token) as input, and spits out a vector of dimension `features`.
* __call__: This method defines the behavior of the forward pass through your Flax Module. It uses the submodules declared in `setup` to calculate how to generate the output of the current Module given the input. In SimpleDecoder's case, the input, x, is a integer tensor of shape batch_size, sequence_length. x[i, j] represents the jth token at the ith element in the batch. self.token_embedding(x) will generate a new tensor of shape batch_size, sequence_length, vocab_size, turning the integer token x[i, j] into a 1-dimensional vector stored at emb[i, j, :].

#### Let's run SimpleDecoder with randomly initialized weights
```
# Boilerplate to initialize weights.
decoder = SimpleDecoder(vocab_size=vocab_size)
start_token = 23
dummy = jnp.ones((4, 8), dtype=jnp.int16)
params = decoder.init(jax.random.PRNGKey(0), dummy)

# Generate text from randomly initialized SimpleDecoder model.
generated_sequence = decoder.apply(params, start_token, method=decoder.generate, max_length=20)
print("Generated sequence:", decode(generated_sequence))
```
The print statement should produce some gibberish:
```
Generated sequence: KplEzUplEzUplEzUplEz
```

#### Training loop deep dive
Next, we will train the model to predict the next token in the training text given the previous token. The get_batch utility function prepares a batch of training data for us. Let's take a look:
```
def get_batch(random_key, data, batch_size, block_size):
  """Generate a batch of data of inputs x and targets y.

  Args:
    random_key (jax.random.PRNGKey): Random number generator key.
    data (array-like): 1d JAX array of integer tokens
    batch_size (int): Batch size.
    block_size (int): The maximum input context length.
  
  Returns:
    x (array-like): 2d JAX array of shape (batch_size, block_size).
    y (array-like): 2d JAX array of shape (batch_size, block_size).
        x[i, j] == y[i, j-1] where j > 0.
  """
  # generate a small batch of data of inputs x and targets y
  ix = jax.random.randint(random_key, shape=(batch_size, 1), minval=0, maxval=len(data)-block_size)
  x = dynamic_slice_vmap(data, ix, (block_size,))
  y = dynamic_slice_vmap(data, ix+1, (block_size,))
  return x, y
```
The important thing to note here is that y is that y[i, j] represents the target token for the input at x[i, j] and x[i, j] == y[i, j -1] for all i, j where j > 0. For example, if x is [["h" "e" "l" "l" "o"]] then y would be [["e" "l" "l" "o" " "]]. The training loop for SimpleDecoder will try to learn a function f such that f("h") -> "e", f("e") -> "l", f("l") -> "l", f("l") -> "o", f("o") -> " ". As you can imagine, this function f will not be very good at predicting the next token because its context length is too short. In Task 1, we'll change the architecture to use transformers so that we'll instead try to learn a function g such that g(["h"]) -> "e", g(["h" "e"]) -> "l", g(["h" "e" "l"]) -> "l", g(["h" "e" "l" "l"]) -> "o", g(["h" "e" "l" "l" "o"]) -> "o".

```
@jax.jit
def train_step(state, x, y):
  """Run one step of training.
  Args:
    state (jax.training.TrainState): Jax TrainState containing weights and
      optimizer states.
    x (array-like): 2d JAX int array of shape (batch_size, block_size).
    y (array-like): 2d JAX int array of shape (batch_size, block_size).

  Returns:
    state (jax.training.TrainState): The new train state after applying
      gradient descent on weights and updating optimizer states.
    loss (float): Loss for this training step.
  """
  def _loss(params):
    predictions = state.apply_fn(params, x) # B, T, vocab_size
    loss = optax.softmax_cross_entropy_with_integer_labels(predictions, y)
    return loss.mean()
  loss, grads = jax.value_and_grad(_loss)(state.params)
  state = state.apply_gradients(grads=grads)
  return state, loss

# Initialize the decoder instance. Importantly, the decoder object doesn't contain any state.
# All state in Flax is stored separately in the state variable below.
decoder = SimpleDecoder(vocab_size=vocab_size)

# initialize_train_state will initialize the weights and optimizer states of all submodules declared by decoder and store them
# in the `state` variable.
state = initialize_train_state(decoder)
x, y = get_batch(random_subkey, train_data, batch_size=batch_size, block_size=block_size)

state, loss = train_step(state, x, y)
```
The train_step function will generate a `predictions` tensor of shape batch_size, sequence_length, vocab_size. Then we calculate the cross entropy loss between the predictions and the groundtruth tokens contained in y. Finally we calculate gradients and backpropagate the gradients to update the weights.

The last thing to notice is that the @jax.jit decorator for train_step function. jit stands for Just-In-Time compilation. jax.jit is a core functionality of jax that compiles a user-defined Python function that executes Jax operations into optimized low-level instructions that executes more quickly.
> [!Important]
> If you were to apply @jax.jit decorator to get_batch, you should observe an error. Can you figure out why jax.jit works for train_step but not get_batch?

You're now ready to execute the run_training_loop cell to train the model. After training until evaluation loss doesn't go down anymore, the model repeats itself like so:
```
Generated sequence: KI the the the the t
```
> [!NOTE]
> What is your intuition for why the model keeps repeating the word "the" over and over?
### Task 1 - Implement MiniGPT as shown in Diagram 1.
This is the first task where you will be writing code. Fill out the TODO sections in the boilerplate code in Task 1. Your goal is to achieve a eval loss of < 1.9 in the TestTask1 unit test. When you reach this milestone, you should notice that the model produces sensible Shakespearean words. You may interactively test your code in the Colab by running the run_training_loop/train_step/eval_step functions.

### Task 2 - Implement SelfAttention.
In Task 1, we used Flax's built-in Attention Module. In this task, we will implement this module from scratch. We will be implementing causal attention, which means that in the self-attention module, the query for a token at position t can only refer to keys and values for tokens whose positions are <= t. Refer to the decoder Attention section in this [link](https://www.geeksforgeeks.org/self-attention-in-nlp/) for a refresher on Self Attention. If you are stuck, this [link](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html) walks you through implementing Self Attention in Pytorch. 

To pass this task, you will need to pass the TestAttention unit test, which checks that your Attention implementation is functionally correct.

### Task 3 - Implement MultiHeadAttention using Einsum.
Einsum is a notation that allows you to succinctly expressing a combination of matrix multiplication and addition operations. Einsum operations are highly optimized in Jax and are widely used in developing Neural Networks. This link has a great [introduction](https://rockt.github.io/2018/04/30/einsum) to Einsum. Your goal in this task is to re-implement MultiHeadAttention using Einsum Expressions.

