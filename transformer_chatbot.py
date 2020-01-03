import tensorflow as tf
import tensorflow_datasets as tfds
import os
import re
import matplotlib.pyplot as plt
# import numpy as np
import time
import datetime

# Based on the following original non-subclassed implementation:
# https://medium.com/tensorflow/a-transformer-chatbot-tutorial-with-tensorflow-2-0-88bf59e66fe2

# the following returns the path to the downloaded dataset file:
download_from_url = tf.keras.utils.get_file(
    'cornell_movie_dialorgs.zip', origin='http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip',
    extract=True)
path_to_dataset = os.path.join(os.path.dirname(download_from_url), 'cornell movie-dialogs corpus')

path_to_movie_lines = os.path.join(path_to_dataset, 'movie_lines.txt')
path_to_movie_conversations = os.path.join(path_to_dataset, 'movie_conversations.txt')

# first take a look at a sample of the data:
with open(path_to_movie_lines, errors='ignore') as file:
    lines = file.readlines()
    print()
    for i, line in enumerate(lines[:10]):
        print("Example Line {}:".format(i), line)
    print("A big mess!! --> LET'S PREPROCESS!")

# movie_conversations.txt has the following format: ID of the first character, IDof the second character, ID of ID of
# the movie that this conversation occurred, and a list of line IDs. The character and movie information can be found
# in movie_characters_metadata.txt and movie_titles_metadata.txt respectively.

    # u0 + ++$+++ u2 + ++$+++ m0 + ++$+++ [‘L194’, ‘L195’, ‘L196’, ‘L197’]
    # u0 + ++$+++ u2 + ++$+++ m0 + ++$+++ [‘L198’, ‘L199’]
    # u0 + ++$+++ u2 + ++$+++ m0 + ++$+++ [‘L200’, ‘L201’, ‘L202’, ‘L203’]
    # u0 + ++$+++ u2 + ++$+++ m0 + ++$+++ [‘L204’, ‘L205’, ‘L206’]
    # u0 + ++$+++ u2 + ++$+++ m0 + ++$+++ [‘L207’, ‘L208’]

# movie_lines.txt has the following format: ID of the conversation line, ID of the character who uttered this phase, ID
# of the movie, name of the character and the text of the line.

    # L901 +++$+++ u5 +++$+++ m0 +++$+++ KAT +++$+++ He said everyone was doing it. So I did it.
    # L900 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ As in…
    # L899 +++$+++ u5 +++$+++ m0 +++$+++ KAT +++$+++ Now I do. Back then, was a different story.
    # L898 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ But you hate Joey
    # L897 +++$+++ u5 +++$+++ m0 +++$+++ KAT +++$+++ He was, like, a total babe

# Actual preprocessing -------------------------------------------------------------------------------------------------

# Somewhat arbitrary, for development efficiency purposes:
MAX_SAMPLES = 50000
MAX_LENGTH = 40


def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()  # remember, strip gets rid of the initial and ending characters
    # (spaces in this case)

    # creating a space between punctuations and words:
    sentence = re.sub(r"([?.!,])", r"\1", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z.?!,]+", " ", sentence)
    sentence = sentence.strip()
    return sentence


def load_conversations():
    # We first go to the lines in movie_lines:
    id2line = {}      # dictionary of line id to text:
    with open(path_to_movie_lines, errors='ignore') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')
        id2line[parts[0]] = parts[4]  # idline as dict key and line as dict value

    # We then go to the __? in movie_conversations:
    inputs, outputs = [], []
    with open(path_to_movie_conversations, 'r') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')

        conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
        for i in range(len(conversation) - 1):
            inputs.append(preprocess_sentence(id2line[conversation[i]]))
            outputs.append(preprocess_sentence(id2line[conversation[i + 1]]))
            if len(inputs) >= MAX_SAMPLES:
                return inputs, outputs

    return inputs, outputs


questions, answers = load_conversations()

print('\nSample question: {}'.format(questions[20]))
print('Sample answer: {}'.format(answers[20]))
# --> the preprocessing could be improved, for we lost info like the apostrophes in don't or in I'm.

# Build tokenizer:
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(questions + answers, target_vocab_size=2**13)

# Define start and end token:
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# Define vocab size with the start and end tokens:
VOCAB_SIZE = tokenizer.vocab_size + 2

print('Tokenized sample question: {}'.format(tokenizer.encode(questions[20])))

# Tokenize, filter, and pad sentences:


def tokenize_filter_by_maxlen(inputs, outputs):

    tokenized_inputs, tokenized_outputs = [], []

    # tokenize and add start and end tokens:
    for inputs, outputs in zip(inputs, outputs):
        input = START_TOKEN + tokenizer.encode(inputs) + END_TOKEN
        output = START_TOKEN + tokenizer.encode(outputs) + END_TOKEN

    # filter sentences based on sentence length:
        if len(input) and len(output) <= MAX_LENGTH:  # vs: len(inputs) <= MAX_LENGTH and len(outputs) <= MAX_LENGTH
            tokenized_inputs.append(input)
            tokenized_outputs.append(output)

    # ** we could pad it on the dataset object later on, but we will do it here for now
    # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

    return tokenized_inputs, tokenized_outputs


questions, answers = tokenize_filter_by_maxlen(questions, answers)

print(questions)

print('\nVocab size: {}'.format(VOCAB_SIZE))
print('Number of samples: {}'.format(len(questions)))

# Create tf.data.Dataset -----------------------------------------------------------------------------------------------

BATCH_SIZE = 64
BUFFER_SIZE = 20000

dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        # "Teacher forcing": passing the true output to the next time step regardless of what the model predicts at the
        # current time step (i.e. decoder inputs use the previous target as input):
        'dec_inputs': answers[:, :-1]
    },
    {
        # remove START_TOKEN from targets
        'outputs': answers[:, 1:]
    }
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
print(dataset)
dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes=({'inputs': (None,),
                                                           'dec_inputs': (None,)},
                                                          {'outputs': (None,)}))   # (None,) for lists of lists as input
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
print(dataset)

# ----------------------------------------------------------------------------------------------------------------------


def scaled_dot_product_attention(query, keys, values, mask=None):
    """Calculate the attention weights.
      q, k, v must have matching leading dimensions.
      k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.

      Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None.

      Returns:
        output, attention_weights
      """
    matmul = tf.matmul(query, keys, transpose_b=True)

    depth = tf.cast(tf.shape(keys)[-1], tf.float32)
    logits_scaled = matmul / tf.sqrt(depth)

    # add mask to not take into account padded tokens:
    if mask is not None:
        logits_scaled += (mask * -1e9)

    attention_weights = tf.nn.softmax(logits_scaled, axis=-1)
    output = tf.matmul(attention_weights, values)

    return output


class MultiHeadAttention(tf.keras.layers.Layer):
    """The MultiHeadAttention runs all 8 attention heads across all other locations in the sequence, returning a new
    vector of the same length at each location."""

    def __init__(self, d_model, num_heads, name="Multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)

        self.num_heads = num_heads
        self.d_model = d_model

        # Set the depth (dimensionality of the model for each head)
        # for each attention head based on the num of heads and depth of the overall model:
        self.depth = self.d_model // self.num_heads

        self.linear_q = tf.keras.layers.Dense(units=d_model)
        self.linear_q.input_spec = tf.keras.layers.InputSpec(shape=(None, None, None), min_ndim=2)

        self.linear_k = tf.keras.layers.Dense(units=d_model)
        self.linear_v = tf.keras.layers.Dense(units=d_model)
        self.concatenate_heads = tf.keras.layers.Concatenate()
        self.output_linear = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        """Split the last dimension into (num_heads, depth).
            Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
            """
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))  # -1 for seq_len
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def __call__(self, inputs, mask=None):

        # Handle both dictionary of datasets and a single dataset:
        if isinstance(inputs, dict):
            query, keys, values = inputs['query'], inputs['keys'], inputs['values']
        else:
            query, keys, values = inputs, inputs, inputs

        batch_size = tf.shape(query)[0]
        # query shape == (num_samples, seq_len_q, depth ("d_model"))
        # tf.shape(query): tf.Tensor([  1  60 512], shape=(3,), dtype=int32)

        # linear of query, keys, values:
        query = self.linear_q(query)
        keys = self.linear_k(keys)
        values = self.linear_v(values)

        # Split heads
        query = self.split_heads(query, batch_size)
        keys = self.split_heads(keys, batch_size)
        values = self.split_heads(values, batch_size)

        # Scaled dot product attention heads:
        attention = scaled_dot_product_attention(query, keys, values, mask)  # (batch_size, num_heads, seq_len_q, depth)

        # Concat the different heads outputs:
        # 1st, change the num_heads for the seq_len dimensions by transposing:
        concatenate_1 = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        # 2nd, reshape the input to put back together num_heads and depth into d_model (as before split heads step):
        concatenate_2 = tf.reshape(concatenate_1,
                                   shape=(batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        # Last linear transformation:
        output = self.output_linear(concatenate_2)

        return output


def create_padding_mask(x):
    """Masking of the padding made to the sequences due to their different length."""
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):
    """In the decoder, the self-attention layer is only allowed to attend to earlier positions in the output sequence.
    This is done by masking future positions (setting them to -inf)."""
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)


class PositionalEncoding(tf.keras.layers.Layer):
    """Meant to represent the distance (number of words) between word i and j, hence the name
    Relative Position Representation (RPR), since full attentional models like the transformer lose the sequential
    ordering information."""
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def __call__(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


# Positional Encoding sample -------------------------------------------------------------------------------------------
# sample_pos_encoding = PositionalEncoding(50, 512)
#
# plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
# plt.xlabel('Depth')
# plt.xlim((0, 512))
# plt.ylabel('Position')
# plt.colorbar()
# plt.show()
# ----------------------------------------------------------------------------------------------------------------------


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, units, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.multihead_att = MultiHeadAttention(d_model, num_heads)

        # Residual + normalization layers:
        self.add_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.add_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # dropout:
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)

        # 'point wise feed forward' (2 dense layers, teh 1st with relu activation):
        self.dense = tf.keras.layers.Dense(units, activation='relu')
        self.feed_forward = tf.keras.layers.Dense(d_model)

    def __call__(self, inputs, mask=None):

        x = self.multihead_att(inputs, mask)
        residual_norm = self.add_norm(x + inputs)

        x = self.dense(residual_norm)
        x = self.feed_forward(x)
        output = self.add_norm(x + residual_norm)

        return output


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, units, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.multihead_att = MultiHeadAttention(d_model, num_heads)
        self.multihead_att_2 = MultiHeadAttention(d_model, num_heads)

        # Residual + normalization layers:
        self.add_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.add_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.add_norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Regularization layers:
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_3 = tf.keras.layers.Dropout(dropout_rate)

        # 'point wise feed forward' (2 dense layers, the 1st with relu activation):
        self.dense = tf.keras.layers.Dense(units, activation='relu')
        self.feed_forward = tf.keras.layers.Dense(d_model)

    def __call__(self, inputs, enc_outputs, look_ahead_mask=None, padding_mask=None):

        x = self.multihead_att(inputs, look_ahead_mask)
        x = self.dropout_1(x)
        residual_norm_1 = self.add_norm_1(inputs + x)

        second_block_input = {'query': residual_norm_1, 'keys': enc_outputs, 'values': enc_outputs}

        x = self.multihead_att_2(second_block_input,  padding_mask)
        x = self.dropout_2(x)
        residual_norm_2 = self.add_norm_2(residual_norm_1 + x)

        x = self.dense(residual_norm_2)
        x = self.feed_forward(residual_norm_2)
        x = self.dropout_3(x)
        output = self.add_norm_3(residual_norm_2 + x)

        return output


class Encoder(tf.keras.layers.Layer):

    def __init__(self, vocab_size, d_model, num_heads, units, num_layers, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, input_length=MAX_LENGTH)
        # d_model is chosen on the basis of the embedding dimensionality that we choose, since the weights for our model
        # will come from the number of words together with their dimensionality in the embedding space, that is why
        # d_model and embedding dimensions match. If we would have liked the embedding dimensions to be 300, then
        # d_model would have been so as well.

        self.pos_encoding = PositionalEncoding(vocab_size, d_model)
        self.encoder_layers = [EncoderLayer(d_model, num_heads, units) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def __call__(self, inputs, padding_mask=None):

        seq_len = tf.shape(inputs)[1]

        x = self.embedding(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, padding_mask)

        return x


class Decoder(tf.keras.layers.Layer):

    def __init__(self, vocab_size, d_model, num_heads, units, num_layers, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, input_length=MAX_LENGTH)
        self.pos_encoding = PositionalEncoding(vocab_size, d_model)

        self.decoder_layers = [DecoderLayer(d_model, num_heads, units) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def __call__(self, inputs, enc_outputs, look_ahead_mask=None, padding_mask=None):

        seq_len = tf.shape(inputs)[1]

        x = self.embedding(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, enc_outputs, look_ahead_mask, padding_mask)

        return x


class Transformer(tf.keras.Model):

    def __init__(self, vocab_size, d_model, num_heads, units, num_layers):
        super(Transformer, self).__init__()

        self.d_model = d_model

        self.encoder = Encoder(vocab_size, d_model, num_heads, units, num_layers)
        self.decoder = Decoder(vocab_size, d_model, num_heads, units, num_layers)

        self.final_output = tf.keras.layers.Dense(vocab_size)

    def __call__(self, enc_inputs, dec_inputs, masking=False):

        # Create maskings:
        if masking is False:
            enc_padding_mask = None
            dec_padding_mask = None
            look_ahead_mask = None
        else:
            enc_padding_mask = create_padding_mask(enc_inputs)
            dec_padding_mask = create_padding_mask(enc_inputs)
            look_ahead_mask = create_look_ahead_mask(dec_inputs)

        enc_output = self.encoder(enc_inputs, enc_padding_mask)
        x = self.decoder(dec_inputs, enc_output, look_ahead_mask, dec_padding_mask)
        outputs = self.final_output(x)

        return outputs


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Training -------------------------------------------------------------------------------------------------------------

# Hyperparameters for testing the model and training
# (can play with these as well; the paper "Attention is all you need has several configurations and their performance):
num_layers = 6
d_model = 384
num_heads = 8
units = 1024
EPOCHS = 30
# vocab_size is still the same, defined at the beginning

# Create the model:
transformer_model = Transformer(VOCAB_SIZE, d_model, num_heads, units, num_layers)

# Define the loss function: --------------------------------------------------------------------------------------------


def loss_function(target, predicted):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')
    loss = loss_object(target, predicted)

    # get true for the values that are not 0, false if they are:
    mask = tf.cast(tf.math.not_equal(target, 0), tf.float32)
    loss *= mask

    return tf.reduce_mean(loss)


# Define metrics:
train_loss = tf.keras.metrics.Mean(name='training_loss')
train_accuracy = tf.keras.metrics.Accuracy(name='training_accuracy')

# Define Custom Learning rate and Optimizer: ---------------------------------------------------------------------------


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """From the formula (from the paper "Attention is all you need"):
     lrate = d_model ** -0.5 * min(step_num ** -0.5, step_num * warmup_steps ** -1.5). """
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)  # step_num ** -0.5  -> -1/2 is the reciprocal of the sqrt == -0.5
        arg2 = step * (self.warmup_steps ** - 1.5)  # step_num * warmup_steps ** -1.5

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)  # d_model ** -0.5 * min(arg1, arg2)


learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# Training loop and checkpoints: ---------------------------------------------------------------------------------------

# Checkpoints:
checkpoint_path = "./checkpoints/transformer_chatbot/train"
checkpoint = tf.train.Checkpoint(model=transformer_model, optimizer=optimizer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=3)

# Create summary writer for tensorboard:
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = './logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# python -m tensorboard.main --logdir=/logs/gradient_tape  (command)

# Training:
for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    for batch, (inputs, targets) in enumerate(dataset):

        with tf.GradientTape() as tape:

            prediction_logits = transformer_model(inputs['inputs'], inputs['dec_inputs'], masking=True)
            predictions = tf.argmax(prediction_logits, axis=-1)

            loss = loss_function(targets['outputs'], prediction_logits)

            gradients = tape.gradient(loss, transformer_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer_model.trainable_variables))

            train_loss(loss)  # this metric might be calculating the mean again after the loss function uses
            # tf.reduce_mean()
            train_accuracy(targets['outputs'], predictions)

        # Writes summaries for tensorboard:
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=batch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=batch)

    # ------------------------------------------------------------------------------------------------------------------
    # Print Progress information: --------------------------------------------------------------------------------------

        print('\nEpoch {} Batch {} => Nº Elements {} | Loss {:.4f} Accuracy {:.4f}'
              .format(epoch + 1, batch + 1, (batch + 1) * BATCH_SIZE, train_loss.result(), train_accuracy.result()))

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = checkpoint_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

    print('-' * 80)
    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
    print('Time taken for 1 epoch: {} secs/n; {} min'.format(time.time() - start, (time.time() - start)/60))
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

