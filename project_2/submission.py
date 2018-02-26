## Submission.py for COMP6714-Project2
###################################################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
import gensim
import sys
import re
import spacy
nlp = spacy.load('en')

def process_data(input_data):
    filename = input_data
 
    final_file = zipfile.ZipFile(filename)
    file_list  = final_file.namelist()

    data_file = 'data_precess.txt'
    
    for file in file_list:
  
        m = re.match(r'^[a-z]+.*\.txt$', file)
        if m:

            content = final_file.read(file).decode().split('\n')
            content = [s for s in content if s.strip()]

            for line in content:
##                line = line.lower()
##                L = line.split()
##                s = ''
##                for word in L:
##                    m = re.match(r'^[a-z]+$', word)
##                    if m:
##                        s += word
##                        s += ' '
                        
                doc = nlp(line)
                for token in doc:
##                    m = re.match(r'^[a-zA-Z]+$', token.text)
##                    if not m:
##                        continue

                    if not token.is_alpha:
                        continue


                    if token.pos_ == 'ADJ':
                        with open(data_file, 'a') as f1:
                            f1.write(token.text.lower() + ' ')
                        continue

##                    if token.is_stop:
##                        continue

                    if len(token.text) == 1:
                        continue

                    if token.pos_ == 'PRON':
                        continue

                    if token.pos_ == 'PROPN':
                        continue

                    
                    if token.pos_ == 'DET':
                        continue

                    if token.pos_ == 'ADP':
                        continue

                    if token.pos_ == 'CCONJ':
                        continue

                    if token.pos_ == 'NOUN' or token.pos_ == 'VERB':
                        with open(data_file, 'a') as f2:
                            f2.write(str(token.lemma_).lower() + ' ')
                    
                        continue

                    if len(token.text.strip()) == 0:
                        continue
                    if not token.text.strip():
                        continue
                    
                    with open(data_file, 'a') as f3:
                        f3.write(str(token.text).lower() + ' ')
                    

    return data_file
                            
    


def generate_batch(data, reverse_dictionary, batch_size, num_samples, skip_window):

    global data_index
##    print('*', batch_size, num_samples)
    assert batch_size % num_samples == 0
    assert num_samples <= 2 * skip_window
    
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # span is the width of the sliding window
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span]) # initial buffer content = first sliding window
    
##    print('data_index = {}, buffer = {}'.format(data_index, [reverse_dictionary[w] for w in buffer]))

    data_index += span
    for i in range(batch_size // num_samples):
        context_words = [w for w in range(span) if w != skip_window]
        random.shuffle(context_words)
        words_to_use = collections.deque(context_words) # now we obtain a random list of context words
        for j in range(num_samples): # generate the training pairs
            batch[i * num_samples + j] = buffer[skip_window]
            context_word = words_to_use.pop()
            labels[i * num_samples + j, 0] = buffer[context_word] # buffer[context_word] is a random context word
        
        # slide the window to the next position    
        if data_index == len(data):
            buffer = data[:span]
            data_index = span
        else: 
            buffer.append(data[data_index]) # note that due to the size limit, the left most word is automatically removed from the buffer.
            data_index += 1
        
##        print('data_index = {}, buffer = {}'.format(data_index, [reverse_dictionary[w] for w in buffer]))
        

    data_index = (data_index + len(data) - span) % len(data) # move data_index back by `span`
    return batch, labels



data_index = 0

def adjective_embeddings(data_file, embeddings_file_name, num_steps, embedding_dim):
    f = open(data_file)
    L = f.readlines()
    lexicon = []
    for line in L:
        s = line.split()
        s = [i.strip() for i in s if i.strip()]
        lexicon.extend(s)


    global data_index
    
    count = collections.Counter(lexicon).most_common()
    vocabulary_size = len(count)


    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    for word in lexicon:
        index = dictionary.get(word)
        data.append(index)

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))   



    # Specification of Training data:
    batch_size = 30   # Size of mini-batch for skip-gram model.
    embedding_size = embedding_dim  # Dimension of the embedding vector.
    skip_window = 3      # How many words to consider left and right of the target word.
    num_samples = 6         # How many times to reuse an input to generate a label.
    num_sampled = 64      # Sample size for negative examples.
    logs_path = './log/'

    # Specification of test Sample:
    sample_size = 20       # Random sample of words to evaluate similarity.
    sample_window = 100    # Only pick samples in the head of the distribution.
    sample_examples = np.random.choice(sample_window, sample_size, replace=False) # Randomly pick a sample of size 16

    ## Constructing the graph...
    graph = tf.Graph()

    with graph.as_default():
        
        with tf.device('/cpu:0'):
            # Placeholders to read input data.
            with tf.name_scope('Inputs'):
                train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
                
            # Look up embeddings for inputs.
            with tf.name_scope('Embeddings'):            
                sample_dataset = tf.constant(sample_examples, dtype=tf.int32)
                embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)
         
                # Construct the variables for the NCE loss
                softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                          stddev=1.0 / math.sqrt(embedding_size)))
                softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
            
            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            with tf.name_scope('Loss'):
                loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, 
                                                 labels=train_labels, inputs=embed, 
                                                 num_sampled=num_sampled, num_classes=vocabulary_size))
            
            # Construct the Gradient Descent optimizer using a learning rate of 0.01.
            with tf.name_scope('Adam'):
                optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss)

            # Normalize the embeddings to avoid overfitting.
            with tf.name_scope('Normalization'):
                norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
                normalized_embeddings = embeddings / norm
                
            sample_embeddings = tf.nn.embedding_lookup(normalized_embeddings, sample_dataset)
            similarity = tf.matmul(sample_embeddings, normalized_embeddings, transpose_b=True)
            
            # Add variable initializer.
            init = tf.global_variables_initializer()
            
            
            # Create a summary to monitor cost tensor
            tf.summary.scalar("cost", loss)
            # Merge all summary variables.
            merged_summary_op = tf.summary.merge_all()


    num_steps = num_steps

    with tf.Session(graph=graph) as session:
        session.run(init)
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        
        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(data, reverse_dictionary, batch_size, num_samples, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            
            # We perform one update step by evaluating the optimizer op using session.run()
            _, loss_val, summary = session.run([optimizer, loss, merged_summary_op], feed_dict=feed_dict)
            
            summary_writer.add_summary(summary, step )
            average_loss += loss_val

            if step % 5000 == 0:
                if step > 0:
                    average_loss /= 5000
                
                    # The average loss is an estimate of the loss over the last 5000 batches.
                    #print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0

                
        final_embeddings = normalized_embeddings.eval()

        with open(embeddings_file_name ,'a') as file:
            file.write('{} {}\n'.format(vocabulary_size, str(embedding_size)))
            for i in range(len(reverse_dictionary)):
                file.write(reverse_dictionary[i] + ' ')
                for j in (final_embeddings[i]):
                    file.write(str.format('{0:.6f}', float(j)) + ' ')
                file.write('\n')

def Compute_topk(model_file, input_adjective, top_k):
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)
    try:
        L = model.most_similar(input_adjective, topn=1000)
        s = ''
        for i in L:
            s += i[0]
            s += ' '

        R = []
        doc = nlp(s)
        for token in doc:
            if token.pos_ == 'ADJ':
                R.append(str(token.text))


        res = R[:top_k]
        return res

    except KeyError:
        return []
    
    
    
