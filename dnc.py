import numpy as np
import tensorflow as tf


class DNC:
    def __init__(self, input_size, output_size, num_words=256, word_size=64, num_heads=4):
        self.input_size = input_size
        self.output_size = output_size
        self.num_words = num_words #N
        self.word_size = word_size #W
        self.num_heads = num_heads #R

        self.interface_size = num_heads*word_size + 3*word_size + 5*num_heads + 3
        # Read Vectors (num_heads*word_size)
        # Read Strength (num_heads)
        # Write Key (word_size)
        # Write Strength (1)
        # Erase Vector (word_size)
        # Write Vector (word_size)
        # Free Gates (num_heads)
        # Allocation Gate (1)
        # Write Gate (1)
        # Read Modes (num_heads*3)
        self.nn_input_size = num_heads * word_size + input_size
        self.nn_output_size = output_size + self.interface_size

        self.interface_vec = tf.truncated_normal([1, self.interface_size])

        self.mem_mat = tf.zeros([num_words, word_size]) #N*W
        self.usage_vec = tf.fill([num_words, 1], 1e-6) #N*1
        self.link_mat = tf.zeros([num_words*num_words]) #N*N
        self.precedence_weight = tf.zeros([num_words, 1]) #N*1

        self.read_weights = tf.fill([num_words, num_heads], 1e-6) #N*R
        self.write_weights = tf.fill([num_words, 1], 1e-6) #N*1

        self.read_vecs = tf.fill([num_words, num_heads], 1e-6) #N*R

        self.W1 = tf.Variable(tf.truncated_normal([self.nn_input_size,128]), stdev=0.1, name='Layer 1 Weights', dtype=tf.float32)
        self.b1 = tf.Variable(tf.zeros([128]), name='Layer 1 Bias', dtype=tf.float32)
        self.W2 = tf.Variable(tf.truncated_normal([128,256]), stdev=0.1, name='Layer 2 Weights', dtype=tf.float32)
        self.b2 = tf.Variable(tf.zeros([256]), name='Layer 2 Bias', dtype=tf.float32)

    def content_lookup(self, key, str):
        norm_mem = tf.nn.l2_normalize(self.mem_mat, dim=1) #N*W
        norm_key = tf.nn.l2_normalize(key) #1*1 for write, 1*R for read
        sim = tf.matmul(norm_mem, norm_key, transpose_b=True) #N*1 for write, N*R for read
        return tf.nn.softmax(sim*str, dim=1)

    def allocation_weighting(self):
        sorted_usage_vec, free_list = tf.nn.top_k(-1 * tf.squeeze(self.usage_vec), k=self.num_words)
        sorted_usage_vec *= -1
        cumprod = tf.cumprod(sorted_usage_vec, axis=0, exclusive=False)
        return (1-sorted_usage_vec)*cumprod

    def step_m(self, x):
        flat_read_vec = tf.reshape(self.read_vecs, [1, self.word_size*self.num_heads])
        input = tf.concat(1, [x, flat_read_vec])

        l1_out = tf.matmul(input, self.W1) + self.b1
        l1_act = tf.nn.relu(l1_out)
        l2_out = tf.matmul(l1_act, self.W2) + self.b2
        l2_act = tf.nn.relu(l2_out)

        partition = [0]*(self.num_heads*self.word_size) + [1]*(self.num_heads) + [2]*(self.word_size) + [3] + \
                    [4]*(self.word_size) + [5]*(self.word_size) + [6]*(self.num_heads) + [7] + [8] + [9]*(self.num_heads*3)

        (read_keys, read_str, write_key, write_str, erase_vec, write_vec, free_gates, alloc_gate, write_gate, read_modes) = \
        tf.dynamic_partition(self.interface_vec, partition, 10)

        read_keys = tf.reshape(read_keys,[self.num_heads, self.word_size]) #R*W
        read_str = 1 + tf.nn.softplus(tf.expand_dims(read_str, 0)) #1*R
        write_key = tf.expand_dims(write_key, 0)#1*W
        write_str = 1 + tf.nn.softplus(tf.expand_dims(write_str, 0)) #1*1
        erase_vec = tf.nn.sigmoid(tf.expand_dims(erase_vec, 0)) #1*W
        write_vec = tf.expand_dims(write_vec, 0) #1*W
        free_gates = tf.nn.sigmoid(tf.expand_dims(free_gates, 0)) #1*R
        alloc_gate = tf.nn.sigmoid(alloc_gate) #1
        write_gate = tf.nn.sigmoid(write_gate) #1
        read_modes = tf.nn.softmax(tf.reshape(read_modes, [3, self.num_heads])) #3*R

        retention_vec = tf.reduce_prod(1-free_gates*self.read_weights, reduction_indices=1)
        self.usage_vec = (self.usage_vec + self.write_weights - self.usage_vec * self.write_weights) * retention_vec

        alloc_weights = self.allocation_weighting() #N*1
        write_lookup_weights = self.content_lookup(write_key, write_str) #N*1
        self.write_weights = write_gate*(alloc_gate*alloc_weights + (1-alloc_gate)*write_lookup_weights)

        self.mem_mat = self.mem_mat*(1-tf.matmul(self.write_weights, erase_vec)) + tf.matmul(self.write_weights, write_vec)

        nnweight_vec = tf.matmul(self.write_weights,tf.ones([1,self.num_words])) #N*N
        self.link_mat = (1-nnweight_vec-tf.transpose(nnweight_vec))*self.link_mat + tf.matmul(self.write_weights, tf.transpose(self.precedence_weight))

        self.precedence_weight = (1-tf.reduce_sum(self.write_weights, reduction_indices=0))*self.precedence_weight + self.write_weights

        forw_w = read_modes[2]*tf.matmul(self.link_mat, self.read_weights) #(N*N,N*R)->N*R
        look_w = read_modes[1]*self.content_lookup(read_keys, read_str) #N*R
        back_w = read_modes[0]*tf.matmul(self.link_mat, self.read_weights, transpose_b=True) #N*R

        self.read_weights = back_w + look_w + forw_w #N*R
        self.read_vecs = tf.matmul(self.mem_mat, self.read_weights, transpose_a=True) #(W*N,N*R)->W*R
