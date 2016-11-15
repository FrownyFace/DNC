import numpy as np
import tensorflow as tf


class DNC:
    def __init__(self, input_size, output_size, num_words=256, word_size=64, num_heads=4):
        self.input_size = input_size #X
        self.output_size = output_size #Y
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

        self.nn_out = tf.truncated_normal([1, self.output_size], stddev=0.1)
        self.interface_vec = tf.truncated_normal([1, self.interface_size], stddev=0.1)

        self.mem_mat = tf.zeros([num_words, word_size]) #N*W
        self.usage_vec = tf.fill([num_words, 1], 1e-6) #N*1
        self.link_mat = tf.zeros([num_words*num_words]) #N*N
        self.precedence_weight = tf.zeros([num_words, 1]) #N*1

        self.read_weights = tf.fill([num_words, num_heads], 1e-6) #N*R
        self.write_weights = tf.fill([num_words, 1], 1e-6) #N*1

        self.read_vecs = tf.fill([num_heads, word_size], 1e-6) #R*W

        ###NETWORK VARIABLES
        self.i_data = tf.placeholder(tf.float32, [1, self.input_size], name='input_node')
        self.o_data = tf.placeholder(tf.float32, [1, self.output_size], name='output_node')

        self.W1 = tf.Variable(tf.truncated_normal([self.nn_input_size, 128], stddev=0.1), name='layer1_weights', dtype=tf.float32)
        self.b1 = tf.Variable(tf.zeros([128]), name='layer1_bias', dtype=tf.float32)
        self.W2 = tf.Variable(tf.truncated_normal([128, self.nn_output_size], stddev=0.1), name='layer2_weights', dtype=tf.float32)
        self.b2 = tf.Variable(tf.zeros([self.nn_output_size]), name='layer2_bias', dtype=tf.float32)

        ###DNC OUTPUT WEIGHTS
        self.nn_out_weights = tf.Variable(tf.truncated_normal([self.nn_output_size, self.output_size], stddev=1.0), name='net_output_weights')
        self.interface_weights = tf.Variable(tf.truncated_normal([self.nn_output_size, self.interface_size], stddev=1.0), name='interface_weights')
        self.read_vecs_out_weight = tf.Variable(tf.truncated_normal([self.num_heads*self.word_size, self.output_size], stddev=1.0), name='read_vector_weights')

    def content_lookup(self, key, str):
        norm_mem = tf.nn.l2_normalize(self.mem_mat, dim=1) #N*W
        norm_key = tf.nn.l2_normalize(key, dim=0) #1*1 for write, 1*R for read
        sim = tf.matmul(norm_mem, norm_key, transpose_b=True) #N*1 for write, N*R for read
        return tf.nn.softmax(sim*str, dim=1)

    def allocation_weighting(self):
        sorted_usage_vec, free_list = tf.nn.top_k(-1 * tf.squeeze(self.usage_vec), k=self.num_words)
        sorted_usage_vec *= -1
        cumprod = tf.cumprod(sorted_usage_vec, axis=0, exclusive=False)
        return (1-sorted_usage_vec)*cumprod

    def step_m(self, x):
        input = tf.concat(0, [x, tf.reshape(self.read_vecs, [1, self.num_heads*self.word_size])])

        l1_out = tf.matmul(input, self.W1) + self.b1
        l1_act = tf.nn.relu(l1_out)
        l2_out = tf.matmul(l1_act, self.W2) + self.b2
        l2_act = tf.nn.relu(l2_out)

        #eta = interface vec size
        #Y = output size
        self.nn_out = tf.matmul(l2_act, self.nn_out_weights) #(1*eta+Y, eta+Y*Y)->(1*Y)
        self.interface_vec = tf.matmul(l2_act, self.interface_weights) #(1*eta+Y, eta+Y*eta)->(1*eta)

        self.interface_vec = tf.slice(l2_act, [0, self.output_size], [-1])

        partition = [0]*(self.num_heads*self.word_size) + [1]*(self.num_heads) + [2]*(self.word_size) + [3] + \
                    [4]*(self.word_size) + [5]*(self.word_size) + \
                    [6]*(self.num_heads) + [7] + [8] + [9]*(self.num_heads*3)

        (read_keys, read_str, write_key, write_str,
         erase_vec, write_vec, free_gates, alloc_gate, write_gate, read_modes) = \
            tf.dynamic_partition(self.interface_vec, partition, 10)

        read_keys = tf.reshape(read_keys,[self.num_heads, self.word_size]) #R*W
        read_str = 1 + tf.nn.softplus(tf.expand_dims(read_str, 0)) #1*R
        write_key = tf.expand_dims(write_key, 0) #1*W
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

        self.mem_mat = self.mem_mat*(1-tf.matmul(self.write_weights, erase_vec)) + \
                       tf.matmul(self.write_weights, write_vec)

        nnweight_vec = tf.matmul(self.write_weights,tf.ones([1,self.num_words])) #N*N
        self.link_mat = (1-nnweight_vec-tf.transpose(nnweight_vec))*self.link_mat + \
                        tf.matmul(self.write_weights, tf.transpose(self.precedence_weight))

        self.precedence_weight = (1-tf.reduce_sum(self.write_weights, reduction_indices=0)) * \
                                 self.precedence_weight + self.write_weights

        forw_w = read_modes[2]*tf.matmul(self.link_mat, self.read_weights) #(N*N,N*R)->N*R
        look_w = read_modes[1]*self.content_lookup(read_keys, read_str) #N*R
        back_w = read_modes[0]*tf.matmul(self.link_mat, self.read_weights, transpose_b=True) #N*R

        self.read_weights = back_w + look_w + forw_w #N*R
        self.read_vecs = tf.transpose(tf.matmul(self.mem_mat, self.read_weights, transpose_a=True)) #(W*N,N*R)^T->R*W

        read_vec_mut = tf.matmul(tf.reshape(self.read_vecs, [1, self.num_heads * self.word_size]),
                                 self.read_vecs_out_weight)  # (1*RW, RW*Y)-> (1*Y)
        return self.nn_out+read_vec_mut

def main(argv=None):
    num_seq = 10
    seq_len = 5
    seq_width = 5
    seq = np.random.randint(2, size=seq_len*seq_width).reshape([seq_len, seq_width])
    end = np.asarray([[0]*(seq_width-1)+[1]])
    zer = np.zeros((seq_len-1, seq_width))

    dnc = DNC(input_size=5, output_size=5, num_words=8, word_size=4, num_heads=1)
    output = dnc.step_m(dnc.i_data)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, dnc.o_data))
    regularizers = (tf.nn.l2_loss(dnc.W1) + tf.nn.l2_loss(dnc.W2) +
                    tf.nn.l2_loss(dnc.b1) + tf.nn.l2_loss(dnc.b2))
    loss += 5e-4 * regularizers
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

    with tf.Session() as sess:
        final_i_data = np.concatenate((seq, end, zer), axis=0)
        final_o_data = np.concatenate((np.zeros((seq_len, seq_width)), seq), axis=0)
        tf.initialize_all_variables().run()
        print('Initialized!')
        feed_dict = {dnc.i_data : final_i_data, dnc.o_data : final_o_data}
        l, _, predictions = sess.run([loss, optimizer, output],feed_dict=feed_dict)

if __name__ == '__main__':
    tf.app.run()
