### incorporating multiple tables
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits, load_iris
from sklearn.metrics import accuracy_score
import time


class lsh:
    def __init__(self, hash_size, data_dim, num_tables, random_type=None):
        self.num_rand_vec = hash_size  # number of buckets will be 2**hash_size eg: 2**2=4 (00,01,10,11)
        self.dim = data_dim
        self.num_tables = num_tables
        self.hash_tables = [{} for _ in range(self.num_tables)]
        self.seeds = [i for i in range(self.num_tables)]
        self.random_vectors = []
        for seed in self.seeds:
            np.random.seed(seed)
            if random_type:
                self.random_vectors.append(self.gen_random_vectors(random_type='normal_gpu'))
            else:  # normal distribution by default
                self.random_vectors.append(self.gen_random_vectors())
        print('TENSORFLOW GPU $$$$$$$$$$$$$ available GPU is -->>,', tf.test.gpu_device_name())

    def gen_random_vectors(self, random_type=None):
        if random_type is None:
            # sample from random_normal distribution by default
            return np.random.randn(self.num_rand_vec, self.dim)
        if random_type == 'normal_gpu':
            return tf.random.normal((self.num_rand_vec, self.dim)).numpy()
        if random_type == 'uniform':
            return np.random.rand(self.num_rand_vec, self.dim)

    #     @tf.function
    def make_hash_key(self, inp):
        return ''.join(inp)

    def fit(self, data, label):
        assert data.shape[1] == self.dim, 'dimension of input data is {} and dimension in LSH object is {}'.format(
            data.shape, self.dim)
        # sess = tf.Session()
        # sess = tf.compat.v1.InteractiveSession()
        print('fitting')
        gpu_availability = tf.test.is_gpu_available()
        print('is GPU availabale ->', gpu_availability)
        #         if gpu_availability:
        session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
        #         with tf.device("/GPU:0"):
        tf.compat.v1.disable_eager_execution()
        #         print('creating session on gpu, eager execution disabled ')
        data_ph = tf.compat.v1.placeholder("float", [None, self.dim])
        rand_hash_vec_ph = tf.compat.v1.placeholder("float", [None, self.dim])
        distance_matrix_ = tf.matmul(data_ph, tf.transpose(rand_hash_vec_ph))
        euclidean_dist_ = tf.sqrt(tf.reduce_sum(distance_matrix_ ** 2, axis=1))
        # data_ph = tf.convert_to_tensor(data,dtype=tf.float32)
        distance_as_keys = tf.strings.as_string(tf.cast((distance_matrix_ > 0), tf.int8))
        keys_ = tf.strings.reduce_join(distance_as_keys, axis=1)  # [b'1001101',b'11001011',....]
        unique_keys_, idx = tf.unique(keys_)
        sorted_distance_idx = tf.argsort(euclidean_dist_)
        #         (distance, bucket_key, data)

        #         tf.range(0,self.num_tables)
        for rand_vec, hash_table in zip(self.random_vectors, self.hash_tables):
            # rand_hash_vec_ph = rand_vec
            # distance_matrix_ = tf.matmul(data_ph, tf.transpose(rand_hash_vec_ph))
            # euclidean_dist_ = tf.sqrt(tf.reduce_sum(distance_matrix_ ** 2, axis=1))
            t1 = time.time()
            distance_matrix, euclidean_dist, keys, unique_keys, sorted_dist_idx = session.run(
                [distance_matrix_, euclidean_dist_, keys_,
                 unique_keys_, sorted_distance_idx],
                feed_dict={data_ph: data,
                           rand_hash_vec_ph: rand_vec})
            print(type(distance_matrix), 'type(distance_matrix)', distance_matrix.shape)
            print('total keys', len(keys))
            print('unique_keys', len(unique_keys))
            print('time taken for matmul ', time.time() - t1)
            print(sorted_dist_idx, len(sorted_dist_idx))
            # distance_matrix, euclidean_dist = np.array(distance_matrix_), np.array(euclidean_dist_)
            #             keys = list(map(self.make_hash_key, (distance_matrix > 0).astype('int').astype('str')))
            # print('euclidean dist', (distance_matrix > 0).astype('int'))

            # the keys contain string of length=hash_size (2 bits or 3 bits..) for each document.
            # Eg '00','01','10','11'  for hash_size of 2 bits.
            #             unique_keys = set(keys)
            for key in unique_keys:
                hash_table[key] = []

            assert len(keys) == len(euclidean_dist), 'shape of euclidean dist matrix is {} and length of keys list' \
                                                     ' is {}. They must match'.format(len(euclidean_dist), len(keys))
            #             sorted_keys = [(dist,key,key_idx) for dist, key, key_idx in
            #                            sorted(zip(list(euclidean_dist),keys,range(len(keys))),  key=lambda pair: pair[0])]
            # key_idx represents the document index in original data. We need to preserve this info before sorting
            # the points in one bucket based on their distances from randomly projected vectors.
            for idx in sorted_dist_idx:
                #                 print(keys[idx],euclidean_dist[idx], data[idx])
                hash_table[keys[idx]].append((euclidean_dist[idx], data[idx], label[idx]))

        #             for key in sorted_keys:  # (dist,key,key_idx)
        #                 hash_table[key[1]].append((key[0],key[2]))  # (distance, doc_idx) <<<<<<<< very important
        # each key:value pair in hash table looks like this
        # '01':[(euclidean_dist, document_idx_in_data), () ,() , ......]
        # '01' is a bucket, in which a document might be present in.
        #         else:
        #             print('fitting skipped')
        #             print(hash_table.keys())
        return 'success'

    def sort_buckets_elements(self):
        """
        call this after function after dividing the data into several buckets.
        Idea is to sort the bucket elements based on distance from random projection vector. Given a query point,
        we can find the top k similar elements by doing exhaustive search in the bucket. But if the bucket elements
        are sorted we can find top k similar items without having to compare all the
        elements in a particular bucket. This will speedup query.
        """
        pass

    def hash_table(self):
        return self.hash_tables

    def hash_table_dist(self):
        distributions = []
        for hash_table in self.hash_tables:
            summary_of_table = {}
            for k, v in hash_table.items():
                summary_of_table[k] = len(v)
            distributions.append(summary_of_table)
        return distributions

    def query(self, query_data):
        key_for_each_table = []
        for rand_vec in self.random_vectors:
            key = ''.join((np.dot(query_data, rand_vec.T) > 0).astype('int').astype('str'))
            key_for_each_table.append(key)  # each point will be assigned to exactly on bucket in one hash table
        result = []
        assert len(key_for_each_table) == len(
            self.hash_tables), 'Some data point is not assigned to any bucket in a hash table'
        for hash_table, key in zip(self.hash_tables, key_for_each_table):
            if key in hash_table.keys():
                result.extend(hash_table[key])
        #         print(keys_for_each_table)
        return list(set(result))

    def fast_query(self, query_data):
        if len(query_data.shape) == 1:
            query_data = query_data.reshape((1, -1))
        euc_dist_with_key_for_each_table = []
        for rand_vec in self.random_vectors:
            distance_matrix = np.dot(query_data, rand_vec.T)
            euclidean_dist = np.sqrt(np.sum(distance_matrix ** 2, axis=1))
            key = list(map(self.make_hash_key, (distance_matrix > 0).astype('int').astype('str')))
            #             print(key[0].encode('utf-8'),'key<<')
            euc_dist_with_key_for_each_table.append((euclidean_dist[0], key[0].encode('utf-8')))
            # each point will be assigned to exactly on bucket in one hash table.
            # euc_dist_with_key_for_each_table is a list of tuple-->
            # [(key in hash table, euclidean distance from rand vector), (), ...]
        result = []
        assert len(euc_dist_with_key_for_each_table) == len(
            self.hash_tables), 'Some data point is not assigned to any bucket in a hash table'
        for hash_table, distance_key_tuple, in zip(self.hash_tables, euc_dist_with_key_for_each_table):
            # print('<<<<<<<<<<<',euc_dist_with_key_for_each_table)
            distance = distance_key_tuple[0]
            # print('query_distance', distance)
            key = distance_key_tuple[1]
            if key in hash_table.keys():
                # now use the key( or bucket) of each hash table, along with euclidean distance from random projection
                # vectors to find the most similar items to the query doc, instead of exhaustive search.
                # Using binary search on distances.
                bucket_elements = hash_table[key]
                # print('bucket_elements', bucket_elements)
                candidates = self.binary_search(arr=bucket_elements, query_distance=distance, max_k=5)
                # print('candidates',candidates)
                result.extend(candidates)
        result_docs, labels = np.vstack([tupl[1] for tupl in result]), [tupl[2] for tupl in result]
        #         print(result_docs,labels,',,,,,,,')
        return result_docs, labels

    #     def fast_query(self, query_data):
    #         if len(query_data.shape) == 1:
    #             query_data = query_data.reshape((1,-1))
    #         euc_dist_with_key_for_each_table = []
    #         session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    #         tf.compat.v1.disable_eager_execution()
    #         data_ph = tf.compat.v1.placeholder("float", [None, self.dim])
    #         rand_hash_vec_ph = tf.compat.v1.placeholder("float", [None, self.dim])
    #         distance_matrix_ = tf.matmul(data_ph, tf.transpose(rand_hash_vec_ph))
    #         euclidean_dist_ = tf.sqrt(tf.reduce_sum(distance_matrix_ ** 2, axis=1))
    #         # data_ph = tf.convert_to_tensor(data,dtype=tf.float32)
    #         distance_as_keys = tf.strings.as_string(tf.cast((distance_matrix_ > 0),tf.int8))
    #         keys_ = tf.strings.reduce_join(distance_as_keys,axis=1) # [b'1001101',b'11001011',....]
    # #         tf.while_loop(i>0)
    #         for rand_vec in self.random_vectors:
    #             distance_matrix,euclidean_dist,keys = session.run([distance_matrix_, euclidean_dist_,keys_],
    #                                                       feed_dict={data_ph: query_data,
    #                                                                  rand_hash_vec_ph: rand_vec})
    # #             distance_matrix = np.dot(query_data, rand_vec.T)
    # #             euclidean_dist = np.sqrt(np.sum(distance_matrix ** 2, axis=1))
    # #             key = list(map(self.make_hash_key, (distance_matrix > 0).astype('int').astype('str')))
    #             print(type(distance_matrix),'type(distance_matrix)',distance_matrix.shape)
    #             print('total keys',len(keys))
    #             euc_dist_with_key_for_each_table.append((euclidean_dist,keys))
    #             # each point will be assigned to exactly on bucket in one hash table.
    #             # euc_dist_with_key_for_each_table is a list of tuple-->
    #             # [(key in hash table, euclidean distance from rand vector), (), ...]
    # #         print(euc_dist_with_key_for_each_table,'euc_dist_with_key_for_each_table', len(euc_dist_with_key_for_each_table))
    # #         print(len(euc_dist_with_key_for_each_table[0]))
    #         result = []
    #         assert len(euc_dist_with_key_for_each_table) == len(self.hash_tables), 'Some data point is not assigned to any bucket in a hash table'
    #         for hash_table, distance_key_tuple, in zip(self.hash_tables, euc_dist_with_key_for_each_table):
    #             # print('<<<<<<<<<<<',euc_dist_with_key_for_each_table)

    #             for dist_key in distance_key_tuple:
    #                 print('dist_key', dist_key)
    #                 key = dist_key[1]
    #                 distance = dist_key[0]
    #                 print('key and bucket', key, distance)
    #                 if key in hash_table.keys():
    #                     # now use the key( or bucket) of each hash table, along with euclidean distance from random projection
    #                     # vectors to find the most similar items to the query doc, instead of exhaustive search.
    #                     # Using binary search on distances.
    #                     bucket_elements = hash_table[key]
    #                     # print('bucket_elements', bucket_elements)
    #                     candidates = self.binary_search(arr=bucket_elements,query_distance=distance,max_k=5)
    #                     # print('candidates',candidates)
    #                     result.extend(candidates)
    # #         print(len(result),'this much candidates')
    # #         result_doc_idexes = [tupl[1] for tupl in result]
    #         return result

    def binary_search(self, arr, query_distance, max_k):
        mid = len(arr) // 2
        if arr[mid][0] > query_distance:
            arr = arr[:mid]
            if len(arr) <= max_k:
                return arr
            else:
                return self.binary_search(arr, query_distance, max_k)
        if arr[mid][0] < query_distance:
            arr = arr[mid:]
            if len(arr) <= max_k:
                return arr
            else:
                return self.binary_search(arr, query_distance, max_k)
