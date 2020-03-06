### incorporating multiple tables
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits, load_iris
from sklearn.metrics import accuracy_score


class lsh:
    def __init__(self, hash_size, data_dim, num_tables):
        self.num_rand_vec = hash_size  # number of buckets will be 2**hash_size eg: 2**2=4 (00,01,10,11)
        self.dim = data_dim
        self.num_tables = num_tables
        self.hash_tables = [{} for _ in range(self.num_tables)]
        self.seeds = [i for i in range(self.num_tables)]
        self.random_vectors = []
        for seed in self.seeds:
            np.random.seed(seed)
            self.random_vectors.append(self.gen_random_vectors(random_type='normal'))
            # np.random.randint(low=0, high=255, size=(self.num_rand_vec, self.dim))
            #

    def gen_random_vectors(self,random_type=None):
        if random_type is None:
            return np.random.randn(self.num_rand_vec, self.dim)
        if random_type == 'normal':
            return np.random.randn(self.num_rand_vec, self.dim)
        if random_type == 'uniform':
            return np.random.rand(self.num_rand_vec, self.dim)

    def make_hash_key(self, inp):
        return ''.join(inp)

    def fit(self, data):
        assert data.shape[1] == self.dim, 'dimension of input data is {} and dimension in LSH object is {}'.format(
            data.shape, self.dim)
        for rand_vec, hash_table in zip(self.random_vectors, self.hash_tables):
            distance_matrix = np.dot(data, rand_vec.T)
            euclidean_dist = np.sqrt(np.sum(distance_matrix**2,axis=1))
            keys = list(map(self.make_hash_key, (distance_matrix > 0).astype('int').astype('str')))
            # the keys contain string of length=hash_size (2 bits or 3 bits..) for each document. Eg '00','01','10','11'
            # for hash_size of 2 bits.
            unique_keys = set(keys)
            for key in unique_keys:
                hash_table[key] = []
            assert len(keys) == len(euclidean_dist), 'shape of euclidean dist matrix is {} and length of keys list is {}.' \
                                                ' They must match'.format(len(euclidean_dist), len(keys))
            sorted_keys = [(dist,key,key_idx) for dist, key, key_idx in
                            sorted(zip(list(euclidean_dist),keys,range(len(keys))),  key=lambda pair: pair[0])]
            # key_idx represents the document index in original data. We need to preserve this info before sorting the
            # points in one bucket based on their distances from randomly projected vectors.
            for key in sorted_keys:  #(dist,key,key_idx)
                hash_table[key[1]].append((key[0],key[2]))
                # each key:value pair in hash table looks like this
                # '01':[(euclidean_dist, document_idx_in_data), () ,() , ......]
                # '01' is a bucket, in which a document might be present in.
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
        distribuitions = []
        for hash_table in self.hash_tables:
            summary_of_table = {}
            for k, v in hash_table.items():
                summary_of_table[k] = len(v)
            distribuitions.append(summary_of_table)
        return distribuitions

    def query(self, query_data):
        key_for_each_table = []
        for rand_vec in self.random_vectors:
            key = ''.join((np.dot(query_data, rand_vec.T) > 0).astype('int').astype('str'))
            key_for_each_table.append(key)  # each point will be assigned to exactly on bucket in one hash table
        result = []
        assert len(key_for_each_table) == len(self.hash_tables), 'Some data point is not assigned to any bucket in a hash table'
        for hash_table, key in zip(self.hash_tables, key_for_each_table):
            if key in hash_table.keys():
                result.extend(hash_table[key])
        #         print(keys_for_each_table)
        return list(set(result))


