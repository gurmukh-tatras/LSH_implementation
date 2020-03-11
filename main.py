import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from LSH import lsh as lsh_
import time
from sklearn.metrics.pairwise import cosine_similarity


def prepare_mnist_data(num_samples,test_size=0.2):
    print('loading data')
    mnist = pd.read_csv('mnist-in-csv/mnist_train.csv',nrows=num_samples)
    mnist_data = np.array(mnist[mnist.columns[1:]])
    mnist_labels = np.array(mnist[mnist.columns[0]])
    X_train, X_test, y_train, y_test = train_test_split(mnist_data, mnist_labels,test_size=test_size,random_state=11)
    # X_train = X_train / 255
    # X_test = X_test / 255
    return X_train, X_test, y_train, y_test


def exhaustive_search(query_doc,candidates):
    sims = cosine_similarity(candidates,query_doc)
    max_sim_idx = sims.argmax()
    return max_sim_idx


def candidate_label_distribution(candidates,y_train):
    candidate_label_dist = {i: 0 for i in range(10)}
    for doc_tuple in candidates:
        # candidates is list of tuple, each tuple contains distance from random projection vector and doc index
        candidate_label_dist[y_train[doc_tuple[1]]] += 1
    return candidate_label_dist


def main():
    X_train, X_test, y_train, y_test = prepare_mnist_data(num_samples=3000,test_size=0.1)
    lsh = lsh_(hash_size=10, data_dim=X_train.shape[1], num_tables=10)

    t1 = time.time()
    lsh.fit(X_train)
    print('total time taken for fitting', time.time() - t1)
    rand_doc = 18
    query = X_test[rand_doc]
    print('label of query doc ', y_test[rand_doc])

    t1 = time.time()
    candidates = (lsh.query(query))
    print('total time taken for prediction', time.time() - t1)
    print('number of candidates:', len(candidates))
    print('candidate label distribution:', candidate_label_distribution(candidates,y_train))

    # print('hash tables dist', lsh.hash_table_dist())
    t1 = time.time()
    fast_query_candidates = lsh.fast_query(query)
    print('total time taken for prediction using fast_query', time.time() - t1)
    candidate_label_dist = {i: 0 for i in range(10)}
    for doc_tuple in fast_query_candidates:
        # candidates is list of tuple, each tuple contains distance from random projection vector and doc index
        candidate_label_dist[y_train[doc_tuple]] += 1
    print('candidate label distribution:',candidate_label_dist )
    print('fast_query_candidates ',fast_query_candidates)
    print('fast_query_candidates length',len(fast_query_candidates))
    candidate_docs = X_train[fast_query_candidates]
    most_similar_doc_idx = exhaustive_search(query_doc=query.reshape(1,-1)/255, candidates=candidate_docs/255)
    print('label for the query is ', y_train[most_similar_doc_idx])


main()