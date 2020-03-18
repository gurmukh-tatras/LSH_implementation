# LSH Implementation
This is the implementation of Random projection method to speed up 
nearest neighbour search. 

# Highlights of this implementation
We store elements in each bucket in each hash table in a sorted way.
Euclidean distance from randomly projected plane is taken as the key
based on which each elements in the bucket is sorted. 
While querying, we just find the top k items which are similar to
the query item based on it's distance from the randomly projected plane.
These planes are different for each hash table.

# Comparison with Sklearn KNN
- Training KNN model in sklearn on 27000 images and testing on 3000 images
gives 96% accuracy in 124 seconds.
- Training our LSH model on 27000 images and testing on 3000 images
gives 88.8% accuracy in 10.6 seconds (10 hash tables with bit size 12)
 and 93.5% accuracy in 32 seconds (30 hash tables with bit size 12)

To Do:
- Add support for multiprocessing to reduce fitting time
- Saving lsh object offline and loading it directly into memory for inference
- Comparison with sklearn implementation of KNN and LSH

#Experiemnts on mnist images:
- 8 bit size, 70 k docs, 1 hash tables, acc is 47.85%
  - TF fit in 1 sec prediction for 6k docs in 3.44 sec
- 8 bit size, 70 k docs, 2 hash tables, acc is 62.23%
  - TF fit in 2 sec prediction for 6k docs in 5.2 sec
- 8 bit size, 70 k docs, 5 hash tables, acc is 80.2%
  - TF fit in 4.9 sec prediction for 6k docs in 17.4 sec
- 8 bit size, 70 k docs, 10 hash tables, acc is 87.8%
  - TF fit in 10.1 sec, prediction for 6k docs in 27.3 sec
  - Normal fit in 9.2 sec, prediction for 6k  docs in 23.8 secs
- 12 bit size, 70 k docs, 200 hash tables, acc is 97.68%
  - TF fit in 236 sec prediction for 6k docs in 334.63 sec
  - Normal fit in 218.27 sec, prediction for 6k  docs in 316.61 secs

