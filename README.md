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


To Do:
- Add support for multiprocessing to reduce fitting time
- Saving lsh object offline and loading it directly into memory for inference
- Comparison with sklearn implementation of KNN and LSH

 