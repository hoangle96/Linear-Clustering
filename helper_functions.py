import numpy as np
from scipy.stats import entropy

def convert_assignment_to_clusters(assignment, data):
    assignment_unique = np.sort(np.unique(assignment))
    partitions = [[] for _ in range(len(assignment_unique))]
    assignment = tuple(assignment)
    assignment_unique = tuple(assignment_unique)
    for i,v in enumerate(assignment):
        partitions[assignment_unique.index(v)].append(data[i])
    return partitions

def normalize(a): 
    # normalize matrix so sum of each row is 1
    row_sums = a.sum(axis = 1)
    return(a / row_sums[:, np.newaxis])

def roundup(x):
    return x if x % 10 == 0 else x + 10 - x % 10

def cal_impurity(cluster, impurity = 'gini'):
    p_z = np.sum(cluster)
    p_x_z_joint = np.sum(cluster, axis = 0)
    p_x_given_z = p_x_z_joint/p_z

    if impurity == 'gini':
        ans = p_z*np.sum(p_x_given_z*(1-p_x_given_z))
    elif impurity == 'entropy':
        ans = p_z*entropy(p_x_given_z, base = 2)
    return ans 

def check_splitable(cluster):
    mean_ = np.mean(cluster, axis = 0)
    not_singleton = len(cluster) > 1 
    all_dups = np.all(np.isclose(cluster, mean_))
    res = not_singleton and not all_dups
    return res

def get_total_len_of_all_partition(clusters):
    sum_ = 0
    for cluster in clusters:
        sum_ += len(cluster)
    return sum_