import numpy as np
from scipy.stats import entropy
from helper_functions import convert_assignment_to_clusters

def f(x): # for entropy
    return x*(-np.log2(x))

def g(x): # for entropy
    return -np.log2(x)

def f_gini(x):
    return x*(1-x)
def g_gini(x):
    return 1-x

def partition_entropy_rg(assignment, data, k):
    # weighted entropy in Cicalese et. al. 2019
    m, n = data.shape
    clusters = convert_assignment_to_clusters(assignment, data)
    ans = 0
    for cluster in clusters:
        if len(cluster) != 0:
            sum_cluster = np.sum(cluster)
            cluster = np.sum(cluster, axis = 0)
            ans -= np.sum(cluster*np.log2(cluster/sum_cluster))
    return ans

def partition_entropy(assignment, data, converted = False):
    m, n = data.shape
    if not converted:
        clusters = convert_assignment_to_clusters(assignment, data)
    else:
        clusters = assignment
    eps = 1e-15
    ans = 0
    e = 0

    for cluster in clusters:
        if len(cluster) != 0:
            p_z = np.sum(cluster)
            p_x_z_joint = np.sum(cluster, axis = 0)
            p_x_given_z = p_x_z_joint/p_z
            e += p_z*np.max(p_x_given_z)
            ans += p_z*entropy(p_x_given_z, base = 2)
            
    U = f(e) + (n-1) *f((1-e)/(n-1)) # upperbound for e
    L = g(e) # lowerbound for e

    return ans, e, U/L

def gini(assignment, data, k):
    m, n = data.shape

    clusters = convert_assignment_to_clusters(assignment, data)
    ans = 0
    e = 0

    for cluster in clusters:
        if len(cluster) != 0:
            p_z = np.sum(cluster)
            p_x_z_joint = np.sum(cluster, axis = 0)
            p_x_given_z = p_x_z_joint/p_z
            e += p_z*np.max(p_x_given_z)

            ans += p_z*np.sum(p_x_given_z*(1-p_x_given_z))
            # e += p_z*np.max(p_x_given_z)
            # ans += p_z*entropy(p_x_given_z, base = 2)
        
    U = f_gini(e) + (n-1) *f_gini((1-e)/(n-1)) # upperbound for e
    L = g_gini(e) # lowerbound for e

    return ans, e, U/L