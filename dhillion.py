import numpy as np
from collections import defaultdict
from helper_functions import convert_assignment_to_clusters, normalize
from scipy.stats import entropy
from entropy import partition_entropy, partition_entropy_rg


class divisive_cluster(object):
    def __init__(self, init_type = "nguyen", k = 0, data = None, assignment = None, seed = None):
        # data is always the joint distribution with rows being words and columns being classes
        # However, in this paper, all the data points are from conditional probability distribution
        # In the paper l is the number of document classes, here we will use n to denote it
        # marginalize P(C)
        if not seed is None:
            np.random.seed(seed)

        self.data = data 

        self.init_type = init_type
        self.k = k

        self.entropy = 0
        self.impurity = 0

        if init_type == "nguyen":
            """Initialize DC with the result of Nguyen 2020"""
            if not (assignment is None):
                self.assignment = assignment
                self.clusters = np.asarray(convert_assignment_to_clusters(self.assignment, data))
                self.k_means(threshold = 1e-30, n_iters = 1)
                # self.entropy = partition_entropy_rg(self.assignment, self.data, self.k)
                self.impurity, _, _ = partition_entropy(self.assignment, self.data, self.k, converted = False)
            else:
                assert not (assignment is None)
        else:
            """Initialize DC following the original paper"""
            # initial assignment p(c_j|w_t) = maxi p(c_i|w_t), k = n
            # self.assignment = np.argmax(data, axis = 1)
            # self.assignment = self.argmax_randtie_masking_generic(data, axis = 1)
            self.assignment = self.argmax_randtie_masking_generic(data, axis = 1)
            clusters = convert_assignment_to_clusters(self.assignment, self.data)
            self.clusters = clusters

            _, n = data.shape
            if k > n:
                # split each cluster arbitrarily into at least floor(k/l) clusters
                n_to_split = k//n
                new_clusters = []
                for cluster in clusters:
                    if len(cluster) > n_to_split:
                        splited_arrs = np.array_split(np.array(cluster), n_to_split)
                        new_clusters += splited_arrs
                while len(new_clusters) < k:
                    len_list = [len(new_clusters[i]) for i in range(len(new_clusters))]   
                    max_idx = np.argmax(len_list) 
                    splited_arrs = np.array_split(np.array(cluster), n_to_split)
                    new_clusters += splited_arrs
                self.clusters = new_clusters
            elif k < n:
                for i in range(k, len(clusters)):
                    clusters[k-1] += clusters[i]
                self.clusters = clusters[:k]
            self.clusters = np.asarray(self.clusters)
            impurity = partition_entropy_rg(self.assignment, self.data, self.k)
          

    
    def random_num_per_grp_cumsumed(self, L):
        # For each element in L pick a random number within range specified by it
        # The final output would be a cumsumed one for use with indexing, etc.
        r1 = np.random.rand(np.sum(L)) + np.repeat(np.arange(len(L)),L)
        offset = np.r_[0,np.cumsum(L[:-1])]
        return r1.argsort()[offset]

    def argmax_randtie_masking_generic(self, a, axis=1): 
        max_mask = a==a.max(axis=axis,keepdims=True)
        m,n = a.shape
        L = max_mask.sum(axis=axis)
        set_mask = np.zeros(L.sum(), dtype=bool)
        select_idx = self.random_num_per_grp_cumsumed(L)
        set_mask[select_idx] = True
        if axis==0:
            max_mask.T[max_mask.T] = set_mask
        else:
            max_mask[max_mask] = set_mask
        return max_mask.argmax(axis=axis) 

    def cal_cluster(self, cluster):
        # calculate new "centroids"
        cluster = np.array(cluster)
        pi_cluster = np.sum(cluster) # sum of all priors
        p_class_given_cluster_j = np.sum(cluster, axis = 0)/pi_cluster
        p_class_given_cluster_j = p_class_given_cluster_j.reshape((p_class_given_cluster_j.shape[0],-1))
        return p_class_given_cluster_j.T

    # equation 13
    def cal_q(self, clusters, data):
        q = 0
        m, n = data.shape
        for cluster in clusters:
            cluster = np.array(cluster)

            # p_class_given_word = cluster / m
            p_class_given_cluster = self.cal_cluster(cluster)
            p_word = np.sum(cluster, axis = 1)
            kl_div = self.cal_kl_div_from_pts_to_centroid(cluster, p_class_given_cluster)
            q += np.sum(kl_div*p_word)
        return q

    def cal_kl_div_from_pt_to_centroids(self, data_pt, centroids, norm = True):
        centroids = normalize(centroids)        
        data_pt = data_pt/np.sum(data_pt)
        return np.sum(data_pt*(np.log2(data_pt/centroids)), axis = 1)

    def cal_kl_div_from_pts_to_centroid(self, data_pts, centroid, norm= True):
        data_pts = normalize(data_pts)
        centroid = centroid/np.sum(centroid)
        return np.sum(data_pts*(np.log2(data_pts/centroid)), axis = 1)

    def k_means(self, threshold = 1e-30, n_iters = 50):
        # step 2
        m = len(self.data)
        self.pi_cluster = [None]*self.clusters.shape[0]
        prev_q = 0
        # prev_assn = np.zeros((1, m))
        for iter_ in range(n_iters):
            # print(iter_)
            # expectation
            q = self.cal_q(self.clusters, self.data)
            # print(q)
            diff = q - prev_q
            prev_q = q
            if np.abs(diff) > threshold:
            # converge = np.array_equal(prev_assn, self.assignment)
            # if not converge:
            # for each cluster calculate its "centroid", denoted as p(C|W_j) in the paper
            # then calculate the distance of the current cluster W_j to each point.
            # If the distance between any point to the current cluster is smaller than the distance to the previous cluster
            # then change the assignment of that point to the current cluster
                for idx, cluster in enumerate(self.clusters):
                    if len(cluster) != 0:
                        p_c_given_cluster_j = self.cal_cluster(np.array(cluster))
                        self.pi_cluster[idx] = p_c_given_cluster_j
                self.pi_cluster = np.asarray(self.pi_cluster)
                self.pi_cluster = self.pi_cluster.reshape((self.pi_cluster.shape[0], -1))

                prev_assn = np.copy(self.assignment)
                # maximization step
                for idx, d in enumerate(self.data):
                    d = d.T
                    kl_div = self.cal_kl_div_from_pt_to_centroids(d, np.array(self.pi_cluster))
                    new_assn = np.argmin(kl_div)
                    self.assignment[idx] = new_assn
                self.clusters = convert_assignment_to_clusters(self.assignment, self.data)

                self.impurity, e, r_max = partition_entropy(self.assignment, self.data, converted = False)
            else: 
                break
    
    def get_impurity(self):
        return self.impurity