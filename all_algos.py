import numpy as np
import helper_functions as h

def algo1(data): # k == n 
    assignment = np.argmax(data, axis = 1)
    return assignment

def algo2(N, K, clusters, impurity_func = 'entropy'): # K > N
    # greedy splitting
    new_clusters = [None]*K
    for i in range(len(clusters)):
        new_clusters[i] = clusters[i]

    t = 1
    while t <= K - N:
        # Finding the largest impurity.
        impurity_list = []
        splitable_list = [] # not singleton and have at least 2 different entries
        for i in range(N+t-1):
            impurity = h.cal_impurity(new_clusters[i], impurity_func)
            splitable_list.append(h.check_splitable(new_clusters[i]))
            impurity_list.append(impurity)
        not_splitable = np.logical_not(np.array(splitable_list))
        
        impurity_list = np.array(impurity_list)
        impurity_list[not_splitable] = 0
        idx = np.argmax(impurity_list)
        max_cluster = np.array(new_clusters[idx])

        # Splitting based on the largest attribution.
        p_x_given_z_max = np.nansum(max_cluster, axis = 0)/np.sum(max_cluster)
        max_attr_idx = np.nanargmax(p_x_given_z_max)
        p_x_star_z_star = np.nanmax(p_x_given_z_max)
        new_clusters[idx] = [y for y in max_cluster if y[max_attr_idx]/np.sum(y)- p_x_star_z_star<=0]
        new_clusters[N+t-1] = [y for y in max_cluster if y[max_attr_idx]/np.sum(y) - p_x_star_z_star > 0] 

        t += 1 
    return new_clusters

def algo3(N, K, clusters): # K < N
    # greedy merge
    t = 0
    while t < N - K:
        print(t)
        min_impurity_loss = np.infty
        min_i = 0
        min_j = 0
        for i in range(len(clusters)):
            # print(i)
            # for j in range(i+1, K-t+1):
            for j in range(i+1, len(clusters)):

                # print(i,j)
                new_cluster = clusters[i] + clusters[j]

                impurity = h.cal_impurity(new_cluster)
                impurity_loss = impurity - h.cal_impurity(clusters[i]) - h.cal_impurity(clusters[j])
                if impurity_loss <  min_impurity_loss:
                    min_impurity_loss = impurity_loss
                    min_i = i
                    min_j = j

        if min_impurity_loss != np.infty:
            clusters[min_i] = clusters[min_i] + clusters[min_j]
            clusters.pop(min_j)

        t += 1
    return clusters
