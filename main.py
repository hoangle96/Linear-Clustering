import numpy as np 
import entropy

from pathlib import Path
from helper_functions import convert_assignment_to_clusters as convert_cl

from dhillion import divisive_cluster
import argparse 
from all_algos import algo1, algo2, algo3
import time
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


if __name__ == "__main__":
    # pass argument and checking
    parser = argparse.ArgumentParser(description='Argument for R(e_max) algorithm')
    parser.add_argument('-filename', 
                        type=str, 
                        help='Input filename (without extension)')
    parser.add_argument('-K', type = int, 
                        help = "Number of clusters")

    # parse arg
    args = parser.parse_args()
    filename = Path(args.filename+".csv")

    data = np.genfromtxt(filename, delimiter=',')
    m, n = data.shape
    k = args.K
    assert()
    # nguyen 2021    
    nguyen_begin = time.time()
    assnment = algo1(data)
    partitions = convert_cl(assnment, data)

    if len(partitions) == k:
        clusters = partitions
    elif len(partitions) > k:
        clusters = algo2(len(partitions), k, partitions)
    else:
        clusters = algo3(len(partitions), k, partitions)
    nguyen_end = time.time()

    nguyen_time = nguyen_end - nguyen_begin
    impurity_nguyen, e, R_e = entropy.partition_entropy(partitions, data, converted = True)

    # dhillion et al
    s = np.random.randint(low=1, high=100, size=1)[0]
    dhillion_begin = time.time()
    dc_dhillion = divisive_cluster(init_type = "dhillion", data = data, k = k, seed = s)
    dc_dhillion.k_means()
    dhillion_end = time.time()
    dhillion_time = dhillion_end - dhillion_begin
    impurity_dhillion = dc_dhillion.get_impurity()

    print("Dimenions of data: m = %d, n = %d. Partitioning into %d clusters" %(m, n, k))
    print("Nguyen et. al.: result R(e): {R_e:.2f}, e: {e:.2f}, entropy: {impurity:.2f}, taking {time:.2f}(s)".format(R_e = R_e, e = e, impurity = impurity_nguyen, time = nguyen_time))
    print("Dhillion et. al.: result entropy: {impurity:.2f}, taking {time:.2f}(s)".format(impurity = impurity_dhillion, time = dhillion_time))

    