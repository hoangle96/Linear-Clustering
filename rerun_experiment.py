import numpy as np 
import entropy
import pandas as pd 

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
                        help='Input filename')

    # parse arg
    args = parser.parse_args()
    filename = Path(args.filename+".csv")
    # filename = Path("ng20.csv")

    data = np.genfromtxt(filename, delimiter=',')
    m, n = data.shape
    # k = args.K
    print("Dimenions of data: %d, %d" %(m, n))

    k_list = list(range(2,20)) + list(range(20, 51, 10)) + [100,200,500,1000,2000]
    
    e_list = []
    R_e_list = []
    nguyen_impurity_list = []
    nguyen_time_list = []

    dhllion_impurity_list = []
    dhllion_time_list = []

    for k in k_list:
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

        e_list.append(e)
        R_e_list.append(R_e)
        nguyen_impurity_list.append(impurity_nguyen)
        nguyen_time_list.append(nguyen_time)

        # dhillion et al
        s = np.random.randint(low=1, high=100, size=1)[0]
        dhillion_begin = time.time()
        dc_dhillion = divisive_cluster(init_type = "dhillion", data = data, k = k, seed = s)
        dc_dhillion.k_means()
        dhillion_end = time.time()
        dhillion_time = dhillion_end - dhillion_begin
        impurity_dhillion = dc_dhillion.get_impurity()

        dhllion_impurity_list.append(impurity_dhillion)
        dhllion_time_list.append(dhillion_time)

        print("K = ", k)
        print("Nguyen et. al. result R(e): {R_e:.2f}, e: {e:.2f}, entropy: {impurity:.2f}, taking {time:.2f}(s)".format(R_e = R_e, e = e, impurity = impurity_nguyen, time = nguyen_time))
        print("Dhillion et. al. result entropy: {impurity:.2f}, taking {time:.2f}(s)".format(impurity = impurity_dhillion, time = dhillion_time))

    df = pd.DataFrame(data = {'k':k_list, 'e': e_list, 'R(e)':R_e_list, \
                'entropy_nguyen': nguyen_impurity_list, "nguyen_time":nguyen_time_list,\
                'dhllion_nguyen': dhllion_impurity_list, "dhllion_time":dhllion_time_list})
    name = args.filename+"_entropies.csv"
    df.to_csv(name, index = False)