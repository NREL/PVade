import numpy as np
import glob
import sys
import time
from mpi4py import MPI


def get_net_traction(path_to_files):

    try:
        fp = open("%s/sim_log.txt" % (path_to_files), "r")
    except:
        raise ValueError("Cannot find %s" % (path_to_files))
    else:
        fp.close()

    csv_list = sorted(glob.glob("%s/time_series/*.csv" % (path_to_files)))

    data_out = []

    for csv in csv_list:
        raw_data = np.genfromtxt(csv, delimiter=",", skip_header=1)
        p_top = raw_data[:, 3]
        p_bot = raw_data[:, 7]

        data_out.append(p_top - p_bot)

    data_out = np.array(data_out)

    output_filename = "%s/net_traction" % (path_to_files)
    np.savetxt(output_filename + ".csv", data_out, delimiter=",")
    np.save(output_filename + ".npy", data_out)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_procs = comm.Get_size()

try:
    prefix = sys.argv[1]
except:
    prefix = "output/param_study"

run_list_global = sorted(glob.glob("%s/theta_*/uref_*" % (prefix)))
run_list_local = np.array_split(run_list_global, num_procs)[rank]

if rank == 0:
    for k in run_list_global:
        print(k)
    print("================================================================")
    print("Found %d runs to process" % (len(run_list_global)))
    print(
        "Dividing work between %d cores (~%d runs per core)"
        % (num_procs, len(run_list_local))
    )
    print("================================================================")

for path_to_files in run_list_local:
    tic = time.time()
    get_net_traction(path_to_files)
    toc = time.time()
    print(
        "Rank %d finished processing %s in %f seconds."
        % (rank, path_to_files, toc - tic)
    )
