# -*- coding: utf-8 -*-

import subprocess, os, fcntl, time
import pandas as pd
import numpy as np

# helper to call mu2ratio.jar to calculate improper integrals
def _call_mu2ratio(jar_exec_path, csv_in, csv_weight_out):
    """
    csv_in format example:
    point_id	mu	sigma
    0	-9	10.1
    1	-0.6147069300775537	0.6401610934510453
    2	-3	0.47035284682323586
    3	-7	20.1
    """
    cmd = "java -jar {} {} {}".format(jar_exec_path, csv_in, csv_weight_out)
    """
    csv_weight_out format example:
    ...
    68  0.0017814966603510842
    69  0.004083630047757645
    70  0.00808761697203964
    71  0.014127926622452684
    72  0.02518025135396476
    73  0.059555390796960664
    74  0.2709186035482364
    75  0.08492610637092239
    76  0.03475975757360851
    77  0.01814366348544336
    78  0.010393708082240622
    79  0.00568141349210741
    80  0.002865444415898595
    ...
    """
    subprocess.call(cmd, shell=True)
    
def posterior2ratio(y, var):
    df = pd.DataFrame({"point_id": list(range(len(y))), "mu": y, "sigma": np.sqrt(var)})
    os.makedirs(os.path.join(os.getcwd(), "temp"), exist_ok=True)
    # protect the temp dir in case other testclicks processes are running
    file_lock = open(os.path.join(os.getcwd(), "temp", ".lock"), 'w')
    while True:
        try:
            fcntl.lockf(file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            df[["point_id", "mu", "sigma"]].to_csv(os.path.join(os.getcwd(), "temp", "musigma.csv"), index=False)
            break
        except IOError:
            time.sleep(0.1)
    
    _call_mu2ratio(os.path.join(os.getcwd(), "mu2ratio", "mu2ratio.jar"),
                   os.path.join(os.getcwd(), "temp", "musigma.csv"),
                   os.path.join(os.getcwd(), "temp", "weights.csv"))

    df_weight = pd.read_csv(os.path.join(os.getcwd(), "temp", "weights.csv"), header=None).sort_values(0)
    weights = np.array(df_weight[1])
    fcntl.lockf(file_lock, fcntl.LOCK_UN)
    file_lock.close()
    return weights  # return a numpy array
