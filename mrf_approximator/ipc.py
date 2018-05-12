# -*- coding: utf-8 -*-

import subprocess, os, fcntl, time
import pandas as pd
import numpy as np

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
    #print("cmd =", cmd)
    subprocess.call(cmd, shell=True)
    
def posterior2ratio(y, var):
    df = pd.DataFrame({"point_id": list(range(len(y))), "mu": y, "sigma": np.sqrt(var)})
    os.makedirs("./temp", exist_ok=True)
    file_lock = open("./temp/.lock", 'w')
    while True:
        try:
            fcntl.lockf(file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            df[["point_id", "mu", "sigma"]].to_csv("./temp/musigma.csv", index=False)
            break
        except IOError:
            time.sleep(0.1)
    
    _call_mu2ratio("./mu2ratio/mu2ratio.jar", "./temp/musigma.csv", "./temp/weights.csv")
    df_weight = pd.read_csv("./temp/weights.csv", header=None).sort_values(0)
    weights = np.array(df_weight[1])
    fcntl.lockf(file_lock, fcntl.LOCK_UN)
    file_lock.close()
    return weights # return a numpy array