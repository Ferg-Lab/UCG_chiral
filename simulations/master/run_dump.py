import os
import numpy as np
import pandas as pd
import pickle
from scipy.interpolate import CubicSpline
from typing import List, Dict, Tuple, Optional


from process_traj import *


# ============================================================
if __name__ == "__main__":


    kt_reduced = TTTT

    ref_dump_path = '/project2/andrewferguson/sivadasetty/doe/ucg/chiral_tetramer/SystemSize1000tetramers/T-'+str(kt_reduced)+'_freqdouble_racemic/dump_rerun_files/ucg_traj_reformat_joinfrag_vmd_fixed_forces_binary2_chirality.lammpstrj'

    aa_traj = dump_to_batch(ref_dump_path, True)

    #for i in range(4):
    #    print(f'i {i} \n')
    #    print(aa_traj[i])
    #    print('\n')
    #print(len(aa_traj))#[:6])
    #print(len([x for i,x in enumerate(aa_traj) if i % 4 < 2])) #[:4])

    #print([x for i, x in enumerate(aa_traj) if (i % 8) < 2])
    #print(len([x for i, x in enumerate(aa_traj) if (i % 8) < 2]))

    #print([x for i, x in enumerate(aa_traj) if (i % 10) < 2])
    print(len([x for i, x in enumerate(aa_traj) if (i % 10) < 2]))

    print('END')
    print(len(aa_traj))


