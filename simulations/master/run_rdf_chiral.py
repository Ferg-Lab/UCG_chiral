import os
import numpy as np
import pandas as pd
import MDAnalysis as mda # NOTE -- version is 2.4 - not for processing traj (doesn't support nojump)
import MDAnalysis.transformations as tf
from MDAnalysis.analysis import distances
from MDAnalysis.lib import distances as lib_distances
import MDAnalysis.analysis.rms as rms
import pickle
import sys
sys.path.append('/project2/andrewferguson/sivadasetty/doe/ucg/notebooks/chiral/scripts/')
from analysis_scripts import *


################################################################################################
# Read trajectory
################################################################################################
for iter in np.arange(0,2000): #62,85):

    trajpath =  'iter_' + str(iter) + '/'

    if not os.path.exists(trajpath  + '/UCG_chiral.lammpstrj'):
        continue

    rdf_results_file_max_6 = trajpath + 'nosymmetry_rdf_halfbox_maxbound_8_nbinary.pkl'
    symm_rdf_results_file_max_6 = trajpath + 'symmetry_rdf_halfbox_maxbound_8_nbinary.pkl'
    count_results_file_max6_filename = trajpath + 'count_ld.pkl'

    if os.path.exists(count_results_file_max6_filename):
        continue


    ucg_file_tb_traj_trr = trajpath + 'UCG_chiral.lammpstrj'
    ucg_chirality_traj = read_lammps_dump_xyz(ucg_file_tb_traj_trr) # READ ONLY ONCE

    #################################################################################################
    # RDF
    ################################################################################################
    box_bounds = [2.0870640000000002e+01, 2.0870640000000002e+01, 2.0870640000000002e+01, 90, 90, 90]
    bins = 50
    ucg_comRDF_max_6 = target_compute_rdf(ucg_chirality_traj, box_bounds, bins, start=1, stop=-1, 
                                          min_bound=0., max_bound=8., normalization=True, symmetry=False, skip=1)
    with open(rdf_results_file_max_6, 'wb') as file:
        pickle.dump(ucg_comRDF_max_6, file)
    
    #################################################################################################
    # RDF
    ################################################################################################
    box_bounds = [2.0870640000000002e+01, 2.0870640000000002e+01, 2.0870640000000002e+01, 90, 90, 90]
    bins = 50
    ucg_comRDF_max_6 = target_compute_rdf(ucg_chirality_traj, box_bounds, bins, start=1, stop=-1, 
                                          min_bound=0., max_bound=8., normalization=True, symmetry=True, skip=1)
    with open(symm_rdf_results_file_max_6, 'wb') as file:
        pickle.dump(ucg_comRDF_max_6, file)
    
    #################################################################################################
    ## Chiral counts
    #################################################################################################
    ucg_count = target_compute_averageChirality(ucg_chirality_traj, start=1, stop=-1, skip=1)
    with open(count_results_file_max6_filename, 'wb') as file:
        pickle.dump(ucg_count, file)







   

 
