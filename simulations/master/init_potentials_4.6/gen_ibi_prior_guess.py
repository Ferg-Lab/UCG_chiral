import os
import numpy as np
import pandas as pd
import pickle
from scipy.interpolate import CubicSpline

import sys
sys.path.append('../../../../../../scripts/')
sys.path.append('..')
from analysis_scripts import *
from write_potentials import *

#### Step 1: load AA RDF
################################################################################################
# Read AA RDF [See scripts/run_rdf.py  for calculation details]
################################################################################################
TTTT=4.6
ref_temperature=TTTT
aa_traj_file_path = '/project2/andrewferguson/sivadasetty/doe/ucg/chiral_tetramer/SystemSize1000tetramers/T-'+str(ref_temperature)+'_freqdouble_racemic/'
rdf_results_file_max_6 = aa_traj_file_path +  'chiral_results/nosymmetry_rdf_halfbox_maxbound_8_nbinary.pkl'
with open(rdf_results_file_max_6, 'rb') as file:
    ucg_comRDF_max_6 = pickle.load(file)

################################################################################################
# Save Boltzmann inverted potentials and forces
################################################################################################
ucg_comRDF = ucg_comRDF_max_6 
kt_reduced = ref_temperature
#prior potentials
ull_pot = -kt_reduced*np.log(ucg_comRDF['t1']['rdf']*2) + kt_reduced*np.log(ucg_comRDF['t1']['rdf']*2)[-1]
udd_pot = -kt_reduced*np.log(ucg_comRDF['t2']['rdf']*2) + kt_reduced*np.log(ucg_comRDF['t2']['rdf']*2)[-1]
uld_pot = -kt_reduced*np.log(ucg_comRDF['t12']['rdf']*1) + kt_reduced*np.log(ucg_comRDF['t12']['rdf']*1)[-1]

xll = ucg_comRDF['t1']['bin_centers'] #[valid_mask_ll]
xdd = ucg_comRDF['t2']['bin_centers'] #[valid_mask_dd]
xld = ucg_comRDF['t12']['bin_centers'] #[valid_mask_ld]

### Initial trial: NaN bins values replaced non-decreasing numbers 
#ull_pot[0] = 330
#ull_pot[1] = 250
#ull_pot[2] = 140
#ull_pot[3] = 80
#ull_pot[4] = 60
#ull_pot[5] = 50
#ull_pot[6] = 40
#ull_pot[7] = 30
#
#uld_pot[0] = 750
#uld_pot[1] = 540
#uld_pot[2] = 410
#uld_pot[3] = 330
#uld_pot[4] = 250
#uld_pot[5] = 140
#uld_pot[6] = 80
#uld_pot[7] = 60
#uld_pot[8] = 50
#uld_pot[9] = 45
#uld_pot[10] = 41

# ULL = UDD; So UDD not used
#udd_pot[0] = 680
#udd_pot[1] = 410
#udd_pot[2] = 330
#udd_pot[3] = 250
#udd_pot[4] = 140
#udd_pot[5] = 80
#udd_pot[6] = 60
#udd_pot[7] = 50

ull_pot_valid = ull_pot
#udd_pot_valid = ull_pot # DD = LL [IGNORING udd_pot]
uld_pot_valid = uld_pot

#ull_smoothened = ull_pot_valid
#uld_smoothened = uld_pot_valid

# Smoothen the potentials using Equation 7 in [http://dx.doi.org/10.1063/1.4880555]
ull_smoothened = np.zeros_like(ull_pot_valid)
for n in range(1, len(ull_pot_valid)-1):
    ull_smoothened[n] = (ull_pot_valid[n-1] + ull_pot_valid[n] + ull_pot_valid[n+1]) / 3
# handle boundaries (copy or extrapolate)
ull_smoothened[0] = ull_pot_valid[0]  # or some other rule
ull_smoothened[-1] = ull_pot_valid[-1]

# Smoothen the potentials using Equation 7 in [http://dx.doi.org/10.1063/1.4880555]
uld_smoothened = np.zeros_like(uld_pot_valid)
for n in range(1, len(uld_pot_valid)-1):
    uld_smoothened[n] = (uld_pot_valid[n-1] + uld_pot_valid[n] + uld_pot_valid[n+1]) / 3
# handle boundaries (copy or extrapolate)
uld_smoothened[0] = uld_pot_valid[0]  # or some other rule
uld_smoothened[-1] = uld_pot_valid[-1]

##udd_smoothened = ull_smoothened

ull_force = -np.gradient(ull_smoothened,xll)
ull_force -= ull_force[-1]
#udd_force = ull_force
uld_force = -np.gradient(uld_smoothened,xld)
uld_force -= uld_force[-1]

# Save potential
#write_potentials(xll, ull_smoothened, ull_force, xdd, udd_smoothened, udd_force, xld, uld_smoothened, uld_force, 'ibi_guess/prior_model/')
write_potentials(xll, ull_smoothened, ull_force, xld, uld_smoothened, uld_force, 'ibi_guess/prior_model/')



