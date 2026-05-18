LAMMPS `fix` command parameters to evolve chirality of the UCG bead
----

- PARAMETERS: tau_chiral, unused_param, logistic_parameter, logistic_inflection, logistic_numerator, random_seed, cv_mode, dist_cutoff 
n_neighbors, maxiter

- EXAMPLE: fix             evolve_chirality all update/chirality tau_chiral unused_param logistic_parameter logistic_inflection logistic_numerator random_seed cv_mode dist_cutoff n_neighbors maxiter
---
- tau_chiral = chiral evolution frequency.
- unused_param = set to 3. Not important (previously supported tanh parameter).
- logistic_parameter = a in logistic function.
- logistic_inflection = set to 0.5. It is b in logistic function.
- logistic_numerator = set to 1.0. It is c in logistic function.
- random_seed = 42. It is seed to draw random uniform number.
- cv_mode = set to 0. It is for evolving chirality as in the paper. 1 ignore neighbors.
- dist_cutoff = 4.0. Not important (previously employed to select n neighbors within a distance cutoff).
- n_neighbors = set to 12. Number of N nearest neighbors.
- maxiter = Set to 1 only . Not important (previously employed for self-consistent assignment of chirality.).

