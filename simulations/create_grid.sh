for temperature in 5.0 # 4.6 4.5 4.2 3.8
do
for nsteps in 50000
do
for chiral_lag in 100 #100 #50 #1
do
mkdir t_"$temperature"_chiral_lag_"$chiral_lag"_nsteps_"$nsteps"
cd t_"$temperature"_chiral_lag_"$chiral_lag"_nsteps_"$nsteps"

# 5.0, 4.6, 4.2 [1,50,100]
cp -r ../master/init_potentials_4.6/iter_0 .

# 3.8 [50]
#cp -r ../t_4.2_chiral_lag_"$chiral_lag"_nsteps_"$nsteps"/iter_1299 iter_0

# 3.8 - 100
#cp -r ../master/init_potentials_4.6/iter_0 .

cp ../master/master.conf .
cp ../master/master.lammps .
cp ../master/mdrun.in .
cp ../master/run_0.sh .
cp ../master/ucg_initialconfig_ee_zero.dat .
cp ../master/*.py .
sed s/TTTT/$temperature/g ../master/process_traj.py > process_traj.py

sed s/CCCC/$chiral_lag/g ../master/job_ib_parallel.submit > temp.submit
sed s/TUTU/$temperature/g temp.submit > temp2.submit
sed s/NNNN/$nsteps/g temp2.submit > job_ib_parallel.submit
#sbatch job_ib_parallel.submit

rm -f temp.submit temp2.submit

cd ..
done
done
done
