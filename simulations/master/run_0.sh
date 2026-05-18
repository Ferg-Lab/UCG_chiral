for i in SSSS
do
    sed s/OOOO/iter_$i/g master.lammps > temp.lammps
    sed s/OOOO/iter_$i/g get_coeffs.py > temp.py
    sigmoid=`python temp.py`
    sed s/SIGM/$sigmoid/g temp.lammps > temp2.lammps
    sed s/TTTT/UUUU/g temp2.lammps > temp3.lammps
    sed s/LLLL/LULU/g temp3.lammps > temp4.lammps
    sed s/NNNN/NUNU/g temp4.lammps > ucg-lmp_chiral.lammps 
    rm -f temp.py temp.lammps temp2.lammps temp3.lammps temp4.lammps

    sed s/SIGM/$sigmoid/g master.conf > state.conf

    source mdrun.in 

    mv UCG* iter_$i/
    mv out* iter_$i/
    mv log* iter_$i/
    mv ucg-lmp_chiral.lammps iter_$i/
    mv state.conf iter_$i/

done




