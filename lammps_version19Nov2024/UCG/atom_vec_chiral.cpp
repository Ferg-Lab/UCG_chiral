/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 https://www.lammps.org/, Sandia National Laboratories
 LAMMPS development team: developers@lammps.org

 Copyright (2003) Sandia Corporation.  Under the terms of Contract
 DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 certain rights in this software.  This software is distributed under
 the GNU General Public License.

 See the README file in the top-level LAMMPS directory.
 ------------------------------------------------------------------------- */

#include "atom_vec_chiral.h"

#include "atom.h"

#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecChiral::AtomVecChiral(LAMMPS *lmp) : AtomVec(lmp)
{
  molecular = Atom::MOLECULAR;
  bonds_allow = angles_allow = dihedrals_allow = impropers_allow = 1;
  mass_type = PER_TYPE;

  atom->molecule_flag = atom->q_flag = 1;
  atom->chirality_flag = 1;

  // strings with peratom variables to include in each AtomVec method
  // strings cannot contain fields in corresponding AtomVec default strings
  // order of fields in a string does not matter
  // except: fields_data_atom & fields_data_vel must match data file

  // clang-format off
  fields_grow = {"q", "molecule", "num_bond", "bond_type", "bond_atom", "num_angle", "angle_type",
    "angle_atom1", "angle_atom2", "angle_atom3", "num_dihedral", "dihedral_type", "dihedral_atom1",
    "dihedral_atom2", "dihedral_atom3", "dihedral_atom4", "num_improper", "improper_type",
    "improper_atom1", "improper_atom2", "improper_atom3", "improper_atom4", "nspecial", "special",
    "chirality"};
  fields_copy = {"q", "molecule", "num_bond", "bond_type", "bond_atom", "num_angle", "angle_type",
    "angle_atom1", "angle_atom2", "angle_atom3", "num_dihedral", "dihedral_type", "dihedral_atom1",
    "dihedral_atom2", "dihedral_atom3", "dihedral_atom4", "num_improper", "improper_type",
    "improper_atom1", "improper_atom2", "improper_atom3", "improper_atom4", "nspecial", "special",
    "chirality"};
  fields_comm = {"q", "chirality"};
  fields_border = {"q", "molecule", "chirality"};
  fields_border_vel = {"q", "molecule", "chirality"};
  fields_exchange = {"q", "molecule", "num_bond", "bond_type", "bond_atom", "num_angle",
    "angle_type", "angle_atom1", "angle_atom2", "angle_atom3", "num_dihedral", "dihedral_type",
    "dihedral_atom1", "dihedral_atom2", "dihedral_atom3", "dihedral_atom4", "num_improper",
    "improper_type", "improper_atom1", "improper_atom2", "improper_atom3", "improper_atom4",
    "nspecial", "special", "chirality"};
  fields_restart = {"q", "molecule", "num_bond", "bond_type", "bond_atom", "num_angle",
    "angle_type", "angle_atom1", "angle_atom2", "angle_atom3", "num_dihedral", "dihedral_type",
    "dihedral_atom1", "dihedral_atom2", "dihedral_atom3", "dihedral_atom4", "num_improper",
    "improper_type", "improper_atom1", "improper_atom2", "improper_atom3", "improper_atom4",
    "chirality"};
  fields_create = {"q", "molecule", "num_bond", "num_angle", "num_dihedral", "num_improper",
    "nspecial", "chirality"};
  fields_data_atom = {"id", "molecule", "type", "q", "x", "chirality"};
  fields_data_vel = {"id", "v"};
  // clang-format on

  setup_fields();
}

/* ----------------------------------------------------------------------
   set local copies of all grow ptrs used by this class, except defaults
   needed in replicate when 2 atom classes exist and it calls pack_restart()
------------------------------------------------------------------------- */

void AtomVecChiral::grow_pointers()
{
  num_bond = atom->num_bond;
  bond_type = atom->bond_type;
  num_angle = atom->num_angle;
  angle_type = atom->angle_type;
  num_dihedral = atom->num_dihedral;
  dihedral_type = atom->dihedral_type;
  num_improper = atom->num_improper;
  improper_type = atom->improper_type;
  nspecial = atom->nspecial;

  chirality = atom->chirality;
}

/* ----------------------------------------------------------------------
   initialize non-zero atom quantities
------------------------------------------------------------------------- */

void AtomVecChiral::create_atom_post(int ilocal)
{
  chirality[ilocal] = 1.0;
}

/* ----------------------------------------------------------------------
   modify what AtomVec::data_atom() just unpacked
   or initialize other atom quantities
------------------------------------------------------------------------- */

void AtomVecChiral::data_atom_post(int ilocal)
{
  num_bond[ilocal] = 0;
  num_angle[ilocal] = 0;
  num_dihedral[ilocal] = 0;
  num_improper[ilocal] = 0;
  nspecial[ilocal][0] = 0;
  nspecial[ilocal][1] = 0;
  nspecial[ilocal][2] = 0;
}

/* ----------------------------------------------------------------------
   assign an index to named atom property and return index
   return -1 if name is unknown to this atom style
------------------------------------------------------------------------- */

int AtomVecChiral::property_atom(const std::string &name)
{
  if (name == "chirality") return 0;
  return -1;
}

/* ----------------------------------------------------------------------
   pack per-atom data into buf for ComputePropertyAtom
   index maps to data specific to this atom style
------------------------------------------------------------------------- */

void AtomVecChiral::pack_property_atom(int index, double *buf, int nvalues, int groupbit)
{
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int n = 0;

  if (index == 0) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit)
        buf[n] = chirality[i];
      else
        buf[n] = 0.0;
      n += nvalues;
    }
  }
}
