//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file turb.cpp
//  \brief Problem generator for turbulence generator
//

// C headers

// C++ headers
#include <cmath>
#include <ctime>
#include <sstream>
#include <stdexcept>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../fft/athena_fft.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp"

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

Real gm1,gem1,ge,g;

 void electron_update(MeshBlock *pmb,const Real time, const Real dt,const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);


std::int64_t rseed; // seed for turbulence power spectrum

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (SELF_GRAVITY_ENABLED) {
    Real four_pi_G = pin->GetReal("problem","four_pi_G");
    Real eps = pin->GetOrAddReal("problem","grav_eps", 0.0);
    SetFourPiG(four_pi_G);
    SetGravityThreshold(eps);
  }

  // turb_flag is initialzed in the Mesh constructor to 0 by default;
  // turb_flag = 1 for decaying turbulence
  // turb_flag = 2 for impulsively driven turbulence
  // turb_flag = 3 for continuously driven turbulence
  turb_flag = pin->GetInteger("problem","turb_flag");
  if (turb_flag != 0) {
#ifndef FFT
    std::stringstream msg;
    msg << "### FATAL ERROR in TurbulenceDriver::TurbulenceDriver" << std::endl
        << "non zero Turbulence flag is set without FFT!" << std::endl;
    ATHENA_ERROR(msg);
    return;
#endif
  }
  return;
}
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {

    gm1 = peos->GetGamma() - 1.0;
    g = peos->GetGamma();
    ge = 4.0/3.0;
    gem1 = ge-1.0;


  return;
}
//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        phydro->u(IDN,k,j,i) = 1.0;

        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;

        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = 1.0;
        }


                // set entropy
        if (NSCALARS > 0) pscalars->s(0,k,j,i) = 1.0; //total
        if (NSCALARS > 1) pscalars->s(1,k,j,i) = 0.1; //electron
      }
    }
  }
}


//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
}


//NOTE: primitives are half time step (or initial), conservatives at end of time step (or half)
void electron_update(MeshBlock *pmb,const Real time, const Real dt,const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons){


        AthenaArray<Real> prim_new,bcc_new;
        prim_new.NewAthenaArray(NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1);
        bcc_new.NewAthenaArray(NFIELD, pmb->ncells3, pmb->ncells2, pmb->ncells1);
        pmb->peos->ConservedToPrimitive(cons, prim, pmb->pfield->b,prim_new, bcc_new,pmb->pcoord, 
          pmb->is, pmb->ie, pmb->js, pmb->je, pmb->ks, pmb->ke);

     for (int k=pmb->ks; k<=pmb->ke; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {

        Real dh = prim(IDN,k,j,i);
        Real pnew = prim_new(IPR,k,j,i);
        Real dnew = prim_new(IDN,k,j,i);
        Real r_actual = pnew/std::pow(dnew,gm1+1);

        Real s_actual = dnew * r_actual;

        Real Q = std::pow(dh,gm1)/gm1 * (s_actual - pmb->pscalars->s(0,k,j,i))/dt;

        Real fe = 0.5;

        pmb->pscalars->r(1,k,j,i) += fe * gem1/(gm1) * std::pow(dh,g-ge) * (r_actual - pmb->pscalars->r(0,k,j,i));
        pmb->pscalars->s(1,k,j,i) = dnew * pmb->pscalars->r(1,k,j,i);


        pmb->pscalars->s(0,k,j,i) = s_actual;
        pmb->pscalars->r(0,k,j,i) = r_actual;

      

      }
    }
  }

  prim_new.DeleteAthenaArray();
  bcc_new.DeleteAthenaArray();
}