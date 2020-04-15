//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file slotted_cylinder.cpp
//  \brief Slotted cylinder passive scalar advection problem generator for 2D/3D problems.
//
//========================================================================================

// C++ headers
#include <algorithm>  // min, max
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <stdio.h>


// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"
#include "../utils/gl_quadrature.hpp"

// Parameters which define initial solution -- made global so that they can be shared
namespace {
constexpr Real d0 = 1.0;
Real gm1,gem1,ge,g;


} // namespace

 void electron_update(MeshBlock *pmb,const Real time, const Real dt,const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
void FixedBoundary1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void FixedBoundary2(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int il, int iu, int jl, int ju, int kl, int ku, int ngh);
//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {

    //EnrollUserExplicitSourceFunction(electron_update);
    if (pin->GetString("mesh","ix1_bc") == "user") EnrollUserBoundaryFunction(BoundaryFace::inner_x1, FixedBoundary1);
    if (pin->GetString("mesh","ox1_bc") == "user") EnrollUserBoundaryFunction(BoundaryFace::outer_x1, FixedBoundary2);



  return;
}

//----------------------------------------------------------------------------------------
// Fixed boundary condition
// Inputs:
//   pmb: pointer to MeshBlock
//   pcoord: pointer to Coordinates
//   time,dt: current time and timestep of simulation
//   is,ie,js,je,ks,ke: indices demarkating active region
// Outputs:
//   prim: primitives set in ghost zones
//   bb: face-centered magnetic field set in ghost zones
// Notes:
//   does nothing

void FixedBoundary1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int is, int ie, int js, int je, int ks, int ke, int ngh) {

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        fprintf(stderr,"rho: %g ijk: %d %d %d \n", prim(IDN,k,j,is-i),i,j,k) ;
        fprintf(stderr,"r: %g s: %g \n", pmb->pscalars->r(1,k,j,is-i), pmb->pscalars->s(1,k,j,is-i));
      }
    }}
  return;
}
void FixedBoundary2(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
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

//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  AthenaArray<Real> vol(ncells1);

   // Prepare index bounds
  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;
  if (block_size.nx2 > 1) {
    jl -= (NGHOST);
    ju += (NGHOST);
  }
  int kl = ks;
  int ku = ke;
  if (block_size.nx3 > 1) {
    kl -= (NGHOST);
    ku += (NGHOST);
  }
  // initialize conserved variables
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      pcoord->CellVolume(k, j, il, iu, vol);
      for (int i=il; i<=iu; i++) {
        // background fluid:


        Real x = pcoord->x1v(i);
        Real L = pcoord->x1f(ie+1) - pcoord->x1f(is);
        Real xs = pcoord->x1f(is);
        Real xe = pcoord->x1f(ie+1);
        Real xmid = xs + L/2.0;

        Real p0 = 1e-4;
        Real v0 = 1.0;

        if (x > xmid) v0 = -v0;
        phydro->u(IDN,k,j,i) = d0;
        phydro->w(IDN,k,j,i) = d0;
        phydro->w1(IDN,k,j,i) = d0;
        phydro->u(IM1,k,j,i) = d0*v0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        // assuming isothermal EOS:
        phydro->u(IEN,k,j,i) = d0 * SQR(v0)/2.0 + p0/gm1; 

        // set entropy
        if (NSCALARS > 0) pscalars->s(0,k,j,i) = p0/std::pow(d0,g); //total
        if (NSCALARS > 1) {
                pscalars->s(1,k,j,i) = d0 * p0/std::pow(d0,ge); //electron
                pscalars->r(1,k,j,i) = p0/std::pow(d0,ge);

        }
        
      }
    }
  }
  return;
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


