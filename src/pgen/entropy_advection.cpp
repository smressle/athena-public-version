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
Real gm1,gem1,ge,g,gamma_adi;


} // namespace
void inner_boundary_source_function(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> *flux,
  const AthenaArray<Real> &cons_old,const AthenaArray<Real> &cons_half, AthenaArray<Real> &cons,
  const AthenaArray<Real> &prim_old,const AthenaArray<Real> &prim_half,  AthenaArray<Real> &prim, 
  const FaceField &bb_half, const FaceField &bb,
  const AthenaArray<Real> &s_old,const AthenaArray<Real> &s_half, AthenaArray<Real> &s_scalar, AthenaArray<Real> &prim_scalar);

void electron_update(const Real dt, const AthenaArray<Real> *flux,
  Coordinates *pcoord, EquationOfState *peos, Field *pfield, PassiveScalars *ps,
  const AthenaArray<Real> &cons_old, const AthenaArray<Real> &cons_half, AthenaArray<Real> &cons,
  const AthenaArray<Real> &prim_old, const AthenaArray<Real> &prim_half, AthenaArray<Real> &prim, 
  const FaceField &bb_half,const FaceField &bb, 
  const AthenaArray<Real> &s_old,const AthenaArray<Real> &s_half, AthenaArray<Real> &s_scalar, 
  AthenaArray<Real> &r_scalar, int is, int ie, int js, int je, int ks, int ke );
 //void electron_update(MeshBlock *pmb,const Real time, const Real dt,const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
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


    EnrollUserRadSourceFunction(inner_boundary_source_function);


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
        if (NSCALARS>1) fprintf(stderr,"r: %g s: %g \n", pmb->pscalars->r(1,k,j,is-i), pmb->pscalars->s(1,k,j,is-i));
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
    gamma_adi = g;
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

  Real L = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;
  Real xs = pmy_mesh->mesh_size.x1min;
  Real xmid = xs + L/2.0;
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      pcoord->CellVolume(k, j, il, iu, vol);
      for (int i=il; i<=iu; i++) {
        // background fluid:


        Real x = pcoord->x1v(i);
        // Real L = pcoord->x1f(ie+1) - pcoord->x1f(is);
        // Real xs = pcoord->x1f(is);
        // Real xe = pcoord->x1f(ie+1);
        Real xmid = xs + L/2.0;

        Real p0 = 1e-8;
        Real v0 = 1.0;
        Real d0 = 1.0;
        if (x > xmid) v0 = -v0;

        //if (x>xs + L/4.0 && x<xs + 3.0*L/4.0) d0 = .25;
        phydro->u(IDN,k,j,i) = d0;
        phydro->w(IDN,k,j,i) = d0;
        phydro->w1(IDN,k,j,i) = d0;
        phydro->w(IVX,k,j,i) = v0;
        phydro->w(IPR,k,j,i) = p0;
        phydro->w1(IPR,k,j,i) = p0;
        phydro->w1(IVX,k,j,i) = v0;
        phydro->u(IM1,k,j,i) = d0*v0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;

        phydro->u(IEN,k,j,i) = d0 * SQR(v0)/2.0 + p0/gm1; 

        // set entropy
        if (NSCALARS > 0) pscalars->s(0,k,j,i) = d0 * p0/std::pow(d0,g); //total
        if (NSCALARS > 1) {
                pscalars->s(1,k,j,i) = d0 * p0/std::pow(d0,ge); //electron
                pscalars->r(1,k,j,i) = p0/std::pow(d0,ge);

        }
        
      }
    }
  }
  return;
}


// //NOTE: primitives are half time step (or initial), conservatives at end of time step (or half)
// void electron_update(MeshBlock *pmb,const Real time, const Real dt,const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons){


//         AthenaArray<Real> prim_new,bcc_new;
//         prim_new.NewAthenaArray(NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1);
//         bcc_new.NewAthenaArray(NFIELD, pmb->ncells3, pmb->ncells2, pmb->ncells1);
//         pmb->peos->ConservedToPrimitive(cons, prim, pmb->pfield->b,prim_new, bcc_new,pmb->pcoord, 
//           pmb->is, pmb->ie, pmb->js, pmb->je, pmb->ks, pmb->ke);

//      for (int k=pmb->ks; k<=pmb->ke; ++k) {
// #pragma omp parallel for schedule(static)
//     for (int j=pmb->js; j<=pmb->je; ++j) {
// #pragma simd
//       for (int i=pmb->is; i<=pmb->ie; ++i) {

//         Real dh = prim(IDN,k,j,i);
//         Real pnew = prim_new(IPR,k,j,i);
//         Real dnew = prim_new(IDN,k,j,i);
//         Real r_actual = pnew/std::pow(dnew,gm1+1);

//         Real s_actual = dnew * r_actual;

//         Real Q = std::pow(dh,gm1)/gm1 * (s_actual - pmb->pscalars->s(0,k,j,i))/dt;

//         Real fe = 0.5;

//         pmb->pscalars->r(1,k,j,i) += fe * gem1/(gm1) * std::pow(dh,g-ge) * (r_actual - pmb->pscalars->r(0,k,j,i));
//         pmb->pscalars->s(1,k,j,i) = dnew * pmb->pscalars->r(1,k,j,i);


//         pmb->pscalars->s(0,k,j,i) = s_actual;
//         pmb->pscalars->r(0,k,j,i) = r_actual;

      

//       }
//     }
//   }

//   prim_new.DeleteAthenaArray();
//   bcc_new.DeleteAthenaArray();
// }
Real kappa_to_ue(Real kappa,Real den, Real gamma_)
{
  return kappa * std::pow(den,gamma_) / (gamma_-1.0);

}
Real ue_to_kappa(Real u, Real den,Real gamma_)
{
  return (gamma_-1.0) * u/std::pow(den,gamma_);

}

void electron_update(const Real dt, const AthenaArray<Real> *flux,Coordinates *pcoord, 
  EquationOfState *peos, Field *pfield, PassiveScalars *ps,
  const AthenaArray<Real> &cons_old, const AthenaArray<Real> &cons_half, AthenaArray<Real> &cons,
  const AthenaArray<Real> &prim_old,const AthenaArray<Real> &prim_half, AthenaArray<Real> &prim, 
  const FaceField &bb_half,const FaceField &bb, 
  const AthenaArray<Real> &s_old, const AthenaArray<Real> &s_half, AthenaArray<Real> &s_scalar, 
  AthenaArray<Real> &r_scalar, 
  int is, int ie, int js, int je, int ks, int ke ) {
  // Create aliases for metric

//not sure how to avoid this #if statement.  
#if (GENERAL_RELATIVITY)
  AthenaArray<Real> &g = pcoord->pmy_block->ruser_meshblock_data[0],&gi = pcoord->pmy_block->ruser_meshblock_data[1];
#else
  AthenaArray<Real> g,gi; //should never be called
#endif


  ///undo passive scalar update

  // ps->AddFluxDivergence(-dt, s_scalar);



  Real d_floor = peos->GetDensityFloor();
  Real p_floor = peos->GetPressureFloor();

  AthenaArray<Real> bcc1;


  int il = is - NGHOST; int jl = js; int kl = ks;
  int iu = ie + NGHOST; int ju = je; int ku = ke;
  if (pcoord->pmy_block->ncells2>1) {
    jl -= NGHOST; ju += NGHOST;
  }
  if (pcoord->pmy_block->ncells3>1) {
    kl -= NGHOST; ku += NGHOST;
  }



  // Go through all cells
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      if (GENERAL_RELATIVITY) pcoord->CellMetric(k, j, is, ie, g, gi);
      for (int i=is; i<=ie; ++i) {


        Real dh = prim_half(IDN,k,j,i);
        Real ph = prim_half(IPR,k,j,i);
        Real pnew = prim(IPR,k,j,i);
        Real dnew = prim(IDN,k,j,i);
        Real r_actual = pnew/std::pow(dnew,gamma_adi);

        Real s_actual = cons(IDN,k,j,i) * r_actual;

        //Real Q = std::pow(dh,gm1)/gm1 * (s_actual - pmb->pscalars->s(0,k,j,i))/dt;

        //Variables needed for fe

        //P = rho k T/ (mu mp)

        bool fixed = false;

        if (GENERAL_RELATIVITY) fixed = peos->GetFixedValue(k,j,i);
        else if (dnew == d_floor || pnew == p_floor) fixed = true;



        if (fixed){ //keep electron pressure unchanged when floor or other fixups are used
          // Real pe_old = r_old * std::pow(dh,ge) ; 

          // pscalars->r(1,k,j,i) = pe_old/std::pow(dnew,ge); //0.1 * pnew/std::pow(dnew,ge); //pe_old/std::pow(dnew,ge);
          // pscalars->s(1,k,j,i) = phydro->u(IDN,k,j,i) * pscalars->r(1,k,j,i);


          r_scalar(1,k,j,i) = 0.1 * pnew/std::pow(dnew,ge); //pe_old/std::pow(dnew,ge);
          s_scalar(1,k,j,i) = cons(IDN,k,j,i) * r_scalar(1,k,j,i);
          if (NSCALARS>2){
            r_scalar(2,k,j,i) = 0.1 * pnew/std::pow(dnew,ge); //pe_old/std::pow(dnew,ge);
            s_scalar(2,k,j,i) = cons(IDN,k,j,i) * r_scalar(2,k,j,i);
          }
          if (NSCALARS>3){
            //Real pe_old = pscalars->s1(3,k,j,i)/phydro->u1(IDN,k,j,i) *std::pow(dh,ge);
            r_scalar(3,k,j,i) = 0.1 * pnew/std::pow(dnew,ge);
            s_scalar(3,k,j,i) = cons(IDN,k,j,i) * r_scalar(3,k,j,i);
          }
        }
        else{ 

          //Sadowski+ 2017 Equation (51)-(52)
          Real fi = cons_old(IDN,k,j,i)/cons(IDN,k,j,i);

          Real uhat = fi * (s_old(0,k,j,i)/prim_old(IDN,k,j,i)) * std::pow(prim(IDN,k,j,i),gamma_adi) /gm1;

          Real uhat_ [6];

          uhat_[0] = uhat;

          for (int n=1; n<NSCALARS; n++) {
             Real se_old = s_old(n,k,j,i);
             Real r_old = se_old/cons_old(IDN,k,j,i);
             uhat_[n] = fi * (kappa_to_ue(r_old,prim(IDN,k,j,i),ge));
           }

          int max_dir = X1DIR;
          if (pcoord->pmy_block->ncells2>1) max_dir += 1;
          if (pcoord->pmy_block->ncells3>1) max_dir += 1;
          for (int dir=X1DIR; dir<=max_dir; ++dir) {
            Real dxp,dx;
            int ip,jp,kp;

            dxp = pcoord->dx1f(i+1) * (dir==X1DIR) + pcoord->dx2f(j+1) * (dir==X2DIR) + pcoord->dx3f(k+1) * (dir==X3DIR);
            dx =  pcoord->dx1f(i)   * (dir==X1DIR) + pcoord->dx2f(j)   * (dir==X2DIR) + pcoord->dx3f(k)   * (dir==X3DIR);
            ip = i + 1 * (dir==X1DIR);
            jp = j + 1 * (dir==X2DIR);
            kp = k + 1 * (dir==X3DIR);

            Real fp = -flux[dir](IDN,kp,jp,ip)/cons(IDN,k,j,i) * dt / dxp ;
            Real fm =  flux[dir](IDN,k ,j ,i )/cons(IDN,k,j,i) * dt / dx  ;



            // Real kp = s_flux[dir](0,kp,jp,ip)/flux[dir](IDN,kp,jp,ip);
            // Real km = s_flux[dir](0,k,j,i)/flux[dir](IDN,k,j,i);

            int i_upwindp,j_upwindp,k_upwindp; 
            if (flux[dir](IDN,kp,jp,ip)<0){
              i_upwindp = i + 1 * (dir==X1DIR);
              j_upwindp = j + 1 * (dir==X2DIR);
              k_upwindp = k + 1 * (dir==X3DIR);
            }
            else{
              i_upwindp = i;
              j_upwindp = j;
              k_upwindp = k;
            }

            int i_upwindm,j_upwindm,k_upwindm; 
            if (flux[dir](IDN,k,j,i)<0){
              i_upwindm = i;
              j_upwindm = j;
              k_upwindm = k;
            }
            else{
              i_upwindm = i - 1 * (dir==X1DIR);
              j_upwindm = j - 1 * (dir==X2DIR);
              k_upwindm = k - 1 * (dir==X3DIR);
            }

            //new densities //
            Real rhop = prim(IDN,k,j,i); //prim(IDN,k_upwindp,j_upwindp,i_upwindp);
            Real rhom = prim(IDN,k,j,i); //prim(IDN,k_upwindm,j_upwindm,i_upwindm);

              // fprintf(stderr,"i j k : %d %d %d \n ip jp kp: %d %d %d \n iupp jupp kupp: %d %d %d \n iupm jump kump: %d %d %d\n fi: %g fp: %g fm: %g \n", 
              // i,j,k,ip,jp,kp,i_upwindp, j_upwindp, k_upwindp, i_upwindm, j_upwindm, k_upwindm, fi,fp,fm);

            for (int n=0; n<NSCALARS; n++) {
              //half or old???
              Real kp = s_half(n,k_upwindp,j_upwindp,i_upwindp)/cons_half(IDN,k_upwindp,j_upwindp,i_upwindp);
              Real km = s_half(n,k_upwindm,j_upwindm,i_upwindm)/cons_half(IDN,k_upwindm,j_upwindm,i_upwindm);

              Real uhatp = kappa_to_ue(kp,rhop,ge);
              Real uhatm = kappa_to_ue(km,rhom,ge);

              if (n==0){
                uhatp = kp*std::pow(rhop,gamma_adi)/(gm1);
                uhatm = km*std::pow(rhom,gamma_adi)/(gm1);
              }

              uhat_[n] += fp * (uhatp) + fm * (uhatm); 
            } //Nscalars

            Real ue_old = kappa_to_ue(s_old(1,k,j,i)/cons_old(IDN,k,j,i),prim(IDN,k,j,i),ge);
            Real up = kappa_to_ue(s_half(1,k_upwindp,j_upwindp,i_upwindp)/cons_half(IDN,k_upwindp,j_upwindp,i_upwindp),prim(IDN,k,j,i),ge);
            Real um = kappa_to_ue(s_half(1,k_upwindm,j_upwindm,i_upwindm)/cons_half(IDN,k_upwindm,j_upwindm,i_upwindm),prim(IDN,k,j,i),ge);
            Real unew  = fi * ue_old  + fp * up + fm * um;

            // r_scalar(1,k,j,i) = ue_to_kappa(unew,prim(IDN,k,j,i),ge);
            // s_scalar(1,k,j,i) = r_scalar(1,k,j,i) * cons(IDN,k,j,i);
            // // s_scalar(1,k,j,i) = fi * s_old(1,k,j,i) + fp * s_half(1,k_upwindp,j_upwindp,i_upwindp) + 
            // // fm * s_half(1,k_upwindm,j_upwindm,i_upwindm); //r_scalar(n,k,j,i) * cons(IDN,k,j,i);
            // // r_scalar(1,k,j,i) = s_scalar(1,k,j,i)/cons(IDN,k,j,i);

          } //dir

           Real Q = ( (prim(IPR,k,j,i)/gm1) - uhat_[0] )/dt ;
           Real fe_[3] = {0.5, 0.5,0.5};

// p ds = p/kappa dkappa = rho^gamma dkappa

          for (int n=1; n<NSCALARS; n++) {
            uhat_[n] += fe_[n-1] * Q * dt;
            r_scalar(n,k,j,i) = ue_to_kappa(uhat_[n],prim(IDN,k,j,i),ge);
            // r_scalar(1,k,j,i) +=  fe_howes * gem1/(gm1) * std::pow(dh,gamma_adi-ge) * (r_actual - r_scalar(0,k,j,i));
            s_scalar(n,k,j,i) = r_scalar(n,k,j,i) * cons(IDN,k,j,i);
            // s_scalar(n,k,j,i) = fi * s_old(n,k,j,i) + fp * s_half(n,k_upwindp,j_upwindp,i_upwindp) + 
            // fm * s_half(n,k_upwindm,j_upwindm,i_upwindm); //r_scalar(n,k,j,i) * cons(IDN,k,j,i);
            // r_scalar(n,k,j,i) = s_scalar(n,k,j,i)/cons(IDN,k,j,i);

          }



        }

        s_scalar(0,k,j,i) = s_actual;
        r_scalar(0,k,j,i) = r_actual;



      }
    }
  }

  return;
}
void inner_boundary_source_function(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> *flux,
  const AthenaArray<Real> &cons_old,const AthenaArray<Real> &cons_half, AthenaArray<Real> &cons,
  const AthenaArray<Real> &prim_old,const AthenaArray<Real> &prim_half,  AthenaArray<Real> &prim, 
  const FaceField &bb_half,const FaceField &bb,
  const AthenaArray<Real> &s_old,const AthenaArray<Real> &s_half, AthenaArray<Real> &s_scalar, AthenaArray<Real> &prim_scalar){
  int i, j, k, dk;
  int is, ie, js, je, ks, ke;



  is = pmb->is;  ie = pmb->ie;
  js = pmb->js;  je = pmb->je;
  ks = pmb->ks;  ke = pmb->ke;
  Real igm1 = 1.0/(gm1);
  Real gamma = gm1+1.;


  if (NSCALARS>1 && ALLOCATE_U2) electron_update(dt,flux, pmb->pcoord, pmb->peos, pmb->pfield,pmb->pscalars,
    cons_old, cons_half, cons, prim_old,prim_half, prim, bb_half,bb, s_old,s_half,s_scalar,prim_scalar, 
    is, ie, js, je, ks, ke ) ;




}
