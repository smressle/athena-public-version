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
#include "../scalars/scalars.hpp"


#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif


std::int64_t rseed; // seed for turbulence power spectrum


// Electron functions and variables
Real electron_energy(MeshBlock *pmb, int iout);
void electron_update(Coordinates *pcoord, EquationOfState *peos, Hydro *phydro, Field *pfield, 
  PassiveScalars *pscalars, int is, int ie, int js, int je, int ks, int ke );
void init_electrons(PassiveScalars *pscalars, Hydro *phydro, Field *pfield,
  int il, int iu, int jl, int ju, int kl, int ku);
Real fe_calc(Real beta, Real Ttot ,Real Te);
Real gm1,gem1,ge,gamma_adi;

void init_electrons(PassiveScalars *pscalars, Hydro *phydro, Field *pfield,
  int il, int iu, int jl, int ju, int kl, int ku){

  Real Te_over_Ttot_init = 0.1;
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
          // set entropy
        if (NSCALARS > 0) {
          pscalars->s(0,k,j,i) = 1.0 * phydro->u(IDN,k,j,i) * 
                                  phydro->w(IPR,k,j,i) / std::pow(phydro->w(IDN,k,j,i),gamma_adi) ; //total
          pscalars->r(0,k,j,i) = pscalars->s(0,k,j,i) / phydro->u(IDN,k,j,i);
          pscalars->s1(0,k,j,i) = pscalars->s(0,k,j,i);
        }
        if (NSCALARS > 1) {
          pscalars->s(1,k,j,i) = Te_over_Ttot_init * phydro->u(IDN,k,j,i) * 
                                  phydro->w(IPR,k,j,i) / std::pow(phydro->w(IDN,k,j,i),ge); //electron
          pscalars->r(1,k,j,i) = pscalars->s(1,k,j,i) / phydro->u(IDN,k,j,i);
          pscalars->s1(1,k,j,i) = pscalars->s(1,k,j,i);

        }
      }
    } 
  } 

  return;
}


Real fe_calc(Real beta, Real Ttot ,Real Te)
{

  Real mrat = 1836.152672; //mp/me
  if (Te<1e-15) Te = 1e-15;
  Real Trat = std::fabs(Ttot/Te);
  //Calculations for fe
  Real c1 = .92 ;//heating constant
  
  if(beta>1.e20 || std::isnan(beta) || std::isinf(beta) ) beta = 1.e20;
  Real mbeta = 2.-.2*std::log10(Trat);
  
  Real c3,c2;
  if(Trat<=1.){
      
      c2 = 1.6 / Trat ;
      c3 = 18. + 5.*std::log10(Trat);
      
  }
  else{
      c2 = 1.2/ Trat ;
      c3 = 18. ;
  }
  
  Real c22 = std::pow(c2,2.);
  Real c32 = std::pow(c3,2.);
  
  Real Qp_over_Qe = c1 * (c22+std::pow(beta,mbeta))/(c32 + std::pow(beta,mbeta)) * exp(-1./beta)*std::pow(mrat*Trat,.5) ;
  

  
  return 1./(1.+Qp_over_Qe);

}

void electron_update(Coordinates *pcoord, EquationOfState *peos, Hydro *phydro, Field *pfield, 
  PassiveScalars *pscalars, int is, int ie, int js, int je, int ks, int ke ) {
  // Create aliases for metric

//not sure how to avoid this #if statement.  
#if (GENERAL_RELATIVITY)
  AthenaArray<Real> &g = phydro->pmy_block->ruser_meshblock_data[0],&gi = phydro->pmy_block->ruser_meshblock_data[1];
#else
  AthenaArray<Real> g,gi; //should never be called
#endif



  Real d_floor = peos->GetDensityFloor();
  Real p_floor = peos->GetPressureFloor();

  AthenaArray<Real> bcc1;


  int il = is - NGHOST; int jl = js; int kl = ks;
  int iu = ie + NGHOST; int ju = je; int ku = ke;
  if (phydro->pmy_block->ncells2>1) {
    jl -= NGHOST; ju += NGHOST;
  }
  if (phydro->pmy_block->ncells3>1) {
    kl -= NGHOST; ku += NGHOST;
  }


  if (MAGNETIC_FIELDS_ENABLED) 
    bcc1.NewAthenaArray(NFIELD, phydro->pmy_block->ncells3, phydro->pmy_block->ncells2, phydro->pmy_block->ncells1);


  if (MAGNETIC_FIELDS_ENABLED) pfield->CalculateCellCenteredField(pfield->b1, bcc1, pcoord, il, iu, jl, ju, kl, ku);
  // Go through all cells
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      if (GENERAL_RELATIVITY) pcoord->CellMetric(k, j, is, ie, g, gi);
      for (int i=is; i<=ie; ++i) {

        Real b_sqh;
        if (GENERAL_RELATIVITY){
          // Calculate normal-frame Lorentz factor at half time step
          Real uu1 = phydro->w1(IM1,k,j,i);
          Real uu2 = phydro->w1(IM2,k,j,i);
          Real uu3 = phydro->w1(IM3,k,j,i);
          Real tmp = g(I11,i)*uu1*uu1 + 2.0*g(I12,i)*uu1*uu2 + 2.0*g(I13,i)*uu1*uu3
                     + g(I22,i)*uu2*uu2 + 2.0*g(I23,i)*uu2*uu3
                     + g(I33,i)*uu3*uu3;
          Real gamma = std::sqrt(1.0 + tmp);


          // Calculate 4-velocity
          Real alpha = std::sqrt(-1.0/gi(I00,i));
          Real u0 = gamma/alpha;
          Real u1 = uu1 - alpha * gamma * gi(I01,i);
          Real u2 = uu2 - alpha * gamma * gi(I02,i);
          Real u3 = uu3 - alpha * gamma * gi(I03,i);
          Real u_0, u_1, u_2, u_3;
          pcoord->LowerVectorCell(u0, u1, u2, u3, k, j, i, &u_0, &u_1, &u_2, &u_3);



          // Calculate 4-magnetic field
          Real bb1 = bcc1(IB1,k,j,i);
          Real bb2 = bcc1(IB2,k,j,i);
          Real bb3 = bcc1(IB3,k,j,i);
          Real b0 = g(I01,i)*u0*bb1 + g(I02,i)*u0*bb2 + g(I03,i)*u0*bb3
                    + g(I11,i)*u1*bb1 + g(I12,i)*u1*bb2 + g(I13,i)*u1*bb3
                    + g(I12,i)*u2*bb1 + g(I22,i)*u2*bb2 + g(I23,i)*u2*bb3
                    + g(I13,i)*u3*bb1 + g(I23,i)*u3*bb2 + g(I33,i)*u3*bb3;
          Real b1 = (bb1 + b0 * u1) / u0;
          Real b2 = (bb2 + b0 * u2) / u0;
          Real b3 = (bb3 + b0 * u3) / u0;
          Real b_0, b_1, b_2, b_3;
          pcoord->LowerVectorCell(b0, b1, b2, b3, k, j, i, &b_0, &b_1, &b_2, &b_3);

          // Calculate magnetic pressure
          b_sqh = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;
        }
        else{ //non relativistic
          if (MAGNETIC_FIELDS_ENABLED) b_sqh = SQR(bcc1(IB1,k,j,i)) + SQR(bcc1(IB2,k,j,i)) + SQR(bcc1(IB3,k,j,i));
        }

        Real dh = phydro->w1(IDN,k,j,i);
        Real ph = phydro->w1(IPR,k,j,i);
        Real pnew = phydro->w(IPR,k,j,i);
        Real dnew = phydro->w(IDN,k,j,i);
        Real r_actual = pnew/std::pow(dnew,gamma_adi);

        Real s_actual = phydro->u(IDN,k,j,i) * r_actual;

        //Real Q = std::pow(dh,gm1)/gm1 * (s_actual - pmb->pscalars->s(0,k,j,i))/dt;

        //Variables needed for fe

        // Real s_old = pscalars->s1(1,k,j,i);
        // Real r_old = s_old/phydro->u1(IDN,k,j,i);

        // Real beta = 2.0 * ph/(b_sqh + 1e-15);
        // Real Ttot = ph/dh;
        // Real Te   = r_old * std::pow(dh,ge) / dh;

        //Real fe = fe_calc(beta,Ttot,Te);
        Real fe = 0.5;

        bool fixed = false;

        if (GENERAL_RELATIVITY) fixed = peos->GetFixedValue(k,j,i);
        else if (dnew == d_floor || pnew == p_floor) fixed = true;

        if (fixed){ //keep electron pressure unchanged when floor or other fixups are used
          // Real pe_old = r_old * std::pow(dh,ge) ; 

          // pscalars->r(1,k,j,i) = pe_old/std::pow(dnew,ge); //0.1 * pnew/std::pow(dnew,ge); //pe_old/std::pow(dnew,ge);
          // pscalars->s(1,k,j,i) = phydro->u(IDN,k,j,i) * pscalars->r(1,k,j,i);


          pscalars->r(1,k,j,i) = 0.1 * pnew/std::pow(dnew,ge); //pe_old/std::pow(dnew,ge);
          pscalars->s(1,k,j,i) = phydro->u(IDN,k,j,i) * pscalars->r(1,k,j,i);
        }
        else{ 
          pscalars->r(1,k,j,i) +=  fe * gem1/(gm1) * std::pow(dh,gamma_adi-ge) * (r_actual - pscalars->r(0,k,j,i));
          pscalars->s(1,k,j,i) = pscalars->r(1,k,j,i) * phydro->u(IDN,k,j,i);
        }


        // Limit electron temperature to be <= total temperature

        // Ttot = pnew/dnew;
        // Te = pscalars->r(1,k,j,i) * std::pow(dnew,ge) / dnew;

        // if (Te >Ttot) Te = Ttot;
        // pscalars->r(1,k,j,i) = Te * dnew/std::pow(dnew,ge);
        // pscalars->s(1,k,j,i) = pscalars->r(1,k,j,i) * phydro->u(IDN,k,j,i);

        // if (Te>Ttot)
        // if (std::isnan(pscalars->s(1,k,j,i)) || std::isinf(pscalars->s(1,k,j,i)) ){
        //    fprintf(stderr,"fixed: %d r_actual: %g s: %g pe_old: %g\n",fixed,r_actual,pscalars->s(0,k,j,i),r_old * std::pow(dh,ge) );
        //    exit(0);
        // }


        pscalars->s(0,k,j,i) = s_actual;
        pscalars->r(0,k,j,i) = r_actual;



      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) bcc1.DeleteAthenaArray();
  return;
}


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

  AllocateUserHistoryOutput(1);
  EnrollUserHistoryOutput(0, electron_energy, "Ue");


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
    gamma_adi = peos->GetGamma();
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

void MeshBlock::UserWorkInLoop() {

electron_update(pcoord, peos, phydro, pfield, pscalars, is, ie, js, je, ks, ke );
  return;
}





Real electron_energy(MeshBlock *pmb, int iout)
{
  Real Ue=0;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> volume; // 1D array of volumes
  // allocate 1D array for cell volume used in usr def history
  volume.NewAthenaArray(pmb->ncells1);


  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k, j, is, ie, volume);
      for(int i=is; i<=ie; i++) {
        Ue+= volume(i) * pmb->pscalars->r(1,k,j,i) * std::pow(pmb->phydro->w(IDN,k,j,i),ge)/(ge-1.0);
      }
    }
  }

  volume.DeleteAthenaArray();
  return Ue;
}