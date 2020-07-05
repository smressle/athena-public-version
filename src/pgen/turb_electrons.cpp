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

Real yr = 31556926.0, pc = 3.09e18;    /* yr and parsec in code units */
Real cl = 2.99792458e10 * (1e3 * yr)/pc ;      /* speed of light in code units */
Real K_in_kev = 8.6e-8;
Real mp_over_kev = 9.994827;   //mp * (pc/kyr)^2/kev
Real mp_over_me = 1836.15267507;  //proton mass in solar masses


Real P0,d0,Pe0,thetae_0;

Real X = 1e-15; //0.7;   // Hydrogen Fraction
//Real Z_sun = 0.02;  //Metalicity
Real muH = 1./X;
Real mue = 2./(1. + X);

//Lodders et al 2003
Real Z_o_X_sun = 0.0177;
Real X_sun = 0.7491;
Real Y_sun =0.2246 + 0.7409 * (Z_o_X_sun);
Real Z_sun = 1.0 - X_sun - Y_sun;
Real muH_sun = 1./X_sun;

Real Z = 3.*Z_sun;

Real mu_highT = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.);  //mean molecular weight in proton masses


std::int64_t rseed; // seed for turbulence power spectrum

Real time_step(MeshBlock *pmb);

Real time_step(MeshBlock *pmb){

  if (pmb->pmy_mesh->ncycle==0) return 1e-20;
  else return 1e10;

}

// Electron functions and variables
Real electron_energy(MeshBlock *pmb, int iout);

// Electron functions and variables
void electron_update(const Real dt, const AthenaArray<Real> *flux,
  Coordinates *pcoord, EquationOfState *peos, Field *pfield, PassiveScalars *ps,
  const AthenaArray<Real> &cons_old, const AthenaArray<Real> &cons_half, AthenaArray<Real> &cons,
  const AthenaArray<Real> &prim_old, const AthenaArray<Real> &prim_half, AthenaArray<Real> &prim, 
  const FaceField &bb_half,const FaceField &bb, 
  const AthenaArray<Real> &s_old,const AthenaArray<Real> &s_half, AthenaArray<Real> &s_scalar, 
  const AthenaArray<Real> &r_half, AthenaArray<Real> &r_scalar, int is, int ie, int js, int je, int ks, int ke );
void init_electrons(PassiveScalars *pscalars, Hydro *phydro, Field *pfield,
  int il, int iu, int jl, int ju, int kl, int ku);
Real fe_howes_(Real beta, Real sigma, Real Ttot ,Real Te);
Real fe_werner_(Real beta, Real sigma, Real Ttot ,Real Te);
Real fe_rowan_(Real beta, Real sigma_w, Real Ttot ,Real Te);
Real gm1,ge,gamma_adi;
Real ue_over_ug_init = 1.0;



Real gamma_rel(Real theta){
  return (10.0 + 20.0*theta)/(6.0 + 15.0*theta) ;
}
//kappa = theta^3/2 * (theta + 2/5)^3/2/rho
Real kappa_to_ue(Real kappa,Real den, Real gamma_)
{
  // return kappa * std::pow(den,gamma_) / (gamma_-1.0);

  //Real kT_e = theta * me cl*cl/
 // Real p_e = rho k T_e /(mue mp) = rho theta_e/mue * (me/mp) cl**2
  Real rhoe = den/mp_over_me/mue;
  Real theta_e = 1.0/5.0 * (std::sqrt(1.0 + 25.0*std::pow(rhoe*kappa,2.0/3.0)) -1.0 );
  Real pe_ = rhoe * theta_e * SQR(cl); 
  return pe_ / (gamma_rel(theta_e) - 1.0); 
}
Real ue_to_kappa(Real u, Real den,Real gamma_)
{
  // return (gamma_-1.0) * u/std::pow(den,gamma_);
  // rhoe = ne * me
  // ne = rho/(mue * mp)
  // rhoe = rho/mue * me/mp
  Real rhoe = den/mp_over_me/mue;
  Real urat = u/(rhoe * SQR(cl));
  Real theta_e = 1.0/30.0 * (-6.0 + 5.0 * urat + std::sqrt(36.0 + 180.0*urat +25.0*SQR(urat)) ) ;
  return std::pow(theta_e,3.0/2.0) * std::pow( (theta_e + 2.0/5.0),3.0/2.0) / rhoe;
}
void init_electrons(PassiveScalars *pscalars, Hydro *phydro, Field *pfield,
  int il, int iu, int jl, int ju, int kl, int ku){

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
          // set entropy
        Real ug = phydro->w(IPR,k,j,i)/gm1;
        Real press = phydro->w(IPR,k,j,i);
        Real rho = phydro->w(IDN,k,j,i);
        if (NSCALARS > 0) {
          pscalars->s(0,k,j,i) = 1.0 * phydro->u(IDN,k,j,i) * 
                                  press / std::pow(rho,gamma_adi) ; //total
          pscalars->r(0,k,j,i) = pscalars->s(0,k,j,i) / phydro->u(IDN,k,j,i);
          pscalars->s1(0,k,j,i) = pscalars->s(0,k,j,i);
        }
        for (int n=1; n<NSCALARS; ++n) {
          pscalars->s(n,k,j,i) = phydro->u(IDN,k,j,i) * ue_to_kappa(Pe0/(gamma_rel(thetae_0)-1.0),rho,ge); //electron
          pscalars->r(n,k,j,i) = pscalars->s(n,k,j,i) / phydro->u(IDN,k,j,i);
          pscalars->s1(n,k,j,i) = pscalars->s(n,k,j,i);

        }
      }
    } 
  } 

  return;
}



void electron_update(const Real dt, const AthenaArray<Real> *flux,Coordinates *pcoord, 
  EquationOfState *peos, Field *pfield, PassiveScalars *ps,
  const AthenaArray<Real> &cons_old, const AthenaArray<Real> &cons_half, AthenaArray<Real> &cons,
  const AthenaArray<Real> &prim_old,const AthenaArray<Real> &prim_half, AthenaArray<Real> &prim, 
  const FaceField &bb_half,const FaceField &bb, 
  const AthenaArray<Real> &s_old, const AthenaArray<Real> &s_half, AthenaArray<Real> &s_scalar, 
  const AthenaArray<Real> &r_half, AthenaArray<Real> &r_scalar, 
  int is, int ie, int js, int je, int ks, int ke ) {
  // Create aliases for metric



  Real d_floor = peos->GetDensityFloor();
  Real p_floor = peos->GetPressureFloor();


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



        Real se_old = s_half(1,k,j,i);
        Real r_old = se_old/cons_half(IDN,k,j,i);



        bool fixed = false;

        if (GENERAL_RELATIVITY) fixed = peos->GetFixedValue(k,j,i);
        else if (dnew == d_floor || pnew == p_floor) fixed = true;



        if (fixed){ //keep electron pressure unchanged when floor or other fixups are used
          // Real pe_old = r_old * std::pow(dh,ge) ; 

          // pscalars->r(1,k,j,i) = pe_old/std::pow(dnew,ge); //0.1 * pnew/std::pow(dnew,ge); //pe_old/std::pow(dnew,ge);
          // pscalars->s(1,k,j,i) = phydro->u(IDN,k,j,i) * pscalars->r(1,k,j,i);


          r_scalar(1,k,j,i) = ue_to_kappa(0.1*pnew/gm1,dnew,ge); //pe_old/std::pow(dnew,ge);
          s_scalar(1,k,j,i) = cons(IDN,k,j,i) * r_scalar(1,k,j,i);
          if (NSCALARS>2){
            r_scalar(2,k,j,i) =ue_to_kappa(0.1*pnew/gm1,dnew,ge); //pe_old/std::pow(dnew,ge);
            s_scalar(2,k,j,i) = cons(IDN,k,j,i) * r_scalar(2,k,j,i);
          }
          if (NSCALARS>3){
            //Real pe_old = pscalars->s1(3,k,j,i)/phydro->u1(IDN,k,j,i) *std::pow(dh,ge);
            r_scalar(3,k,j,i) = ue_to_kappa(0.1*pnew/gm1,dnew,ge);
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
             se_old = s_old(n,k,j,i);
             r_old = se_old/cons_old(IDN,k,j,i);
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

            for (int n=0; n<NSCALARS; n++) {
              //half or old???
              Real kappap = s_half(n,k_upwindp,j_upwindp,i_upwindp)/cons_half(IDN,k_upwindp,j_upwindp,i_upwindp);
              Real kappam = s_half(n,k_upwindm,j_upwindm,i_upwindm)/cons_half(IDN,k_upwindm,j_upwindm,i_upwindm);

              //NOTE: Boundary values not stored in conserved arrays until later in loop, so use old

              if (i_upwindp>ie || i_upwindp <is || j_upwindp>je || j_upwindp<js || k_upwindp>ke || k_upwindp<ks){
                kappap = r_half(n,k_upwindp,j_upwindp,i_upwindp);
              }
              if (i_upwindm>ie || i_upwindm <is || j_upwindm>je || j_upwindm<js || k_upwindm>ke || k_upwindm<ks){
                kappam = r_half(n,k_upwindm,j_upwindm,i_upwindm);
              }

              Real uhatp = kappa_to_ue(kappap,rhop,ge);
              Real uhatm = kappa_to_ue(kappam,rhom,ge);

              if (n==0){
                uhatp = kappap*std::pow(rhop,gamma_adi)/(gm1);
                uhatm = kappam*std::pow(rhom,gamma_adi)/(gm1);
              }

              uhat_[n] += fp * (uhatp) + fm * (uhatm); 

          if (isnan(uhat_[1])) {
          fprintf(stderr,"kappa: %g u: %g rho: %g \n uhatp: %g uhatm : %g kappap: %g kappam: %g \n s_halfp: %g rhop: %g shalfm: %g rhom: %g  \n",ue_to_kappa(uhat_[1],prim(IDN,k,j,i),ge),
          uhat_[1],prim(IDN,k,j,i),uhatp,uhatm, kappap,kappam,s_half(1,k_upwindp,j_upwindp,i_upwindp),cons_half(IDN,k_upwindp,j_upwindp,i_upwindp),
          s_half(1,k_upwindm,j_upwindm,i_upwindm),cons_half(IDN,k_upwindm,j_upwindm,i_upwindm)); 
          exit(0);
        }


            } //Nscalars

          } //dir

           Real Q = ( (prim(IPR,k,j,i)/gm1) - uhat_[0] )/dt ;
           Real fe_[3] = {0.5,0.5,0.5};

// p ds = p/kappa dkappa = rho^gamma dkappa

          for (int n=1; n<NSCALARS; n++) {
            uhat_[n] += fe_[n-1] * Q * dt;
            if (uhat_[n]< 0.01 * pnew/gm1 ) uhat_[n] = 0.01 * pnew/gm1;
            r_scalar(n,k,j,i) = ue_to_kappa(uhat_[n],prim(IDN,k,j,i),ge);
            // r_scalar(1,k,j,i) +=  fe_howes * gem1/(gm1) * std::pow(dh,gamma_adi-ge) * (r_actual - r_scalar(0,k,j,i));
            s_scalar(n,k,j,i) = r_scalar(n,k,j,i) * cons(IDN,k,j,i);
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
  const AthenaArray<Real> &s_old,const AthenaArray<Real> &s_half, AthenaArray<Real> &s_scalar, 
  const AthenaArray<Real> &r_half, AthenaArray<Real> &prim_scalar){
  int i, j, k, dk;
  int is, ie, js, je, ks, ke;



  is = pmb->is;  ie = pmb->ie;
  js = pmb->js;  je = pmb->je;
  ks = pmb->ks;  ke = pmb->ke;
  Real igm1 = 1.0/(gm1);
  Real gamma = gm1+1.;


  if (NSCALARS>1 && ALLOCATE_U2) electron_update(dt,flux, pmb->pcoord, pmb->peos, pmb->pfield,pmb->pscalars,
    cons_old, cons_half, cons, prim_old,prim_half, prim, bb_half,bb, s_old,s_half,s_scalar,r_half, prim_scalar, 
    is, ie, js, je, ks, ke ) ;


  // // Real vmax = 0.0;
  
  // for (k=ks; k<=ke; k++) {
  //   for (j=js; j<=je; j++) {
  //     for (i=is; i<=ie; i++) {


  //       Real v_s = sqrt(gamma*prim(IPR,k,j,i)/prim(IDN,k,j,i));

  //       if (v_s>cs_max) v_s = cs_max;
  //       if ( std::fabs(prim(IVX,k,j,i)) > cs_max) prim(IVX,k,j,i) = cs_max * ( (prim(IVX,k,j,i) >0) - (prim(IVX,k,j,i)<0) ) ;
  //       if ( std::fabs(prim(IVY,k,j,i)) > cs_max) prim(IVY,k,j,i) = cs_max * ( (prim(IVY,k,j,i) >0) - (prim(IVY,k,j,i)<0) ) ;
  //       if ( std::fabs(prim(IVZ,k,j,i)) > cs_max) prim(IVZ,k,j,i) = cs_max * ( (prim(IVZ,k,j,i) >0) - (prim(IVZ,k,j,i)<0) ) ;

  //        prim(IPR,k,j,i) = SQR(v_s) *prim(IDN,k,j,i)/gamma ;

  //        // vmax = std::max(vmax,std::fabs(prim(IVX,k,j,i)));
  //        // vmax = std::max(vmax,std::fabs(prim(IVY,k,j,i)));
  //        // vmax = std::max(vmax,std::fabs(prim(IVZ,k,j,i)));
          
  //     }
  //   }
  // }





  // apply_inner_boundary_condition(pmb,prim,prim_scalar);

  // fprintf(stderr,"vmax in rad_source: %g \n",vmax);

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

    EnrollUserRadSourceFunction(inner_boundary_source_function);

  AllocateUserHistoryOutput(1);
  EnrollUserHistoryOutput(0, electron_energy, "Ue");
    EnrollUserTimeStepFunction(time_step);


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
    d0 = 1.0;
    Real T_K = 1e8;
    Real kbT0_keV = T_K * K_in_kev;
    Pe0 = d0/(mue) / mp_over_kev * kbT0_keV;
    thetae_0 = T_K/5.92989658e9;
    P0 = d0/(mu_highT) / mp_over_kev * kbT0_keV;

    //cs= sqrt(gam * P0/rho);

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
        phydro->u(IDN,k,j,i) = d0;
        phydro->w(IDN,k,j,i) = d0;

        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;

        phydro->w(IVX,k,j,i) = 0.0;
        phydro->w(IVY,k,j,i) = 0.0;
        phydro->w(IVZ,k,j,i) = 0.0;

        phydro->w(IPR,k,j,i) = P0;

        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = 0.0 + P0/gm1;
        }

      }
    }
  }

  if (NSCALARS>0) init_electrons(pscalars, phydro, pfield,is, ie, js, je, ks, ke);


}


//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
}

void MeshBlock::UserWorkInLoop() {

// electron_update(pcoord, peos, phydro, pfield, pscalars, is, ie, js, je, ks, ke );
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
        Ue+= volume(i) * kappa_to_ue(pmb->pscalars->r(1,k,j,i), pmb->phydro->w(IDN,k,j,i),ge);
      }
    }
  }

  volume.DeleteAthenaArray();
  return Ue;
}