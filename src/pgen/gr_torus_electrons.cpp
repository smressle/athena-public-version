//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gr_torus.cpp
//  \brief Problem generator for Fishbone-Moncrief torus.

// C headers
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// C++ headers
#include <algorithm>  // max(), max_element(), min(), min_element()
#include <cmath>      // abs(), cos(), exp(), log(), NAN, pow(), sin(), sqrt()
#include <cstring>    // strcmp()
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str(), string
#include <cfloat>


// Athena++ headers
#include "../athena.hpp"                   // macros, enums, FaceField
#include "../athena_arrays.hpp"            // AthenaArray
#include "../bvals/bvals.hpp"              // BoundaryValues
#include "../coordinates/coordinates.hpp"  // Coordinates
#include "../eos/eos.hpp"                  // EquationOfState
#include "../field/field.hpp"              // Field
#include "../hydro/hydro.hpp"              // Hydro
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"          // ParameterInput
#include "../scalars/scalars.hpp"          // Passive scalars for electrons


// Configuration checking
#if not GENERAL_RELATIVITY
#error "This problem generator must be used with general relativity"
#endif

// Declarations
enum class MagneticFieldConfigs {vertical, normal, renorm};
void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void InflowBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &bb, Real time, Real dt,
                    int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim,AthenaArray<Real> &prim_scalar);
//static void inner_boundary(MeshBlock *pmb,const Real t, const Real dt_hydro, const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim );
void inner_boundary_source_function(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> *flux,
  const AthenaArray<Real> &cons_old,const AthenaArray<Real> &cons_half, AthenaArray<Real> &cons,
  const AthenaArray<Real> &prim_old,const AthenaArray<Real> &prim_half,  AthenaArray<Real> &prim, 
  const FaceField &bb_half, const FaceField &bb,
  const AthenaArray<Real> &s_old,const AthenaArray<Real> &s_half, AthenaArray<Real> &s_scalar, 
  const AthenaArray<Real> &r_half, AthenaArray<Real> &prim_scalar);


namespace {
void GetBoyerLindquistCoordinates(Real x1, Real x2, Real x3, Real *pr,
                                  Real *ptheta, Real *pphi);
void TransformVector(Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl, Real r,
                     Real theta, Real phi,
                     Real *pa0, Real *pa1, Real *pa2, Real *pa3);
Real CalculateLFromRPeak(Real r);
Real CalculateRPeakFromL(Real l_target);
Real LogHAux(Real r, Real sin_theta);
void CalculateVelocityInTorus(Real r, Real sin_theta, Real *pu0, Real *pu3);
void CalculateVelocityInTiltedTorus(Real r, Real theta, Real phi, Real *pu0,
                                    Real *pu1, Real *pu2, Real *pu3);
Real CalculateBetaMin();
bool CalculateBeta(Real r_m, Real r_c, Real r_p, Real theta_m, Real theta_c,
                   Real theta_p, Real phi_m, Real phi_c, Real phi_p, Real *pbeta);
bool CalculateBetaFromA(Real r_m, Real r_c, Real r_p,
                        Real theta_m, Real theta_c, Real theta_p,
                        Real a_cm, Real a_cp, Real a_mc, Real a_pc,
                        Real *pbeta);
Real CalculateMagneticPressure(Real bb1, Real bb2, Real bb3, Real r, Real theta,
                               Real phi);

// Global variables
Real m, a;                                  // black hole parameters
Real r_inner_boundary = 0.0;
Real dfloor,pfloor;

Real gamma_adi, k_adi;                      // hydro parameters
Real r_edge, r_peak, l, rho_max;            // fixed torus parameters
Real psi, sin_psi, cos_psi;                 // tilt parameters
Real log_h_edge, log_h_peak;                // calculated torus parameters
Real pgas_over_rho_peak, rho_peak;          // more calculated torus parameters
Real rho_min, rho_pow, pgas_min, pgas_pow;  // background parameters
MagneticFieldConfigs field_config;                     // type of magnetic field
Real potential_cutoff;                      // sets region of torus to magnetize
Real potential_r_pow, potential_rho_pow;    // set how vector potential scales
Real beta_min;                              // min ratio of gas to mag pressure
int sample_n_r, sample_n_theta;             // number of cells in 2D sample grid
int sample_n_phi;                           // number of cells in 3D sample grid
Real sample_r_rat;                          // sample grid geometric spacing ratio
Real sample_cutoff;                         // density cutoff for sample grid
Real x1_min, x1_max, x2_min, x2_max;        // 2D limits in chosen coordinates
Real x3_min, x3_max;                        // 3D limits in chosen coordinates
Real r_min, r_max, theta_min, theta_max;    // limits in r,theta for 2D samples
Real phi_min, phi_max;                      // limits in phi for 3D samples
Real pert_amp, pert_kr, pert_kz;            // parameters for initial perturbations

Real mp_over_me = 1836.15267507;  //proton mass in solar masses
Real X = 1e-15; //0.7;   // Hydrogen Fraction
//Real Z_sun = 0.02;  //Metalicity
Real mue = 1.0;
Real ue_over_ug_floor = 0.01;
Real ue_over_ug_init = 1.0;



//Lodders et al 2003


Real mu_highT = 1./(2.);  //mean molecular weight in proton masses

} // namespace

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
Real gm1,gem1,ge;

Real gamma_rel(Real theta){
  return (10.0 + 20.0*theta)/(6.0 + 15.0*theta) ;
}
//kappa = theta^3/2 * (theta + 2/5)^3/2/rho
Real kappa_to_ue(Real kappa,Real den, Real gamma_)
{
  // return kappa * std::pow(den,gamma_) / (gamma_-1.0);

  //Real kT_e = theta * me cl*cl/
 // Real p_e = rho k T_e /(mue mp) = rho theta_e/mue * (me/mp) cl**2
  // Real rhoe = den/mp_over_me/mue;
  // Real theta_e = 1.0/5.0 * (std::sqrt(1.0 + 25.0*std::pow(rhoe*kappa,2.0/3.0)) -1.0 );
  // Real pe_ = rhoe * theta_e ; 
  // return pe_ / (gamma_rel(theta_e) - 1.0); 

  Real gamma_e = 4.0/3.0;
  return kappa * std::pow(den,gamma_e)/(gamma_e-1.0);
}
Real ue_to_kappa(Real u, Real den,Real gamma_)
{
  // return (gamma_-1.0) * u/std::pow(den,gamma_);
  // rhoe = ne * me
  // ne = rho/(mue * mp)
  // rhoe = rho/mue * me/mp
  // Real rhoe = den/mp_over_me/mue;
  // Real urat = u/(rhoe);
  // Real theta_e = 1.0/30.0 * (-6.0 + 5.0 * urat + std::sqrt(36.0 + 180.0*urat +25.0*SQR(urat)) ) ;
  // return std::pow(theta_e,3.0/2.0) * std::pow( (theta_e + 2.0/5.0),3.0/2.0) / rhoe;

  Real gamma_e = 4.0/3.0;
  Real Pe = u*(gamma_e-1.0);
  return Pe/std::pow(den,gamma_e);
}

Real kappa_to_thetae(Real kappa,Real den,Real gamma_){
  // Real rhoe = den/mp_over_me/mue;
  // return 1.0/5.0 * (std::sqrt(1.0 + 25.0*std::pow(rhoe*kappa,2.0/3.0)) -1.0 );

  Real gamma_e = 4.0/3.0;
  return kappa*std::pow(den,gamma_e-1)*mp_over_me;


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
          pscalars->s(n,k,j,i) = phydro->u(IDN,k,j,i) * ue_to_kappa(ue_over_ug_init*ug,rho,ge); //electron
          pscalars->r(n,k,j,i) = pscalars->s(n,k,j,i) / phydro->u(IDN,k,j,i);
          pscalars->s1(n,k,j,i) = pscalars->s(n,k,j,i);

          // if ( ( (phydro->pmy_block->block_size.x2min==phydro->pmy_block->pmy_mesh->mesh_size.x2min && j<NGHOST) ) ){
          //   fprintf(stderr,"inits n= %d \n ijk: %d %d %d \n kappa: %g\n",n,i,j,k,pscalars->r(n,k,j,i));
          //   fprintf(stderr,"inits n= %d \n ijk: %d %d %d \n kappa: %g\n",0,i,j,k,pscalars->r(0,k,j,i));

          //}

        }
      }
    } 
  } 

  return;
}


Real fe_howes_(Real beta, Real sigma,Real Ttot ,Real Te)
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

Real fe_werner_(Real beta, Real sigma,Real Ttot ,Real Te)
{


  Real sigma_term = sigma/5.0 / (2.0 + sigma/5.0);
  return 0.25 * ( 1.0 + std::sqrt(sigma_term) );

}

Real fe_rowan_(Real beta, Real sigma_w, Real Ttot ,Real Te){

  Real beta_max = 1.0/(4.0*sigma_w);
  if(beta_max>1.e20 || std::isnan(beta_max) || std::isinf(beta_max) )beta_max = 1e20;
  if(beta>1.e20 || std::isnan(beta) || std::isinf(beta) ) beta = 1.e20;
  if (beta>beta_max) beta = beta_max;

  Real arg_num = std::pow(1.0-beta/beta_max,3.3);
  Real arg_den = (1.0 + 1.2 * std::pow(sigma_w,0.7));
  Real arg = arg_num/arg_den;
  return 0.5 * std::exp(-arg);



}

void electron_update(const Real dt, const AthenaArray<Real> *flux,Coordinates *pcoord, 
  EquationOfState *peos, Field *pfield, PassiveScalars *ps,
  const AthenaArray<Real> &cons_old, const AthenaArray<Real> &cons_half, AthenaArray<Real> &cons,
  const AthenaArray<Real> &prim_old,const AthenaArray<Real> &prim_half, AthenaArray<Real> &prim, 
  const FaceField &bb_half,const FaceField &bb, 
  const AthenaArray<Real> &s_old, const AthenaArray<Real> &s_half, AthenaArray<Real> &s_scalar, 
  const AthenaArray<Real> &r_half_, AthenaArray<Real> &r_scalar, 
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


  if (MAGNETIC_FIELDS_ENABLED) 
    bcc1.NewAthenaArray(NFIELD, pcoord->pmy_block->ncells3, pcoord->pmy_block->ncells2, pcoord->pmy_block->ncells1);


  pfield->CalculateCellCenteredField(bb_half, bcc1, pcoord, is, ie, js, je, ks, ke);

  AthenaArray<Real> x1area,x2area,x3area;
  AthenaArray<Real> vol,x2areap,x3areap;


  x1area.NewAthenaArray(pcoord->pmy_block->ncells1+1);
  x2area.NewAthenaArray(pcoord->pmy_block->ncells1+1);
  x3area.NewAthenaArray(pcoord->pmy_block->ncells1+1);
  x2areap.NewAthenaArray(pcoord->pmy_block->ncells1+1);
  x3areap.NewAthenaArray(pcoord->pmy_block->ncells1+1);
  vol.NewAthenaArray(pcoord->pmy_block->ncells1);

  // Go through all cells
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      if (GENERAL_RELATIVITY) pcoord->CellMetric(k, j, is, ie, g, gi);
      if (GENERAL_RELATIVITY) pcoord->Face1Area(k, j, is, ie+1, x1area);
      if (GENERAL_RELATIVITY) pcoord->Face2Area(k, j, is, ie+1, x2area);
      if (GENERAL_RELATIVITY) pcoord->Face3Area(k, j, is, ie+1, x3area);
      if (GENERAL_RELATIVITY) pcoord->Face2Area(k, j+1, is, ie+1, x2areap);
      if (GENERAL_RELATIVITY) pcoord->Face3Area(k+1, j, is, ie+1, x3areap);
      if (GENERAL_RELATIVITY) pcoord->CellVolume(k,j,is,ie,vol);

      for (int i=is; i<=ie; ++i) {

        Real b_sqh;
        if (GENERAL_RELATIVITY){
          // Calculate normal-frame Lorentz factor at half time step
          Real uu1 = prim_half(IVX,k,j,i);
          Real uu2 = prim_half(IVY,k,j,i);
          Real uu3 = prim_half(IVZ,k,j,i);
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
          b_sqh = SQR(bcc1(IB1,k,j,i)) + SQR(bcc1(IB2,k,j,i)) + SQR(bcc1(IB3,k,j,i));
        }

        Real dh = prim_half(IDN,k,j,i);
        Real ph = prim_half(IPR,k,j,i);
        Real pnew = prim(IPR,k,j,i);
        Real dnew = prim(IDN,k,j,i);
        Real r_actual = pnew/std::pow(dnew,gamma_adi);

        Real s_actual = cons(IDN,k,j,i) * r_actual;

        //Real Q = std::pow(dh,gm1)/gm1 * (s_actual - pmb->pscalars->s(0,k,j,i))/dt;

        //Variables needed for fe

        //P = rho k T/ (mu mp) 

        Real beta = 2.0 * ph/(b_sqh + 1e-15);
        Real sigma = b_sqh/(dh);
        Real sigma_w = b_sqh/(dh + gamma_adi/gm1 * ph);

        Real kbTtot_kev = mu_highT * ph/dh;


        Real se_half = s_half(1,k,j,i);
        Real r_half = se_half/cons_half(IDN,k,j,i);
        Real thetae_half = kappa_to_thetae(r_half,dh,ge);

        Real kbTe_kev   = thetae_half /mp_over_me; 

        Real fe_howes = fe_howes_(beta,sigma, kbTtot_kev,kbTe_kev);

        if (NSCALARS>2)
        {
          se_half = s_half(2,k,j,i);
          r_half = se_half/cons_half(IDN,k,j,i);
          thetae_half = kappa_to_thetae(r_half,dh,ge);

          kbTe_kev   = thetae_half /mp_over_me; 

        }
        Real fe_rowan = fe_rowan_(beta,sigma_w,kbTtot_kev,kbTe_kev);
        if (NSCALARS>3)
        {
          se_half = s_half(3,k,j,i);
          r_half = se_half/cons_half(IDN,k,j,i);
          thetae_half = kappa_to_thetae(r_half,dh,ge);

          kbTe_kev   = thetae_half /mp_over_me; 
        }
        Real fe_werner = fe_werner_(beta,sigma,kbTtot_kev,kbTe_kev);
        //Real fe = 0.5;

        bool fixed = false;

        if (GENERAL_RELATIVITY) fixed = peos->GetFixedValue(k,j,i);
        else if (dnew == d_floor || pnew == p_floor) fixed = true;



        if (fixed){ //keep electron pressure unchanged when floor or other fixups are used
          // Real pe_old = r_old * std::pow(dh,ge) ; 

          // pscalars->r(1,k,j,i) = pe_old/std::pow(dnew,ge); //0.1 * pnew/std::pow(dnew,ge); //pe_old/std::pow(dnew,ge);
          // pscalars->s(1,k,j,i) = phydro->u(IDN,k,j,i) * pscalars->r(1,k,j,i);


          r_scalar(1,k,j,i) = ue_to_kappa(ue_over_ug_floor*pnew/gm1,dnew,ge); //pe_old/std::pow(dnew,ge);
          s_scalar(1,k,j,i) = cons(IDN,k,j,i) * r_scalar(1,k,j,i);
          if (NSCALARS>2){
            r_scalar(2,k,j,i) =ue_to_kappa(ue_over_ug_floor*pnew/gm1,dnew,ge); //pe_old/std::pow(dnew,ge);
            s_scalar(2,k,j,i) = cons(IDN,k,j,i) * r_scalar(2,k,j,i);
          }
          if (NSCALARS>3){
            //Real pe_old = pscalars->s1(3,k,j,i)/phydro->u1(IDN,k,j,i) *std::pow(dh,ge);
            r_scalar(3,k,j,i) = ue_to_kappa(ue_over_ug_floor*pnew/gm1,dnew,ge);
            s_scalar(3,k,j,i) = cons(IDN,k,j,i) * r_scalar(3,k,j,i);
          }
        }
        else{ 

          //Sadowski+ 2017 Equation (51)-(52)
          Real fi = cons_old(IDN,k,j,i)/cons(IDN,k,j,i);

          Real uhat = fi * (s_old(0,k,j,i)/cons_old(IDN,k,j,i)) * std::pow(prim(IDN,k,j,i),gamma_adi) /gm1;

          Real uhat_ [6];

          uhat_[0] = uhat;


          for (int n=1; n<NSCALARS; n++) {
             Real se_old = s_old(n,k,j,i);
             Real r_old = se_old/cons_old(IDN,k,j,i);
             uhat_[n] = fi * (kappa_to_ue(r_old,prim(IDN,k,j,i),ge));
           }

           int idir_end = X2DIR;
           if (pcoord->pmy_block->ncells3>1) idir_end = X3DIR;
          for (int dir=X1DIR; dir<=idir_end; ++dir) {
            Real dxp,dx;
            Real area_p,area;
            int ip,jp,kp;

            if (idir_end==X3DIR) dxp = pcoord->dx1f(i+1) * (dir==X1DIR) + pcoord->dx2f(j+1) * (dir==X2DIR) + pcoord->dx3f(k+1) * (dir==X3DIR);
            else dxp = pcoord->dx1f(i+1) * (dir==X1DIR) + pcoord->dx2f(j+1) * (dir==X2DIR);
            dx =  pcoord->dx1f(i)   * (dir==X1DIR) + pcoord->dx2f(j)   * (dir==X2DIR) + pcoord->dx3f(k)   * (dir==X3DIR);
            
            if (idir_end==X3DIR) area_p = x1area(i+1) * (dir==X1DIR) + x2areap(i) * (dir==X2DIR) + x3areap(i) * (dir==X3DIR);
            else area_p = x1area(i+1) * (dir==X1DIR) + x2areap(i) * (dir==X2DIR);
            area = x1area(i) * (dir==X1DIR) + x2area(i) * (dir==X2DIR) + x3area(i) * (dir==X3DIR);
            

            ip = i + 1 * (dir==X1DIR);
            jp = j + 1 * (dir==X2DIR);
            kp = k + 1 * (dir==X3DIR);

            // Real fp = -flux[dir](IDN,kp,jp,ip)/cons(IDN,k,j,i) * dt / dxp ;
            // Real fm =  flux[dir](IDN,k ,j ,i )/cons(IDN,k,j,i) * dt / dx  ;

            Real fp = -flux[dir](IDN,kp,jp,ip)/cons(IDN,k,j,i) * dt * area_p/vol(i) ;
            Real fm =  flux[dir](IDN,k ,j ,i )/cons(IDN,k,j,i) * dt * area/vol(i)  ;

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

            for (int n=0; n<2; n++) {
              //half or old???
              Real kappap = s_half(n,k_upwindp,j_upwindp,i_upwindp)/cons_half(IDN,k_upwindp,j_upwindp,i_upwindp);
              Real kappam = s_half(n,k_upwindm,j_upwindm,i_upwindm)/cons_half(IDN,k_upwindm,j_upwindm,i_upwindm);

              //NOTE: Boundary values not stored in conserved arrays until later in loop, so use old

              if (i_upwindp>ie || i_upwindp <is || j_upwindp>je || j_upwindp<js || k_upwindp>ke || k_upwindp<ks){
                kappap = r_half_(n,k_upwindp,j_upwindp,i_upwindp);
              }
              if (i_upwindm>ie || i_upwindm <is || j_upwindm>je || j_upwindm<js || k_upwindm>ke || k_upwindm<ks){
                kappam = r_half_(n,k_upwindm,j_upwindm,i_upwindm);
              }

              Real uhatp = kappa_to_ue(kappap,rhop,ge);
              Real uhatm = kappa_to_ue(kappam,rhom,ge);

              if (n==0){
                uhatp = kappap*std::pow(rhop,gamma_adi)/(gm1);
                uhatm = kappam*std::pow(rhom,gamma_adi)/(gm1);
              }

              uhat_[n] += fp * (uhatp) + fm * (uhatm); 

              // if (n==0 && i==64 && j==66){
              //   fprintf(stderr,"dir: %d \n fp fm fi : %g %g %g \n uhatm: %g uhatp: %g",dir, fp,fm,fi,uhatm,uhatp);
              //   fprintf(stderr,"kappa_old: %g kappa_new: %g \n dnew: %g dold: %g \n",s_old(0,k,j,i)/prim_old(IDN,k,j,i),r_actual,dnew,prim_old(IDN,k,j,i));

              // }

              // if ( ( (i_upwindm<is && pcoord->pmy_block->block_size.x1min==pcoord->pmy_block->pmy_mesh->mesh_size.x1min) || (pcoord->pmy_block->block_size.x2min==pcoord->pmy_block->pmy_mesh->mesh_size.x2min && j_upwindm<js) ) ){
              //   Real kappap_act = prim_half(IPR,k_upwindp,j_upwindp,i_upwindp)/std::pow(prim_half(IDN,k_upwindp,j_upwindp,i_upwindp),gamma_adi);
              //   Real kappam_act = prim_half(IPR,k_upwindm,j_upwindm,i_upwindm)/std::pow(prim_half(IDN,k_upwindm,j_upwindm,i_upwindm),gamma_adi);
              //   fprintf(stderr,"n: %d \n kappap_act: %g kappam_act: %g \n kappap: %g kappam: %g \n",n,kappap_act,kappam_act,kappap,kappam);
              //   fprintf(stderr,"ijk: %d %d %d \n ijkupm: %d %d %d \n ijkupp: %d %d %d \n",i,j,k,i_upwindm,j_upwindm,k_upwindm,i_upwindp,j_upwindp,k_upwindp);
              //   fprintf(stderr," x1 range: %g %g x2 range: %g %g x3 range: %g %g \n",pcoord->pmy_block->block_size.x1min,pcoord->pmy_block->block_size.x1max,pcoord->pmy_block->block_size.x2min,pcoord->pmy_block->block_size.x2max,
              //    pcoord->pmy_block->block_size.x3min,pcoord->pmy_block->block_size.x3max);
              // }


           if (std::isnan(fe_howes) || std::isnan(uhat_[n]) ) 
           {
            fprintf(stderr,"n: %d uhat: %g \n uhatp: %g uhatm: %g, fp : %g fm: %g \n kappap: %g kappam: %g rhop: %g rhom: %g \n ijk: %d %d %d iup jup kup: %d %d %d \n iupm jupm kupm: %d %d %d \n", 
              n,uhat_[n],uhatp,uhatm,fp,fm,kappap,kappam,rhop,rhom,i,j,k,i_upwindp,j_upwindp,k_upwindp,i_upwindm,j_upwindm,k_upwindm);
            fprintf(stderr," x1 range: %g %g x2 range: %g %g x3 range: %g %g \n",pcoord->pmy_block->block_size.x1min,pcoord->pmy_block->block_size.x1max,pcoord->pmy_block->block_size.x2min,pcoord->pmy_block->block_size.x2max,
              pcoord->pmy_block->block_size.x3min,pcoord->pmy_block->block_size.x3max);
            exit(0);
          }


              // if (std::isnan(uhat_[n])){
              //   fprintf(stderr,"ijk: %d %d %d iup jup kup: %d %d %d \n iupm jupm kupm: %d %d %d \n kp: %g km: %g\n cons_half: %g %g cons_old: %g %g \n",
              //     i,j,k,i_upwindp,j_upwindp,k_upwindp,i_upwindm,j_upwindm,k_upwindm,kappap,kappam,
              //     cons_old(IDN,k_upwindp,j_upwindp,i_upwindp),cons_half(IDN,k_upwindm,j_upwindm,i_upwindm), cons_old(IDN,k_upwindp,j_upwindp,i_upwindp),cons_old(IDN,k_upwindm,j_upwindm,i_upwindm));
              //   exit(0);
              // }
            } //Nscalars

          } //dir

           Real Q = ( (prim(IPR,k,j,i)/gm1) - uhat_[0] )/dt ;

              // if (i==64 && j==66)
              //   fprintf(stderr,"Q: %g ug: %g uhat: %g \n",Q,prim(IPR,k,j,i)/gm1 ,uhat_[0]);

           Real fe_[3] = {fe_howes, fe_rowan,fe_werner};


// p ds = p/kappa dkappa = rho^gamma dkappa

          for (int n=1; n<NSCALARS; n++) {
            //uhat_[n] += fe_[n-1] * Q * dt;
            uhat_[n] += 0.5 * Q * dt;


            if (uhat_[n]<ue_over_ug_floor * pnew/gm1)uhat_[n] = ue_over_ug_floor * pnew/gm1;
            r_scalar(n,k,j,i) = ue_to_kappa(uhat_[n],prim(IDN,k,j,i),ge);
            // r_scalar(1,k,j,i) +=  fe_howes * gem1/(gm1) * std::pow(dh,gamma_adi-ge) * (r_actual - r_scalar(0,k,j,i));
            s_scalar(n,k,j,i) = r_scalar(n,k,j,i) * cons(IDN,k,j,i);

          }



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


        s_scalar(0,k,j,i) = s_actual;
        r_scalar(0,k,j,i) = r_actual;



      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) bcc1.DeleteAthenaArray();

  x1area.DeleteAthenaArray();
  x2area.DeleteAthenaArray();
  x3area.DeleteAthenaArray();
  x2areap.DeleteAthenaArray();
  x3areap.DeleteAthenaArray();
  vol.DeleteAthenaArray();
  return;
}

void GetKSCoordinates(Real x1, Real x2, Real x3, Real *pr, Real *ptheta, Real *pphi){

  if (COORDINATE_SYSTEM == "kerr-schild" or COORDINATE_SYSTEM == "schwarzschild"){
      *pr = x1;
      *ptheta = x2;
      *pphi = x3;
  }
  else{


    Real R = std::sqrt(SQR(x1) + SQR(x2) + SQR(x3));
    Real r = std::sqrt( SQR(R) - SQR(a) + std::sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a)*SQR(x3) )  )/std::sqrt(2.0);

    *pr = r;
    *ptheta = std::acos(x3/r);
    *pphi = std::atan2( (r*x2 - a*x1) , (a*x2+r*x1)  );   
  }


  return;
}
//----------------------------------------------------------------------------------------
// Function for preparing Mesh
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Read problem-specific parameters from input file
  rho_min = pin->GetReal("hydro", "rho_min");
  rho_pow = pin->GetReal("hydro", "rho_pow");
  pgas_min = pin->GetReal("hydro", "pgas_min");
  pgas_pow = pin->GetReal("hydro", "pgas_pow");
  k_adi = pin->GetReal("problem", "k_adi");
  r_edge = pin->GetReal("problem", "r_edge");
  r_peak = pin->GetReal("problem", "r_peak");
  l = pin->GetReal("problem", "l");
  rho_max = pin->GetReal("problem", "rho_max");
  psi = pin->GetOrAddReal("problem", "tilt_angle", 0.0) * PI/180.0;
  sin_psi = std::sin(psi);
  cos_psi = std::cos(psi);
  if (MAGNETIC_FIELDS_ENABLED) {
    std::string field_config_str = pin->GetString("problem",
                                                  "field_config");
    if (field_config_str == "vertical") {
      field_config = MagneticFieldConfigs::vertical;
    } else if (field_config_str == "normal") {
      field_config = MagneticFieldConfigs::normal;
    } else if (field_config_str == "renorm") {
      field_config = MagneticFieldConfigs::renorm;
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in Problem Generator\n"
          << "unrecognized field_config="
          << field_config_str << std::endl;
      ATHENA_ERROR(msg);
    }

    if (field_config != MagneticFieldConfigs::vertical) {
      potential_cutoff = pin->GetReal("problem", "potential_cutoff");
      potential_r_pow = pin->GetReal("problem", "potential_r_pow");
      potential_rho_pow = pin->GetReal("problem", "potential_rho_pow");
    }
    beta_min = pin->GetReal("problem", "beta_min");
    sample_n_r = pin->GetInteger("problem", "sample_n_r");
    sample_n_theta = pin->GetInteger("problem", "sample_n_theta");
    if (psi != 0.0) {
      sample_n_phi = pin->GetInteger("problem", "sample_n_phi");
    } else {
      sample_n_phi = 1;
    }
    sample_r_rat = pin->GetReal("problem", "sample_r_rat");
    sample_cutoff = pin->GetReal("problem", "sample_cutoff");
    x1_min = pin->GetReal("mesh", "x1min");
    x1_max = pin->GetReal("mesh", "x1max");
    x2_min = pin->GetReal("mesh", "x2min");
    x2_max = pin->GetReal("mesh", "x2max");
    x3_min = pin->GetReal("mesh", "x3min");
    x3_max = pin->GetReal("mesh", "x3max");
  }
  pert_amp = pin->GetOrAddReal("problem", "pert_amp", 0.0);
  pert_kr = pin->GetOrAddReal("problem", "pert_kr", 0.0);
  pert_kz = pin->GetOrAddReal("problem", "pert_kz", 0.0);

  // Enroll boundary functions
  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, InflowBoundary);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, FixedBoundary);

    EnrollUserRadSourceFunction(inner_boundary_source_function);

  //EnrollUserExplicitSourceFunction(electron_update);

  return;
}

//----------------------------------------------------------------------------------------
// Function for preparing MeshBlock
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)
// Notes:
//   user arrays are metric and its inverse

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {



    dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(FLT_MIN)));
    pfloor=pin->GetOrAddReal("hydro","pfloor",(1024*(FLT_MIN)));

    gm1 = peos->GetGamma() - 1.0;
    gamma_adi = peos->GetGamma();
    ge = 4.0/3.0;
    gem1 = ge-1.0;
  if (MAGNETIC_FIELDS_ENABLED) {
    AllocateUserOutputVariables(2);
    SetUserOutputVariableName(0, "gamma");
    SetUserOutputVariableName(1, "pmag");
  } else {
    AllocateUserOutputVariables(1);
    SetUserOutputVariableName(0, "gamma");
  }
  AllocateRealUserMeshBlockDataField(2);
  ruser_meshblock_data[0].NewAthenaArray(NMETRIC, ie+1);
  ruser_meshblock_data[1].NewAthenaArray(NMETRIC, ie+1);


  
  a = pcoord->GetSpin();
  m = pcoord->GetMass();
  Real rh = m * ( 1.0 + std::sqrt(1.0-SQR(a)) );

  r_inner_boundary = rh/2.0;

  return;
}

//----------------------------------------------------------------------------------------
// Function for setting initial conditions
// Inputs:
//   pin: parameters
// Outputs: (none)
// Notes:
//   initializes Fishbone-Moncrief torus
//     sets both primitive and conserved variables
//   defines and enrolls fixed r- and theta-direction boundary conditions
//   references Fishbone & Moncrief 1976, ApJ 207 962 (FM)
//              Fishbone 1977, ApJ 215 323 (F)
//   assumes x3 is axisymmetric direction

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Prepare index bounds
  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;
  if (block_size.nx2 > 1) {
    jl -= NGHOST;
    ju += NGHOST;
  }
  int kl = ks;
  int ku = ke;
  if (block_size.nx3 > 1) {
    kl -= NGHOST;
    ku += NGHOST;
  }

  // Get mass and spin of black hole
  m = pcoord->GetMass();
  a = pcoord->GetSpin();

  // Get ratio of specific heats
  gamma_adi = peos->GetGamma();

  // Reset whichever of l,r_peak is not specified
  if (r_peak >= 0.0) {
    l = CalculateLFromRPeak(r_peak);
  } else {
    r_peak = CalculateRPeakFromL(l);
  }

  // Prepare global constants describing primitives
  log_h_edge = LogHAux(r_edge, 1.0);
  log_h_peak = LogHAux(r_peak, 1.0) - log_h_edge;
  pgas_over_rho_peak = (gamma_adi-1.0)/gamma_adi * (std::exp(log_h_peak)-1.0);
  rho_peak = std::pow(pgas_over_rho_peak/k_adi, 1.0/(gamma_adi-1.0)) / rho_max;

  // Prepare scratch arrays
  AthenaArray<Real> g, gi;
  g.NewAthenaArray(NMETRIC, iu+1);
  gi.NewAthenaArray(NMETRIC, iu+1);

  // Initialize primitive values
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      pcoord->CellMetric(k, j, il, iu, g, gi);
      for (int i=il; i<=iu; ++i) {
        // Calculate Boyer-Lindquist coordinates of cell
        Real r(0.0), theta(0.0), phi(0.0);
        GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &r,
                                     &theta, &phi);
        Real sin_theta = std::sin(theta);
        Real cos_theta = std::cos(theta);
        Real sin_phi = std::sin(phi);
        Real cos_phi = std::cos(phi);

        // Account for tilt
        Real sin_vartheta, cos_vartheta, varphi;
        if (psi != 0.0) {
          Real x = sin_theta * cos_phi;
          Real y = sin_theta * sin_phi;
          Real z = cos_theta;
          Real varx = cos_psi * x - sin_psi * z;
          Real vary = y;
          Real varz = sin_psi * x + cos_psi * z;
          sin_vartheta = std::sqrt(SQR(varx) + SQR(vary));
          cos_vartheta = varz;
          varphi = std::atan2(vary, varx);
        } else {
          sin_vartheta = std::abs(sin_theta);
          cos_vartheta = cos_theta;
          varphi = (sin_theta < 0.0) ? phi-PI : phi;
        }
        // unused variables:
        // Real sin_varphi = std::sin(varphi);
        // Real cos_varphi = std::cos(varphi);

        // Determine if we are in the torus
        Real log_h;
        bool in_torus = false;
        if (r >= r_edge) {
          log_h = LogHAux(r, sin_vartheta) - log_h_edge;  // (FM 3.6)
          if (log_h >= 0.0) {
            in_torus = true;
          }
        }

        // Calculate background primitives
        Real rho = rho_min * std::pow(r, rho_pow);
        Real pgas = pgas_min * std::pow(r, pgas_pow);
        Real uu1 = 0.0;
        Real uu2 = 0.0;
        Real uu3 = 0.0;

        // Overwrite primitives inside torus
        if (in_torus) {
          // Calculate thermodynamic variables
          Real pgas_over_rho = (gamma_adi-1.0)/gamma_adi * (std::exp(log_h)-1.0);
          rho = std::pow(pgas_over_rho/k_adi, 1.0/(gamma_adi-1.0)) / rho_peak;
          pgas = pgas_over_rho * rho;

          // Calculate velocities in Boyer-Lindquist coordinates
          Real u0_bl, u1_bl, u2_bl, u3_bl;
          CalculateVelocityInTiltedTorus(r, theta, phi, &u0_bl, &u1_bl, &u2_bl, &u3_bl);

          // Transform to preferred coordinates
          Real u0, u1, u2, u3;
          TransformVector(u0_bl, 0.0, u2_bl, u3_bl, r, theta, phi, &u0, &u1, &u2, &u3);
          uu1 = u1 - gi(I01,i)/gi(I00,i) * u0;
          uu2 = u2 - gi(I02,i)/gi(I00,i) * u0;
          uu3 = u3 - gi(I03,i)/gi(I00,i) * u0;
        }

        // Set primitive values, including cylindrically symmetric radial velocity
        // perturbations
        Real rr = r * sin_vartheta;
        Real z = r * cos_vartheta;
        Real amp_rel = 0.0;
        if (in_torus) {
          amp_rel = pert_amp * std::sin(pert_kr*rr) * std::cos(pert_kz*z);
        }
        Real amp_abs = amp_rel * uu3;
        Real pert_uur = rr/r * amp_abs;
        Real pert_uutheta = cos_theta/r * amp_abs;
        phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = rho;
        phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = pgas;
        phydro->w(IVX,k,j,i) = phydro->w1(IM1,k,j,i) = uu1 + pert_uur;
        phydro->w(IVY,k,j,i) = phydro->w1(IM2,k,j,i) = uu2 + pert_uutheta;
        phydro->w(IVZ,k,j,i) = phydro->w1(IM3,k,j,i) = uu3;
      }
    }
  }

  // Free scratch arrays

  // Initialize magnetic fields
  if (MAGNETIC_FIELDS_ENABLED) {
    // Determine limits of sample grid
    Real r_vals[8], theta_vals[8], phi_vals[8];
    for (int p = 0; p < 8; ++p) {
      Real x1_val = (p%2 == 0) ? x1_min : x1_max;
      Real x2_val = ((p/2)%2 == 0) ? x2_min : x2_max;
      Real x3_val = ((p/4)%2 == 0) ? x3_min : x3_max;
      GetBoyerLindquistCoordinates(x1_val, x2_val, x3_val, r_vals+p, theta_vals+p,
                                   phi_vals+p);
    }
    r_min = *std::min_element(r_vals, r_vals+8);
    r_max = *std::max_element(r_vals, r_vals+8);
    theta_min = *std::min_element(theta_vals, theta_vals+8);
    theta_max = *std::max_element(theta_vals, theta_vals+8);
    phi_min = *std::min_element(phi_vals, phi_vals+8);
    phi_max = *std::max_element(phi_vals, phi_vals+8);

    // Prepare arrays of vector potential values
    AthenaArray<Real> a_phi_edges, a_phi_cells;
    AthenaArray<Real> a_theta_0, a_theta_1, a_theta_2, a_theta_3;
    AthenaArray<Real> a_phi_0, a_phi_1, a_phi_2, a_phi_3;
    if (field_config != MagneticFieldConfigs::vertical) {
      if (psi == 0.0) {
        a_phi_edges.NewAthenaArray(ju+2, iu+2);
        a_phi_cells.NewAthenaArray(ju+1, iu+1);
      } else {
        a_theta_0.NewAthenaArray(ku+1, ju+1, iu+1);
        a_theta_1.NewAthenaArray(ku+2, ju+2, iu+1);
        a_theta_2.NewAthenaArray(ku+2, ju+1, iu+2);
        a_theta_3.NewAthenaArray(ku+1, ju+2, iu+2);
        a_phi_0.NewAthenaArray(ku+1, ju+1, iu+1);
        a_phi_1.NewAthenaArray(ku+2, ju+2, iu+1);
        a_phi_2.NewAthenaArray(ku+2, ju+1, iu+2);
        a_phi_3.NewAthenaArray(ku+1, ju+2, iu+2);
      }
    }
    Real normalization;

    // Calculate vector potential in normal case
    if (field_config == MagneticFieldConfigs::normal) {
      // Calculate edge-centered vector potential values for untilted disks
      if (psi == 0.0) {
        for (int j=jl; j<=ju+1; ++j) {
          for (int i=il; i<=iu+1; ++i) {
            Real r(0.0), theta(0.0), phi(0.0);
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(kl),
                                         &r, &theta, &phi);
            if (r >= r_edge) {
              Real log_h = LogHAux(r, std::sin(theta)) - log_h_edge;  // (FM 3.6)
              if (log_h >= 0.0) {
                Real pgas_over_rho = (gamma_adi-1.0)/gamma_adi * (std::exp(log_h)-1.0);
                Real rho = std::pow(pgas_over_rho/k_adi, 1.0/(gamma_adi-1.0)) / rho_peak;
                Real rho_cutoff = std::max(rho-potential_cutoff, static_cast<Real>(0.0));
                a_phi_edges(j,i) = std::pow(r, potential_r_pow)
                                   * std::pow(rho_cutoff, potential_rho_pow);
              }
            }
          }
        }
      }

      // Calculate cell-centered vector potential values for untilted disks
      if (psi == 0.0) {
        for (int j=jl; j<=ju; ++j) {
          for (int i=il; i<=iu; ++i) {
            Real r(0.0), theta(0.0), phi(0.0);
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(kl),
                                         &r, &theta, &phi);
            if (r >= r_edge) {
              Real log_h = LogHAux(r, std::sin(theta)) - log_h_edge;  // (FM 3.6)
              if (log_h >= 0.0) {
                Real pgas_over_rho = (gamma_adi-1.0)/gamma_adi * (std::exp(log_h)-1.0);
                Real rho = std::pow(pgas_over_rho/k_adi, 1.0/(gamma_adi-1.0)) / rho_peak;
                Real rho_cutoff = std::max(rho-potential_cutoff, static_cast<Real>(0.0));
                a_phi_cells(j,i) = std::pow(r, potential_r_pow)
                                   * std::pow(rho_cutoff, potential_rho_pow);
              }
            }
          }
        }
      }

      // Calculate A_theta and A_phi for tilted disks
      if (psi != 0.0) {
        for (int k=kl; k<=ku+1; ++k) {
          for (int j=jl; j<=ju+1; ++j) {
            for (int i=il; i<=iu+1; ++i) {
              Real r_vals[4], theta_vals[4], phi_vals[4];
              if (i != iu+1 && j != ju+1 && k != ku+1) {
                GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j),
                                             pcoord->x3v(k), r_vals+0,
                                             theta_vals+0, phi_vals+0);
              }
              if (i != iu+1) {
                GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2f(j),
                                             pcoord->x3f(k), r_vals+1,
                                             theta_vals+1, phi_vals+1);
              }
              if (j != ju+1) {
                GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2v(j),
                                             pcoord->x3f(k), r_vals+2,
                                             theta_vals+2, phi_vals+2);
              }
              if (k != ku+1) {
                GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2f(j),
                                             pcoord->x3v(k), r_vals+3,
                                             theta_vals+3, phi_vals+3);
              }
              for (int p = 0; p < 4; ++p) {
                if ((p == 0 && (i == iu+1 || j == ju+1 || k == ku+1))
                    || (p == 1 && i == iu+1) || (p == 2 && j == ju+1)
                    || (p == 3 && k == ku+1)) {
                  continue;
                }
                if (r_vals[p] < r_edge) {
                  continue;
                }
                Real sin_theta = std::sin(theta_vals[p]);
                Real cos_theta = std::cos(theta_vals[p]);
                Real sin_phi = std::sin(phi_vals[p]);
                Real cos_phi = std::cos(phi_vals[p]);
                Real x = sin_theta * cos_phi;
                Real y = sin_theta * sin_phi;
                Real z = cos_theta;
                Real varx = cos_psi * x - sin_psi * z;
                Real vary = y;
                Real sin_vartheta = std::sqrt(SQR(varx) + SQR(vary));
                // unused variables:
                //Real varphi = std::atan2(vary, varx);
                //Real sin_varphi = std::sin(varphi);
                Real log_h = LogHAux(r_vals[p], sin_vartheta) - log_h_edge;  // (FM 3.6)
                if (!(log_h >= 0.0)) {
                  continue;
                }
                Real pgas_over_rho = (gamma_adi-1.0)/gamma_adi * (std::exp(log_h)-1.0);
                Real rho = std::pow(pgas_over_rho/k_adi, 1.0/(gamma_adi-1.0)) / rho_peak;
                Real rho_cutoff = std::max(rho-potential_cutoff, static_cast<Real>(0.0));
                Real a_varphi = std::pow(r_vals[p], potential_r_pow)
                                * std::pow(rho_cutoff, potential_rho_pow);
                Real dvarphi_dtheta = -sin_psi * sin_phi / SQR(sin_vartheta);
                Real dvarphi_dphi = sin_theta / SQR(sin_vartheta)
                                    * (cos_psi * sin_theta
                                       - sin_psi * cos_theta * cos_phi);
                switch (p) {
                  case 0:
                    a_theta_0(k,j,i) = dvarphi_dtheta * a_varphi;
                    a_phi_0(k,j,i) = dvarphi_dphi * a_varphi;
                    break;
                  case 1:
                    a_theta_1(k,j,i) = dvarphi_dtheta * a_varphi;
                    a_phi_1(k,j,i) = dvarphi_dphi * a_varphi;
                    break;
                  case 2:
                    a_theta_2(k,j,i) = dvarphi_dtheta * a_varphi;
                    a_phi_2(k,j,i) = dvarphi_dphi * a_varphi;
                    break;
                  case 3:
                    a_theta_3(k,j,i) = dvarphi_dtheta * a_varphi;
                    a_phi_3(k,j,i) = dvarphi_dphi * a_varphi;
                    break;
                }
              }
            }
          }
        }
      }
      // Calculate magnetic field normalization
      if (beta_min < 0.0) {
        normalization = 0.0;
      } else {
        Real beta_min_actual = CalculateBetaMin();
        normalization = std::sqrt(beta_min_actual/beta_min);
      }
      // Calculate vector potential in renormalized case
    } else if (field_config == MagneticFieldConfigs::renorm) {
      // Check that this is not a tilted disk
      if (psi != 0.0) {
        std::stringstream msg;
        msg << "### FATAL ERROR in Problem Generator\n"
            << "tilted disks cannot use field_config=renorm" << std::endl;
        ATHENA_ERROR(msg);
      }

      // Prepare global 2D sample arrays for integrating
      AthenaArray<Real> r_face, r_cell, theta_face, theta_cell;
      r_face.NewAthenaArray(sample_n_r+1);
      r_cell.NewAthenaArray(sample_n_r);
      theta_face.NewAthenaArray(sample_n_theta/2+1);
      theta_cell.NewAthenaArray(sample_n_theta/2);
      AthenaArray<Real> a_phi_global_edges, a_phi_global_cells;
      a_phi_global_edges.NewAthenaArray(sample_n_theta/2+1, sample_n_r+1);
      a_phi_global_cells.NewAthenaArray(sample_n_theta/2, sample_n_r);
      AthenaArray<Real> bbr_r_faces, bbr_theta_faces;
      bbr_r_faces.NewAthenaArray(sample_n_theta/2, sample_n_r+1);
      bbr_theta_faces.NewAthenaArray(sample_n_theta/2+1, sample_n_r);

      // Calculate r values
      Real delta_r;
      for (int i = 0; i < sample_n_r; ++i) {
        if (i == 0) {
          r_face(i) = r_min;
          Real ratio_power = 1.0;
          Real ratio_sum = 1.0;
          for (int ii = 1; ii < sample_n_r; ++ii) {
            ratio_power *= sample_r_rat;
            ratio_sum += ratio_power;
          }
          delta_r = (r_max-r_min) / ratio_sum;
        } else {
          delta_r *= sample_r_rat;
        }
        r_face(i+1) = r_face(i) + delta_r;
        r_cell(i) = 0.5 * (r_face(i) + r_face(i+1));
      }

      // Calculate theta values
      for (int j = 0; j < sample_n_theta/2; ++j) {
        if (j == 0) {
          theta_face(j) = theta_min;
        }
        theta_face(j+1) = theta_min
                          + static_cast<Real>(j+1)/static_cast<Real>(sample_n_theta)
                          * (theta_max-theta_min);
        theta_cell(j) = 0.5 * (theta_face(j) + theta_face(j+1));
      }

      // Calculate edge-centered A_phi based on radius and density
      for (int j = 0; j < sample_n_theta/2+1; ++j) {
        for (int i = 0; i < sample_n_r+1; ++i) {
          Real r(0.0), theta(0.0), phi(0.0);
          GetBoyerLindquistCoordinates(r_face(i), theta_face(j), pcoord->x3v(kl), &r,
                                       &theta, &phi);
          Real rho = 0.0;
          if (r >= r_edge) {
            Real log_h = LogHAux(r, std::sin(theta)) - log_h_edge;  // (FM 3.6)
            if (log_h >= 0.0) {
              Real pgas_over_rho = (gamma_adi-1.0)/gamma_adi * (std::exp(log_h)-1.0);
              rho = std::pow(pgas_over_rho/k_adi, 1.0/(gamma_adi-1.0)) / rho_peak;
            }
          }
          a_phi_global_edges(j,i) =
              std::pow(r, potential_r_pow) * std::pow(rho, potential_rho_pow);
        }
      }

      // Calculate cell-centered A_phi based on radius and density
      for (int j = 0; j < sample_n_theta/2; ++j) {
        for (int i = 0; i < sample_n_r; ++i) {
          Real r(0.0), theta(0.0), phi(0.0);
          GetBoyerLindquistCoordinates(r_cell(i), theta_cell(j), pcoord->x3v(kl), &r,
                                       &theta, &phi);
          Real rho = 0.0;
          if (r >= r_edge) {
            Real log_h = LogHAux(r, std::sin(theta)) - log_h_edge;  // (FM 3.6)
            if (log_h >= 0.0) {
              Real pgas_over_rho = (gamma_adi-1.0)/gamma_adi * (std::exp(log_h)-1.0);
              rho = std::pow(pgas_over_rho/k_adi, 1.0/(gamma_adi-1.0)) / rho_peak;
            }
          }
          a_phi_global_cells(j,i) =
              std::pow(r, potential_r_pow) * std::pow(rho, potential_rho_pow);
        }
      }

      // Calculate r-face-centered B^r based on edge-centered A_phi and pgas
      for (int j = 0; j < sample_n_theta/2; ++j) {
        for (int i = 1; i < sample_n_r; ++i) {
          Real r_m = r_cell(i-1);
          Real r_c = r_face(i);
          Real r_p = r_cell(i);
          Real theta_m = theta_face(j);
          Real theta_c = theta_cell(j);
          Real theta_p = theta_face(j+1);
          Real cos_theta = std::cos(theta_c);
          Real det = (SQR(r_c) + SQR(a) * SQR(cos_theta)) * std::abs(std::sin(theta_c));
          Real a_phi_cm = a_phi_global_edges(j,i);
          Real a_phi_cp = a_phi_global_edges(j+1,i);
          Real a_phi_mc = a_phi_global_cells(j,i-1);
          Real a_phi_pc = a_phi_global_cells(j,i);
          Real bbr = 1.0/det * (a_phi_cp-a_phi_cm) / (theta_p-theta_m);
          Real beta;
          bool value_set = CalculateBetaFromA(r_m, r_c, r_p, theta_m, theta_p, theta_c,
                                              a_phi_cm, a_phi_cp, a_phi_mc, a_phi_pc,
                                              &beta);
          if (value_set) {
            bbr_r_faces(j,i) = bbr * std::sqrt(beta);
          }
        }
      }

      // Calculate theta-face-centered B^r based on cell-centered A_phi and pgas
      for (int j = 1; j < sample_n_theta/2; ++j) {
        for (int i = 0; i < sample_n_r; ++i) {
          Real r_m = r_face(i);
          Real r_c = r_cell(i);
          Real r_p = r_face(i+1);
          Real theta_m = theta_cell(j-1);
          Real theta_c = theta_face(j);
          Real theta_p = theta_cell(j);
          Real cos_theta = std::cos(theta_c);
          Real det = (SQR(r_c) + SQR(a) * SQR(cos_theta)) * std::abs(std::sin(theta_c));
          Real a_phi_cm = a_phi_global_cells(j-1,i);
          Real a_phi_cp = a_phi_global_cells(j,i);
          Real a_phi_mc = a_phi_global_edges(j,i);
          Real a_phi_pc = a_phi_global_edges(j,i+1);
          Real bbr = 1.0/det * (a_phi_cp-a_phi_cm) / (theta_p-theta_m);
          Real beta;
          bool value_set = CalculateBetaFromA(r_m, r_c, r_p, theta_m, theta_p, theta_c,
                                              a_phi_cm, a_phi_cp, a_phi_mc, a_phi_pc,
                                              &beta);
          if (value_set) {
            bbr_theta_faces(j,i) = bbr * std::sqrt(beta);
          }
        }
      }

      // Calculate edge-centered A_phi based on r-face-centered B^r
      for (int i = 0; i < sample_n_r+1; ++i) {
        Real r = r_face(i);
        a_phi_global_edges(0,i) = 0.0;
        for (int j = 1; j < sample_n_theta/2+1; ++j) {
          Real theta_m = theta_face(j-1);
          Real theta_c = theta_cell(j-1);
          Real theta_p = theta_face(j);
          Real cos_theta = std::cos(theta_c);
          Real det = (SQR(r) + SQR(a) * SQR(cos_theta)) * std::abs(std::sin(theta_c));
          a_phi_global_edges(j,i) = a_phi_global_edges(j-1,i)
                                    + bbr_r_faces(j-1,i) * det * (theta_p-theta_m);
        }
      }

      // Calculate cell-centered A_phi based on theta-face-centered B^r
      for (int i = 0; i < sample_n_r; ++i) {
        Real r = r_cell(i);
        a_phi_global_cells(0,i) = 0.0;
        for (int j = 1; j < sample_n_theta/2; ++j) {
          Real theta_m = theta_cell(j-1);
          Real theta_c = theta_face(j);
          Real theta_p = theta_cell(j);
          Real cos_theta = std::cos(theta_c);
          Real det = (SQR(r) + SQR(a) * SQR(cos_theta)) * std::abs(std::sin(theta_c));
          a_phi_global_cells(j,i) = a_phi_global_cells(j-1,i)
                                    + bbr_theta_faces(j,i) * det * (theta_p-theta_m);
        }
      }

      // Calculate maximum of vector potential
      Real a_phi_max = 0.0;
      for (int i = 0; i < sample_n_r+1; ++i) {
        for (int j = 0; j < sample_n_theta/2+1; ++j) {
          a_phi_max = std::max(a_phi_max, a_phi_global_edges(j,i));
        }
      }

      // Floor edge-centered A_phi
      for (int i = 0; i < sample_n_r+1; ++i) {
        for (int j = 0; j < sample_n_theta/2+1; ++j) {
          a_phi_global_edges(j,i) = std::max(a_phi_global_edges(j,i),
                                             potential_cutoff*a_phi_max);
        }
      }

      // Floor cell-centered A_phi
      for (int i = 0; i < sample_n_r; ++i) {
        for (int j = 0; j < sample_n_theta/2; ++j) {
          a_phi_global_cells(j,i) = std::max(a_phi_global_cells(j,i),
                                             potential_cutoff*a_phi_max);
        }
      }

      // Calculate minimum value of beta over global sample grid
      Real beta_min_actual = std::numeric_limits<Real>::max();
      for (int j = 0; j < sample_n_theta/2; ++j) {
        for (int i = 0; i < sample_n_r; ++i) {
          Real r_m = r_face(i);
          Real r_c = r_cell(i);
          Real r_p = r_face(i+1);
          Real theta_m = theta_face(j);
          Real theta_c = theta_cell(j);
          Real theta_p = theta_face(j+1);
          if (r_m < r_edge) {
            continue;
          }
          Real log_h = LogHAux(r_c, std::sin(theta_c)) - log_h_edge;
          Real pgas_over_rho = (gamma_adi-1.0)/gamma_adi * (std::exp(log_h)-1.0);
          Real rho = std::pow(pgas_over_rho/k_adi, 1.0/(gamma_adi-1.0)) / rho_peak;
          if (rho < sample_cutoff) {
            continue;
          }
          Real a_phi_cm = 0.5 * (a_phi_global_edges(j,i) + a_phi_global_edges(j,i+1));
          Real a_phi_cp = 0.5 * (a_phi_global_edges(j+1,i) + a_phi_global_edges(j+1,i+1));
          Real a_phi_mc = 0.5 * (a_phi_global_edges(j,i) + a_phi_global_edges(j+1,i));
          Real a_phi_pc = 0.5 * (a_phi_global_edges(j,i+1) + a_phi_global_edges(j+1,i+1));
          Real beta;
          bool value_set = CalculateBetaFromA(r_m, r_c, r_p, theta_m, theta_p, theta_c,
                                              a_phi_cm, a_phi_cp, a_phi_mc, a_phi_pc,
                                              &beta);
          if (value_set) {
            beta_min_actual = std::min(beta_min_actual, beta);
          }
        }
      }

      // Calculate magnetic field normalization
      if (beta_min < 0.0) {
        normalization = 0.0;
      } else {
        normalization = std::sqrt(beta_min_actual/beta_min);
      }

      // Interpolate edge-centered vector potential onto local grid
      for (int j=jl; j<=ju+1; ++j) {
        for (int i=il; i<=iu+1; ++i) {
          Real r_c(0.0), theta_c(0.0), phi(0.0);
          GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(kl),
                                       &r_c, &theta_c, &phi);
          if (theta_c > PI/2.0) {
            theta_c = PI - theta_c;
          }
          int r_index;
          for (r_index = 0; r_index < sample_n_r; ++r_index) {
            if (r_face(r_index+1) > r_c) {
              break;
            }
          }
          Real r_frac;
          if (r_index == 0) {
            r_frac = 0.0;
          } else if (r_index == sample_n_r) {
            r_index = sample_n_r - 1;
            r_frac = 1.0;
          } else {
            r_frac = (r_c-r_face(r_index)) / (r_face(r_index+1)-r_face(r_index));
          }
          int theta_index;
          for (theta_index = 0; theta_index < sample_n_theta/2; ++theta_index) {
            if (theta_face(theta_index+1) > theta_c) {
              break;
            }
          }
          Real theta_frac;
          if (theta_index == 0) {
            theta_frac = 0.0;
          } else if (theta_index == sample_n_theta/2) {
            theta_index = sample_n_theta/2 - 1;
            theta_frac = 1.0;
          } else {
            theta_frac = (theta_c-theta_face(theta_index))
                         / (theta_face(theta_index+1)-theta_face(theta_index));
          }
          Real a_mm = a_phi_global_edges(theta_index,r_index);
          Real a_mp = a_phi_global_edges(theta_index+1,r_index);
          Real a_pm = a_phi_global_edges(theta_index,r_index+1);
          Real a_pp = a_phi_global_edges(theta_index+1,r_index+1);
          a_phi_edges(j,i) = (1.0-r_frac-theta_frac+r_frac*theta_frac) * a_mm
                             + theta_frac*(1.0-r_frac) * a_mp
                             + r_frac*(1.0-theta_frac) * a_pm
                             + r_frac*theta_frac * a_pp;
        }
      }

      // Interpolate cell-centered vector potential onto local grid
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          Real r_c(0.0), theta_c(0.0), phi(0.0);
          GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(kl),
                                       &r_c, &theta_c, &phi);
          if (theta_c > PI/2.0) {
            theta_c = PI - theta_c;
          }
          int r_index;
          for (r_index = 0; r_index < sample_n_r-1; ++r_index) {
            if (r_cell(r_index+1) > r_c) {
              break;
            }
          }
          Real r_frac;
          if (r_index == 0) {
            r_frac = 0.0;
          } else if (r_index == sample_n_r-1) {
            r_index = sample_n_r - 2;
            r_frac = 1.0;
          } else {
            r_frac = (r_c-r_cell(r_index)) / (r_cell(r_index+1)-r_cell(r_index));
          }
          int theta_index;
          for (theta_index = 0; theta_index < sample_n_theta/2-1; ++theta_index) {
            if (theta_cell(theta_index+1) > theta_c) {
              break;
            }
          }
          Real theta_frac;
          if (theta_index == 0) {
            theta_frac = 0.0;
          } else if (theta_index == sample_n_theta/2-1) {
            theta_index = sample_n_theta/2 - 2;
            theta_frac = 1.0;
          } else {
            theta_frac = (theta_c-theta_cell(theta_index))
                         / (theta_cell(theta_index+1)-theta_cell(theta_index));
          }
          Real a_mm = a_phi_global_cells(theta_index,r_index);
          Real a_mp = a_phi_global_cells(theta_index+1,r_index);
          Real a_pm = a_phi_global_cells(theta_index,r_index+1);
          Real a_pp = a_phi_global_cells(theta_index+1,r_index+1);
          a_phi_cells(j,i) = (1.0-r_frac-theta_frac+r_frac*theta_frac) * a_mm
                             + theta_frac*(1.0-r_frac) * a_mp
                             + r_frac*(1.0-theta_frac) * a_pm
                             + r_frac*theta_frac * a_pp;
        }
      }

      // Free global arrays

      // Calculate normalization in vertical case
    } else if (field_config == MagneticFieldConfigs::vertical) {
      // Calculate magnetic field normalization
      if (beta_min < 0.0) {
        normalization = 0.0;
      } else {
        Real beta_min_actual = CalculateBetaMin();
        normalization = std::sqrt(beta_min_actual/beta_min);
      }
      // Handle unknown input
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in Problem Generator\n"
          << "field_config must be \"normal\", \"renorm\", or \"vertical\"" << std::endl;
      ATHENA_ERROR(msg);
    }

    // Set magnetic fields according to vector potential in vertical case
    if (field_config == MagneticFieldConfigs::vertical) {
      // Set B^1
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
          for (int i=il; i<=iu+1; ++i) {
            Real r(0.0), theta(0.0), phi(0.0);
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3v(k),
                                         &r, &theta, &phi);
            Real sin_theta = std::sin(theta);
            Real cos_theta = std::cos(theta);
            Real rr = r * sin_theta;
            Real z = r * cos_theta;
            Real det = (SQR(r) + SQR(a) * SQR(cos_theta)) * sin_theta;
            Real bbr = rr * z / det;
            Real bbtheta = -SQR(rr) / (r * det);
            if (rr < r_edge || det == 0.0 || (bbr == 0.0 && bbtheta == 0.0)) {
              pfield->b.x1f(k,j,i) = 0.0;
            } else {
              Real ut, uphi;
              CalculateVelocityInTorus(r, sin_theta, &ut, &uphi);
              Real br = 1.0/ut * bbr;
              Real btheta = 1.0/ut * bbtheta;
              Real u0, u1, u2, u3;
              TransformVector(ut, 0.0, 0.0, uphi, r, theta, phi, &u0, &u1, &u2, &u3);
              Real b0, b1, b2, b3;
              TransformVector(0.0, br, btheta, 0.0, r, theta, phi, &b0, &b1, &b2, &b3);
              pfield->b.x1f(k,j,i) = (b1 * u0 - b0 * u1) * normalization;
            }
          }
        }
      }

      // Set B^2
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju+1; ++j) {
          for (int i=il; i<=iu; ++i) {
            Real r(0.0), theta(0.0), phi(0.0);
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2f(j), pcoord->x3v(k),
                                         &r, &theta, &phi);
            Real sin_theta = std::sin(theta);
            Real cos_theta = std::cos(theta);
            Real rr = r * sin_theta;
            Real z = r * cos_theta;
            Real det = (SQR(r) + SQR(a) * SQR(cos_theta)) * sin_theta;
            Real bbr = rr * z / det;
            Real bbtheta = -SQR(rr) / (r * det);
            if (rr < r_edge || det == 0.0 || (bbr == 0.0 && bbtheta == 0.0)) {
              pfield->b.x2f(k,j,i) = 0.0;
            } else {
              Real ut, uphi;
              CalculateVelocityInTorus(r, sin_theta, &ut, &uphi);
              Real br = 1.0/ut * bbr;
              Real btheta = 1.0/ut * bbtheta;
              Real u0, u1, u2, u3;
              TransformVector(ut, 0.0, 0.0, uphi, r, theta, phi, &u0, &u1, &u2, &u3);
              Real b0, b1, b2, b3;
              TransformVector(0.0, br, btheta, 0.0, r, theta, phi, &b0, &b1, &b2, &b3);
              pfield->b.x2f(k,j,i) = (b2 * u0 - b0 * u2) * normalization;
            }
          }
        }
      }

      // Set B^3
      for (int k=kl; k<=ku+1; ++k) {
        for (int j=jl; j<=ju; ++j) {
          for (int i=il; i<=iu; ++i) {
            pfield->b.x3f(k,j,i) = 0.0;
          }
        }
      }

      // Set magnetic fields according to vector potential for untilted disks
    } else if (psi == 0.0) {
      // Set B^1
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
          for (int i=il; i<=iu+1; ++i) {
            Real r(0.0), theta(0.0), phi(0.0);
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3v(k),
                                         &r, &theta, &phi);
            Real r_1(0.0), theta_1(0.0), phi_1(0.0);
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k),
                                         &r_1, &theta_1, &phi_1);
            Real r_2(0.0), theta_2(0.0), phi_2(0.0);
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2f(j+1),
                                         pcoord->x3v(k), &r_2, &theta_2, &phi_2);
            Real cos_theta = std::cos(theta);
            Real det = (SQR(r) + SQR(a) * SQR(cos_theta)) * std::abs(std::sin(theta));
            Real bbr =
                1.0/det * (a_phi_edges(j+1,i)-a_phi_edges(j,i)) / (theta_2-theta_1);
            Real a_phi_1, a_phi_2;
            if (i == il) {
              a_phi_1 = 0.5 * (a_phi_edges(j,i) + a_phi_edges(j+1,i));
              a_phi_2 = a_phi_cells(j,i);
              r_1 = r;
              GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j),
                                           pcoord->x3v(k), &r_2, &theta_2, &phi_2);
            } else if (i == iu+1) {
              a_phi_1 = a_phi_cells(j,i-1);
              a_phi_2 = 0.5 * (a_phi_edges(j,i) + a_phi_edges(j+1,i));
              GetBoyerLindquistCoordinates(pcoord->x1v(i-1), pcoord->x2v(j),
                                           pcoord->x3v(k), &r_1, &theta_1, &phi_1);
              r_2 = r;
            } else {
              a_phi_1 = a_phi_cells(j,i-1);
              a_phi_2 = a_phi_cells(j,i);
              GetBoyerLindquistCoordinates(pcoord->x1v(i-1), pcoord->x2v(j),
                                           pcoord->x3v(k), &r_1, &theta_1, &phi_1);
              GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j),
                                           pcoord->x3v(k), &r_2, &theta_2, &phi_2);
            }
            Real bbtheta = -1.0/det * (a_phi_2 - a_phi_1) / (r_2 - r_1);
            if (det == 0.0 || (bbr == 0.0 && bbtheta == 0.0)) {
              pfield->b.x1f(k,j,i) = 0.0;
            } else {
              Real ut, uphi;
              Real sin_theta = std::sin(theta);
              CalculateVelocityInTorus(r, sin_theta, &ut, &uphi);
              Real br = 1.0/ut * bbr;
              Real btheta = 1.0/ut * bbtheta;
              Real u0, u1, u2, u3;
              TransformVector(ut, 0.0, 0.0, uphi, r, theta, phi, &u0, &u1, &u2, &u3);
              Real b0, b1, b2, b3;
              TransformVector(0.0, br, btheta, 0.0, r, theta, phi, &b0, &b1, &b2, &b3);
              pfield->b.x1f(k,j,i) = (b1 * u0 - b0 * u1) * normalization;
            }
          }
        }
      }

      // Set B^2
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju+1; ++j) {
          for (int i=il; i<=iu; ++i) {
            Real r(0.0), theta(0.0), phi(0.0);
            Real r_1(0.0), theta_1(0.0), phi_1(0.0);
            Real r_2(0.0), theta_2(0.0), phi_2(0.0);
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2f(j), pcoord->x3v(k),
                                         &r, &theta, &phi);
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k),
                                         &r_1, &theta_1, &phi_1);
            GetBoyerLindquistCoordinates(pcoord->x1f(i+1), pcoord->x2f(j),
                                         pcoord->x3v(k), &r_2, &theta_2, &phi_2);
            Real cos_theta = std::cos(theta);
            Real det = (SQR(r) + SQR(a) * SQR(cos_theta)) * std::abs(std::sin(theta));
            Real bbtheta = -1.0/det * (a_phi_edges(j,i+1)-a_phi_edges(j,i)) / (r_2-r_1);
            Real a_phi_1, a_phi_2;
            if (j == jl) {
              a_phi_1 = 0.5 * (a_phi_edges(j,i) + a_phi_edges(j,i+1));
              a_phi_2 = a_phi_cells(j,i);
              theta_1 = theta;
              GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j),
                                           pcoord->x3v(k), &r_2, &theta_2, &phi_2);
            } else if (j == ju+1) {
              a_phi_1 = a_phi_cells(j-1,i);
              a_phi_2 = 0.5 * (a_phi_edges(j,i) + a_phi_edges(j,i+1));
              GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j-1),
                                           pcoord->x3v(k), &r_1, &theta_1, &phi_1);
              theta_2 = theta;
            } else {
              a_phi_1 = a_phi_cells(j-1,i);
              a_phi_2 = a_phi_cells(j,i);
              GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j-1),
                                           pcoord->x3v(k), &r_1, &theta_1, &phi_1);
              GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j),
                                           pcoord->x3v(k), &r_2, &theta_2, &phi_2);
            }
            Real bbr = 1.0/det * (a_phi_2 - a_phi_1) / (theta_2 - theta_1);
            if (det == 0.0 || (bbr == 0.0 && bbtheta == 0.0)) {
              pfield->b.x2f(k,j,i) = 0.0;
            } else {
              Real ut, uphi;
              Real sin_theta = std::sin(theta);
              CalculateVelocityInTorus(r, sin_theta, &ut, &uphi);
              Real br = 1.0/ut * bbr;
              Real btheta = 1.0/ut * bbtheta;
              Real u0, u1, u2, u3;
              TransformVector(ut, 0.0, 0.0, uphi, r, theta, phi, &u0, &u1, &u2, &u3);
              Real b0, b1, b2, b3;
              TransformVector(0.0, br, btheta, 0.0, r, theta, phi, &b0, &b1, &b2, &b3);
              pfield->b.x2f(k,j,i) = (b2 * u0 - b0 * u2) * normalization;
            }
          }
        }
      }

      // Set B^3
      for (int k=kl; k<=ku+1; ++k) {
        for (int j=jl; j<=ju; ++j) {
          for (int i=il; i<=iu; ++i) {
            pfield->b.x3f(k,j,i) = 0.0;
          }
        }
      }

      // Set magnetic fields according to vector potential for tilted disks
    } else {
      for (int k=kl+1; k<=ku; ++k) {
        for (int j=jl+1; j<=ju; ++j) {
          for (int i=il+1; i<=iu; ++i) {
            // Declare variables to hold coordinates and fields
            Real r, r_m, r_p, delta_r;
            Real theta, theta_m, theta_p, delta_theta;
            Real phi, phi_m, phi_p, delta_phi;
            Real det;
            Real bbr, bbtheta, bbphi;

            // Set B^1
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3v(k),
                                         &r, &theta, &phi);
            det = (SQR(r) + SQR(a) * SQR(std::cos(theta))) * std::abs(std::sin(theta));
            GetBoyerLindquistCoordinates(pcoord->x1v(i-1), pcoord->x2v(j), pcoord->x3v(k),
                                         &r_m, &theta_m, &phi_m);
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k),
                                         &r_p, &theta_p, &phi_p);
            delta_r = r_p - r_m;
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k),
                                         &r_m, &theta_m, &phi_m);
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2f(j+1), pcoord->x3v(k),
                                         &r_p, &theta_p, &phi_p);
            delta_theta = theta_p - theta_m;
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3f(k),
                                         &r_m, &theta_m, &phi_m);
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3f(k+1),
                                         &r_p, &theta_p, &phi_p);
            delta_phi = phi_p - phi_m;
            bbr = 1.0/det * ((a_phi_3(k,j+1,i)-a_phi_3(k,j,i))/delta_theta
                             - (a_theta_2(k+1,j,i)-a_theta_2(k,j,i))/delta_phi);
            bbtheta = -1.0/det * (a_phi_0(k,j,i)-a_phi_0(k,j,i-1))/delta_r;
            bbphi = 1.0/det * (a_theta_0(k,j,i)-a_theta_0(k,j,i-1))/delta_r;
            if (det == 0.0 || (bbr == 0.0 && bbtheta == 0.0 && bbphi == 0.0)) {
              pfield->b.x1f(k,j,i) = 0.0;
            } else {
              Real ut, ur, utheta, uphi;
              CalculateVelocityInTiltedTorus(r, theta, phi, &ut, &ur, &utheta, &uphi);
              Real sin_theta = std::sin(theta);
              Real cos_theta = std::cos(theta);
              Real delta = SQR(r) - 2.0*m*r + SQR(a);
              Real sigma = SQR(r) + SQR(a) * SQR(cos_theta);
              Real g_tphi = -2.0*m*a*r/sigma * SQR(sin_theta);
              Real g_rr = sigma/delta;
              Real g_thetatheta = sigma;
              Real g_phiphi = (SQR(r) + SQR(a) + 2.0*m*SQR(a)*r/sigma * SQR(sin_theta))
                              * SQR(sin_theta);
              Real bt = g_tphi*ut*bbphi + g_rr*ur*bbr + g_thetatheta*utheta*bbtheta
                        + g_phiphi*uphi*bbphi;
              Real br = (bbr + bt * ur) / ut;
              Real btheta = (bbtheta + bt * utheta) / ut;
              Real bphi = (bbphi + bt * uphi) / ut;
              Real u0, u1, u2, u3;
              TransformVector(ut, ur, utheta, uphi, r, theta, phi, &u0, &u1, &u2, &u3);
              Real b0, b1, b2, b3;
              TransformVector(bt, br, btheta, bphi, r, theta, phi, &b0, &b1, &b2, &b3);
              pfield->b.x1f(k,j,i) = (b1 * u0 - b0 * u1) * normalization;
            }

            // Set B^2
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2f(j), pcoord->x3v(k),
                                         &r, &theta, &phi);
            det = (SQR(r) + SQR(a) * SQR(std::cos(theta))) * std::abs(std::sin(theta));
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k),
                                         &r_m, &theta_m, &phi_m);
            GetBoyerLindquistCoordinates(pcoord->x1f(i+1), pcoord->x2f(j), pcoord->x3v(k),
                                         &r_p, &theta_p, &phi_p);
            delta_r = r_p - r_m;
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j-1), pcoord->x3v(k),
                                         &r_m, &theta_m, &phi_m);
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k),
                                         &r_p, &theta_p, &phi_p);
            delta_theta = theta_p - theta_m;
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2f(j), pcoord->x3f(k),
                                         &r_m, &theta_m, &phi_m);
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2f(j), pcoord->x3f(k+1),
                                         &r_p, &theta_p, &phi_p);
            delta_phi = phi_p - phi_m;
            bbr = 1.0/det * ((a_phi_0(k,j,i)-a_phi_0(k,j-1,i))/delta_theta
                             - (a_theta_1(k+1,j,i)-a_theta_1(k,j,i))/delta_phi);
            bbtheta = -1.0/det * (a_phi_3(k,j,i+1)-a_phi_3(k,j,i))/delta_r;
            bbphi = 1.0/det * (a_theta_3(k,j,i+1)-a_theta_3(k,j,i))/delta_r;
            if (det == 0.0 || (bbr == 0.0 && bbtheta == 0.0 && bbphi == 0.0)) {
              pfield->b.x2f(k,j,i) = 0.0;
            } else {
              Real ut, ur, utheta, uphi;
              CalculateVelocityInTiltedTorus(r, theta, phi, &ut, &ur, &utheta, &uphi);
              Real sin_theta = std::sin(theta);
              Real cos_theta = std::cos(theta);
              Real delta = SQR(r) - 2.0*m*r + SQR(a);
              Real sigma = SQR(r) + SQR(a) * SQR(cos_theta);
              Real g_tphi = -2.0*m*a*r/sigma * SQR(sin_theta);
              Real g_rr = sigma/delta;
              Real g_thetatheta = sigma;
              Real g_phiphi = (SQR(r) + SQR(a) + 2.0*m*SQR(a)*r/sigma * SQR(sin_theta))
                              * SQR(sin_theta);
              Real bt = g_tphi*ut*bbphi + g_rr*ur*bbr + g_thetatheta*utheta*bbtheta
                        + g_phiphi*uphi*bbphi;
              Real br = (bbr + bt * ur) / ut;
              Real btheta = (bbtheta + bt * utheta) / ut;
              Real bphi = (bbphi + bt * uphi) / ut;
              Real u0, u1, u2, u3;
              TransformVector(ut, ur, utheta, uphi, r, theta, phi, &u0, &u1, &u2, &u3);
              Real b0, b1, b2, b3;
              TransformVector(bt, br, btheta, bphi, r, theta, phi, &b0, &b1, &b2, &b3);
              pfield->b.x2f(k,j,i) = (b2 * u0 - b0 * u2) * normalization;
            }

            // Set B^3
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3f(k),
                                         &r, &theta, &phi);
            det = (SQR(r) + SQR(a) * SQR(std::cos(theta))) * std::abs(std::sin(theta));
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3f(k),
                                         &r_m, &theta_m, &phi_m);
            GetBoyerLindquistCoordinates(pcoord->x1f(i+1), pcoord->x2v(j), pcoord->x3f(k),
                                         &r_p, &theta_p, &phi_p);
            delta_r = r_p - r_m;
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2f(j), pcoord->x3f(k),
                                         &r_m, &theta_m, &phi_m);
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2f(j+1), pcoord->x3f(k),
                                         &r_p, &theta_p, &phi_p);
            delta_theta = theta_p - theta_m;
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k-1),
                                         &r_m, &theta_m, &phi_m);
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k),
                                         &r_p, &theta_p, &phi_p);
            delta_phi = phi_p - phi_m;
            bbr = 1.0/det * ((a_phi_1(k,j+1,i)-a_phi_1(k,j,i))/delta_theta
                             - (a_theta_0(k,j,i)-a_theta_0(k-1,j,i))/delta_phi);
            bbtheta = -1.0/det * (a_phi_2(k,j,i+1)-a_phi_2(k,j,i))/delta_r;
            bbphi = 1.0/det * (a_theta_2(k,j,i+1)-a_theta_2(k,j,i))/delta_r;
            if (det == 0.0 || (bbr == 0.0 && bbtheta == 0.0 && bbphi == 0.0)) {
              pfield->b.x3f(k,j,i) = 0.0;
            } else {
              Real ut, ur, utheta, uphi;
              CalculateVelocityInTiltedTorus(r, theta, phi, &ut, &ur, &utheta, &uphi);
              Real sin_theta = std::sin(theta);
              Real cos_theta = std::cos(theta);
              Real delta = SQR(r) - 2.0*m*r + SQR(a);
              Real sigma = SQR(r) + SQR(a) * SQR(cos_theta);
              Real g_tphi = -2.0*m*a*r/sigma * SQR(sin_theta);
              Real g_rr = sigma/delta;
              Real g_thetatheta = sigma;
              Real g_phiphi = (SQR(r) + SQR(a) + 2.0*m*SQR(a)*r/sigma * SQR(sin_theta))
                              * SQR(sin_theta);
              Real bt = g_tphi*ut*bbphi + g_rr*ur*bbr + g_thetatheta*utheta*bbtheta
                        + g_phiphi*uphi*bbphi;
              Real br = (bbr + bt * ur) / ut;
              Real btheta = (bbtheta + bt * utheta) / ut;
              Real bphi = (bbphi + bt * uphi) / ut;
              Real u0, u1, u2, u3;
              TransformVector(ut, ur, utheta, uphi, r, theta, phi, &u0, &u1, &u2, &u3);
              Real b0, b1, b2, b3;
              TransformVector(bt, br, btheta, bphi, r, theta, phi, &b0, &b1, &b2, &b3);
              pfield->b.x3f(k,j,i) = (b3 * u0 - b0 * u3) * normalization;
            }
          }
        }
      }
    }
  }

  // Impose density and pressure floors
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        Real r(0.0), theta(0.0), phi(0.0);
        GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(kl), &r,
                                     &theta, &phi);
        Real &rho = phydro->w(IDN,k,j,i);
        Real &pgas = phydro->w(IEN,k,j,i);
        rho = std::max(rho, rho_min * std::pow(r, rho_pow));
        pgas = std::max(pgas, pgas_min * std::pow(r, pgas_pow));
        phydro->w1(IDN,k,j,i) = rho;
        phydro->w1(IEN,k,j,i) = pgas;
      }
    }
  }

  // Calculate cell-centered magnetic field
  AthenaArray<Real> bb;
  if (MAGNETIC_FIELDS_ENABLED) {
    pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, il, iu, jl, ju, kl,
                                       ku);
  } else {
    bb.NewAthenaArray(3, ku+1, ju+1, iu+1);
  }

  // Initialize conserved values
  if (MAGNETIC_FIELDS_ENABLED) {
    peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, il, iu, jl, ju,
                               kl, ku);
  } else {
    peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, il, iu, jl, ju, kl, ku);
  }


  // Call user work function to set output variables

  //UserWorkInLoop();


  init_electrons(pscalars, phydro, pfield, il, iu, jl, ju, kl,ku);

  return;
}

//----------------------------------------------------------------------------------------
// Function responsible for storing useful quantities for output
// Inputs: (none)
// Outputs: (none)
// Notes:
//   writes to user_out_var array the following quantities:
//     0: gamma (normal-frame Lorentz factor)
//     1: p_mag (magnetic pressure)

void MeshBlock::UserWorkInLoop() {
  // Create aliases for metric
  AthenaArray<Real> &g = ruser_meshblock_data[0], &gi = ruser_meshblock_data[1];


  int il = is - NGHOST; int jl = js; int kl = ks;
  int iu = ie + NGHOST; int ju = je; int ku = ke;

  if (ncells2>1) {
    jl -= NGHOST; ju += NGHOST;
  }
  if (ncells3>1) {
    kl -= NGHOST; ku += NGHOST;
  }

  // Go through all cells
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pcoord->CellMetric(k, j, is, ie, g, gi);
      for (int i=is; i<=ie; ++i) {

          // set entropy
        //if (NSCALARS > 0) pscalars->s(0,k,j,i) = 1.0 * phydro->cons(IDN,k,j,i) * phydro->w(IPR,k,j,i) / std::pow(phydro->w(IDN,k,j,i),gamma_adi) ; //total
        //if (NSCALARS > 1) pscalars->s(1,k,j,i) = 0.1 * phydro->cons(IDN,k,j,i) * phydro->w(IPR,k,j,i) / std::pow(phydro->w(IDN,k,j,i),ge); //electron

        // Calculate normal-frame Lorentz factor
        Real uu1 = phydro->w(IM1,k,j,i);
        Real uu2 = phydro->w(IM2,k,j,i);
        Real uu3 = phydro->w(IM3,k,j,i);
        Real tmp = g(I11,i)*uu1*uu1 + 2.0*g(I12,i)*uu1*uu2 + 2.0*g(I13,i)*uu1*uu3
                   + g(I22,i)*uu2*uu2 + 2.0*g(I23,i)*uu2*uu3
                   + g(I33,i)*uu3*uu3;
        Real gamma = std::sqrt(1.0 + tmp);
        user_out_var(0,k,j,i) = gamma;
        if (!MAGNETIC_FIELDS_ENABLED) {
          continue;
        }

        // Calculate 4-velocity
        Real alpha = std::sqrt(-1.0/gi(I00,i));
        Real u0 = gamma/alpha;
        Real u1 = uu1 - alpha * gamma * gi(I01,i);
        Real u2 = uu2 - alpha * gamma * gi(I02,i);
        Real u3 = uu3 - alpha * gamma * gi(I03,i);
        Real u_0, u_1, u_2, u_3;
        pcoord->LowerVectorCell(u0, u1, u2, u3, k, j, i, &u_0, &u_1, &u_2, &u_3);

        // Calculate 4-magnetic field
        Real bb1 = pfield->bcc(IB1,k,j,i);
        Real bb2 = pfield->bcc(IB2,k,j,i);
        Real bb3 = pfield->bcc(IB3,k,j,i);
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
        Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;
        user_out_var(1,k,j,i) = b_sq/2.0;

      }
    }
  }


// electron_update(pcoord, peos, phydro, pfield, pscalars, is, ie, js, je, ks, ke );
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

void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  return;
}

//----------------------------------------------------------------------------------------
// Inflow boundary condition
// Inputs:
//   pmb: pointer to MeshBlock
//   pcoord: pointer to Coordinates
//   is,ie,js,je,ks,ke: indices demarkating active region
// Outputs:
//   prim: primitives set in ghost zones
//   bb: face-centered magnetic field set in ghost zones

void InflowBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &bb, Real time, Real dt,
                    int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  // Set hydro variables
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il-ngh; i <= il-1; ++i) {
        prim(IDN,k,j,i) = prim(IDN,k,j,il);
        prim(IEN,k,j,i) = prim(IEN,k,j,il);
        prim(IM1,k,j,i) = std::min(prim(IM1,k,j,il), static_cast<Real>(0.0));
        prim(IM2,k,j,i) = prim(IM2,k,j,il);
        prim(IM3,k,j,i) = prim(IM3,k,j,il);
      }
    }
  }
  if (!MAGNETIC_FIELDS_ENABLED) {
    return;
  }

  // Set radial magnetic field
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il-ngh; i <= il-1; ++i) {
        bb.x1f(k,j,i) = bb.x1f(k,j,il);
      }
    }
  }

  // Set polar magnetic field
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju+1; ++j) {
      for (int i=il-ngh; i <= il-1; ++i) {
        bb.x2f(k,j,i) = bb.x2f(k,j,il);
      }
    }
  }

  // Set azimuthal magnetic field
  for (int k=kl; k<=ku+1; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il-ngh; i <= il-1; ++i) {
        bb.x3f(k,j,i) = bb.x3f(k,j,il);
      }
    }
  }
  return;
}

namespace {
//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates
// Notes:
//   conversion is trivial in all currently implemented coordinate systems

void GetBoyerLindquistCoordinates(Real x1, Real x2, Real x3, Real *pr,
                                  Real *ptheta, Real *pphi) {
  if (std::strcmp(COORDINATE_SYSTEM, "schwarzschild") == 0 ||
      std::strcmp(COORDINATE_SYSTEM, "kerr-schild") == 0) {
    *pr = x1;
    *ptheta = x2;
    *pphi = x3;
  }
  return;
}

//----------------------------------------------------------------------------------------
// Function for transforming 4-vector from Boyer-Lindquist to desired coordinates
// Inputs:
//   a0_bl,a1_bl,a2_bl,a3_bl: upper 4-vector components in Boyer-Lindquist coordinates
//   r,theta,phi: Boyer-Lindquist coordinates of point
// Outputs:
//   pa0,pa1,pa2,pa3: pointers to upper 4-vector components in desired coordinates
// Notes:
//   Schwarzschild coordinates match Boyer-Lindquist when a = 0

void TransformVector(Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl, Real r,
                     Real theta, Real phi,
                     Real *pa0, Real *pa1, Real *pa2, Real *pa3) {
  if (std::strcmp(COORDINATE_SYSTEM, "schwarzschild") == 0) {
    *pa0 = a0_bl;
    *pa1 = a1_bl;
    *pa2 = a2_bl;
    *pa3 = a3_bl;
  } else if (std::strcmp(COORDINATE_SYSTEM, "kerr-schild") == 0) {
    Real delta = SQR(r) - 2.0*m*r + SQR(a);
    *pa0 = a0_bl + 2.0*m*r/delta * a1_bl;
    *pa1 = a1_bl;
    *pa2 = a2_bl;
    *pa3 = a3_bl + a/delta * a1_bl;
  }
  return;
}

//----------------------------------------------------------------------------------------
// Function for calculating angular momentum variable l
// Inputs:
//   r: desired radius of pressure maximum
// Outputs:
//   returned value: l = u^t u_\phi such that pressure maximum occurs at r_peak
// Notes:
//   beware many different definitions of l abound
//     this is *not* -u_phi/u_t
//   Harm has a similar function: lfish_calc() in init.c
//     Harm's function assumes M = 1 and that corotation is desired
//     it is equivalent to this, though seeing this requires much manipulation
//   implements (3.8) from Fishbone & Moncrief 1976, ApJ 207 962
//   assumes corotation
//   see CalculateRPeakFromL()

Real CalculateLFromRPeak(Real r) {
  Real num = SQR(SQR(r)) + SQR(a*r) - 2.0*m*SQR(a)*r - a*(SQR(r)-SQR(a))*std::sqrt(m*r);
  Real denom = SQR(r) - 3.0*m*r + 2.0*a*std::sqrt(m*r);
  return 1.0/r * std::sqrt(m/r) * num/denom;
}

//----------------------------------------------------------------------------------------
// Function for calculating pressure maximum radius r_peak
// Inputs:
//   l_target: desired u^t u_\phi
// Outputs:
//   returned value: location of pressure maximum given l_target
// Notes:
//   beware many different definitions of l abound
//     this is *not* -u_phi/u_t
//   uses (3.8) from Fishbone & Moncrief 1976, ApJ 207 962
//   assumes corotation
//   uses bisection to find r such that formula for l agrees with given value
//   proceeds until either absolute tolerance is met
//   returns best value after max_iterations reached if tolerances not met
//   returns NAN in case of failure (e.g. root not bracketed)
//   see CalculateLFromRPeak()

Real CalculateRPeakFromL(Real l_target) {
  // Parameters
  const Real tol_r = 1.0e-10;      // absolute tolerance on abscissa r_peak
  const Real tol_l = 1.0e-10;      // absolute tolerance on ordinate l
  const int max_iterations = 100;  // maximum number of iterations before best res

  // Prepare initial values
  Real r_a = r_min;
  Real r_b = r_max;
  Real r_c = 0.5 * (r_min + r_max);
  Real l_a = CalculateLFromRPeak(r_a);
  Real l_b = CalculateLFromRPeak(r_b);
  Real l_c = CalculateLFromRPeak(r_c);
  if (!((l_a < l_target && l_b > l_target) || (l_a > l_target && l_b < l_target))) {
    return NAN;
  }

  // Find root
  for (int n = 0; n < max_iterations; ++n) {
    if (std::abs(r_b-r_a) <= 2.0*tol_r || std::abs(l_c-l_target) <= tol_l) {
      break;
    }
    if ((l_a < l_target && l_c < l_target) || (l_a > l_target && l_c > l_target)) {
      r_a = r_c;
      l_a = l_c;
    } else {
      r_b = r_c;
      l_b = l_c;
    }
    r_c = 0.5 * (r_min + r_max);
    l_c = CalculateLFromRPeak(r_c);
  }
  return r_c;
}

//----------------------------------------------------------------------------------------
// Function for helping to calculate enthalpy
// Inputs:
//   r: radial Boyer-Lindquist coordinate
//   sin_theta: sine of polar Boyer-Lindquist coordinate
// Outputs:
//   returned value: log(h)
// Notes:
//   enthalpy defined here as h = p_gas/rho
//   references Fishbone & Moncrief 1976, ApJ 207 962 (FM)
//   implements first half of (FM 3.6)

Real LogHAux(Real r, Real sin_theta) {
  Real sin_sq_theta = SQR(sin_theta);
  Real cos_sq_theta = 1.0 - sin_sq_theta;
  Real delta = SQR(r) - 2.0*m*r + SQR(a);                    // \Delta
  Real sigma = SQR(r) + SQR(a)*cos_sq_theta;                 // \Sigma
  Real aa = SQR(SQR(r)+SQR(a)) - delta*SQR(a)*sin_sq_theta;  // A
  Real exp_2nu = sigma * delta / aa;                         // exp(2\nu) (FM 3.5)
  Real exp_2psi = aa / sigma * sin_sq_theta;                 // exp(2\psi) (FM 3.5)
  Real exp_neg2chi = exp_2nu / exp_2psi;                     // exp(-2\chi) (cf. FM 2.15)
  Real omega = 2.0*m*a*r/aa;                                 // \omega (FM 3.5)
  Real var_a = std::sqrt(1.0 + 4.0*SQR(l)*exp_neg2chi);
  Real var_b = 0.5 * std::log((1.0+var_a)
                              / (sigma*delta/aa));
  Real var_c = -0.5 * var_a;
  Real var_d = -l * omega;
  return var_b + var_c + var_d;                              // (FM 3.4)
}

//----------------------------------------------------------------------------------------
// Function for computing 4-velocity components at a given position inside untilted torus
// Inputs:
//   r: Boyer-Lindquist r
//   sin_theta: sine of Boyer-Lindquist theta
// Outputs:
//   pu0: u^t set (Boyer-Lindquist coordinates)
//   pu3: u^\phi set (Boyer-Lindquist coordinates)
// Notes:
//   The formula for u^3 as a function of u_{(\phi)} is tedious to derive, but this
//       matches the formula used in Harm (init.c).

void CalculateVelocityInTorus(Real r, Real sin_theta, Real *pu0, Real *pu3) {
  Real sin_sq_theta = SQR(sin_theta);
  Real cos_sq_theta = 1.0 - sin_sq_theta;
  Real delta = SQR(r) - 2.0*m*r + SQR(a);                    // \Delta
  Real sigma = SQR(r) + SQR(a)*cos_sq_theta;                 // \Sigma
  Real aa = SQR(SQR(r)+SQR(a)) - delta*SQR(a)*sin_sq_theta;  // A
  Real exp_2nu = sigma * delta / aa;                         // exp(2\nu) (FM 3.5)
  Real exp_2psi = aa / sigma * sin_sq_theta;                 // exp(2\psi) (FM 3.5)
  Real exp_neg2chi = exp_2nu / exp_2psi;                     // exp(-2\chi) (cf. FM 2.15)
  Real u_phi_proj_a = 1.0 + 4.0*SQR(l)*exp_neg2chi;
  Real u_phi_proj_b = -1.0 + std::sqrt(u_phi_proj_a);
  Real u_phi_proj = std::sqrt(0.5 * u_phi_proj_b);           // (FM 3.3)
  Real u3_a = (1.0+SQR(u_phi_proj)) / (aa*sigma*delta);
  Real u3_b = 2.0*m*a*r * std::sqrt(u3_a);
  Real u3_c = std::sqrt(sigma/aa) / sin_theta;
  Real u3 = u3_b + u3_c * u_phi_proj;
  Real g_00 = -(1.0 - 2.0*m*r/sigma);
  Real g_03 = -2.0*m*a*r/sigma * sin_sq_theta;
  Real g_33 = (sigma + (1.0 + 2.0*m*r/sigma) * SQR(a)
               * sin_sq_theta) * sin_sq_theta;
  Real u0_a = (SQR(g_03) - g_00*g_33) * SQR(u3);
  Real u0_b = std::sqrt(u0_a - g_00);
  Real u0 = -1.0/g_00 * (g_03*u3 + u0_b);
  *pu0 = u0;
  *pu3 = u3;
  return;
}

//----------------------------------------------------------------------------------------
// Function for computing 4-velocity components at a given position inside tilted torus
// Inputs:
//   r: Boyer-Lindquist r
//   theta,phi: Boyer-Lindquist theta and phi in BH-aligned coordinates
// Outputs:
//   pu0,pu1,pu2,pu3: u^\mu set (Boyer-Lindquist coordinates)
// Notes:
//   first finds corresponding location in untilted torus
//   next calculates velocity at that point in untilted case
//   finally transforms that velocity into coordinates in which torus is tilted

void CalculateVelocityInTiltedTorus(Real r, Real theta, Real phi, Real *pu0,
                                    Real *pu1, Real *pu2, Real *pu3) {
  // Calculate corresponding location
  Real sin_theta = std::sin(theta);
  Real cos_theta = std::cos(theta);
  Real sin_phi = std::sin(phi);
  Real cos_phi = std::cos(phi);
  Real sin_vartheta, cos_vartheta, varphi;
  if (psi != 0.0) {
    Real x = sin_theta * cos_phi;
    Real y = sin_theta * sin_phi;
    Real z = cos_theta;
    Real varx = cos_psi * x - sin_psi * z;
    Real vary = y;
    Real varz = sin_psi * x + cos_psi * z;
    sin_vartheta = std::sqrt(SQR(varx) + SQR(vary));
    cos_vartheta = varz;
    varphi = std::atan2(vary, varx);
  } else {
    sin_vartheta = std::abs(sin_theta);
    cos_vartheta = cos_theta;
    varphi = (sin_theta < 0.0) ? phi-PI : phi;
  }
  Real sin_varphi = std::sin(varphi);
  Real cos_varphi = std::cos(varphi);

  // Calculate untilted velocity
  Real u0_tilt, u3_tilt;
  CalculateVelocityInTorus(r, sin_vartheta, &u0_tilt, &u3_tilt);
  Real u1_tilt = 0.0;
  Real u2_tilt = 0.0;

  // Account for tilt
  *pu0 = u0_tilt;
  *pu1 = u1_tilt;
  if (psi != 0.0) {
    Real dtheta_dvartheta =
        (cos_psi * sin_vartheta + sin_psi * cos_vartheta * cos_varphi) / sin_theta;
    Real dtheta_dvarphi = -sin_psi * sin_vartheta * sin_varphi / sin_theta;
    Real dphi_dvartheta = sin_psi * sin_varphi / SQR(sin_theta);
    Real dphi_dvarphi = sin_vartheta / SQR(sin_theta)
                        * (cos_psi * sin_vartheta + sin_psi * cos_vartheta * cos_varphi);
    *pu2 = dtheta_dvartheta * u2_tilt + dtheta_dvarphi * u3_tilt;
    *pu3 = dphi_dvartheta * u2_tilt + dphi_dvarphi * u3_tilt;
  } else {
    *pu2 = u2_tilt;
    *pu3 = u3_tilt;
  }
  if (sin_theta < 0.0) {
    *pu2 *= -1.0;
    *pu3 *= -1.0;
  }
  return;
}

//----------------------------------------------------------------------------------------
// Function for finding approximate minimum value of plasma beta expected
// Inputs: (none)
// Outputs:
//   returned value: minimum beta found by sampling grid covering whole domain
// Notes:
//   constructs grid over entire mesh, not just block
//   grid is not necessarily the same as used for the problem proper
//   calculation is done entirely in Boyer-Lindquist coordinates

Real CalculateBetaMin() {
  // Prepare container to hold minimum
  Real beta_min_actual = std::numeric_limits<Real>::max();

  // Go through sample grid in phi
  for (int k = 0; k < sample_n_phi; ++k) {
    // Calculate phi values
    Real phi_m = phi_min + static_cast<Real>(k)/static_cast<Real>(sample_n_phi)
                 * (phi_max-phi_min);
    Real phi_p = phi_min + static_cast<Real>(k+1)/static_cast<Real>(sample_n_phi)
                 * (phi_max-phi_min);
    Real phi_c = 0.5 * (phi_m + phi_p);

    // Go through sample grid in theta
    for (int j = 0; j < sample_n_theta; ++j) {
      // Calculate theta values
      Real theta_m = theta_min + static_cast<Real>(j)/static_cast<Real>(sample_n_theta)
                     * (theta_max-theta_min);
      Real theta_p = theta_min + static_cast<Real>(j+1)/static_cast<Real>(sample_n_theta)
                     * (theta_max-theta_min);
      Real theta_c = 0.5 * (theta_m + theta_p);

      // Go through sample grid in r
      Real r_m, delta_r;
      Real r_p = 0.0;
      for (int i = 0; i < sample_n_r; ++i) {
        // Calculate r values
        if (i == 0) {
          r_m = r_min;
          Real ratio_power = 1.0;
          Real ratio_sum = 1.0;
          for (int ii = 1; ii < sample_n_r; ++ii) {
            ratio_power *= sample_r_rat;
            ratio_sum += ratio_power;
          }
          delta_r = (r_max-r_min) / ratio_sum;
        } else {
          r_m = r_p;
          delta_r *= sample_r_rat;
        }
        r_p = r_m + delta_r;
        Real r_c = 0.5 * (r_m + r_p);

        // Calculate beta
        Real beta;
        bool value_set = CalculateBeta(r_m, r_c, r_p, theta_m, theta_c, theta_p, phi_m,
                                       phi_c, phi_p, &beta);
        if (value_set) {
          beta_min_actual = std::min(beta_min_actual, beta);
        }
      }
    }
  }
  return beta_min_actual;
}

//----------------------------------------------------------------------------------------
// Function for calculating beta from four nearby points
// Inputs:
//   r_m,r_c,r_p: inner, center, and outer radii
//   theta_m,theta_c,theta_p: upper, center, and lower polar angles
// Outputs:
//   pbeta: value set to plasma beta at cell center
//   returned value: true if pbeta points to meaningful number (inside torus)
// Notes:
//   references Fishbone & Moncrief 1976, ApJ 207 962 (FM)

bool CalculateBeta(Real r_m, Real r_c, Real r_p, Real theta_m, Real theta_c,
                   Real theta_p, Real phi_m, Real phi_c, Real phi_p, Real *pbeta) {
  // Assemble arrays of points
  Real r_vals[7], theta_vals[7], phi_vals[7];
  r_vals[0] = r_c; theta_vals[0] = theta_c; phi_vals[0] = phi_c;
  r_vals[1] = r_m; theta_vals[1] = theta_c; phi_vals[1] = phi_c;
  r_vals[2] = r_p; theta_vals[2] = theta_c; phi_vals[2] = phi_c;
  r_vals[3] = r_c; theta_vals[3] = theta_m; phi_vals[3] = phi_c;
  r_vals[4] = r_c; theta_vals[4] = theta_p; phi_vals[4] = phi_c;
  r_vals[5] = r_c; theta_vals[5] = theta_c; phi_vals[5] = phi_m;
  r_vals[6] = r_c; theta_vals[6] = theta_c; phi_vals[6] = phi_p;

  // Account for tilt
  Real sin_theta_vals[7], cos_theta_vals[7];
  Real sin_phi_vals[7], cos_phi_vals[7];
  Real sin_vartheta_vals[7], cos_vartheta_vals[7];
  Real sin_varphi_vals[7], cos_varphi_vals[7];
  for (int p = 0; p < 7; ++p) {
    sin_theta_vals[p] = std::sin(theta_vals[p]);
    cos_theta_vals[p] = std::cos(theta_vals[p]);
    sin_phi_vals[p] = std::sin(phi_vals[p]);
    cos_phi_vals[p] = std::cos(phi_vals[p]);
    Real varphi;
    if (psi != 0.0) {
      Real x = sin_theta_vals[p] * cos_phi_vals[p];
      Real y = sin_theta_vals[p] * sin_phi_vals[p];
      Real z = cos_theta_vals[p];
      Real varx = cos_psi * x - sin_psi * z;
      Real vary = y;
      Real varz = sin_psi * x + cos_psi * z;
      sin_vartheta_vals[p] = std::sqrt(SQR(varx) + SQR(vary));
      if (field_config == MagneticFieldConfigs::vertical) {
        break;
      }
      cos_vartheta_vals[p] = varz;
      varphi = std::atan2(vary, varx);
    } else {
      sin_vartheta_vals[p] = std::abs(sin_theta_vals[p]);
      if (field_config == MagneticFieldConfigs::vertical) {
        break;
      }
      cos_vartheta_vals[p] = cos_theta_vals[p];
      varphi = (sin_theta_vals[p] < 0.0) ? phi_vals[p]-PI : phi_vals[p];
    }
    sin_varphi_vals[p] = std::sin(varphi);
    cos_varphi_vals[p] = std::cos(varphi);
  }

  // Determine if we are in the torus (FM 3.6)
  if (r_m < r_edge) {
    return false;
  }
  Real log_h_vals[7];
  for (int p = 0; p < 7; ++p) {
    log_h_vals[p] = LogHAux(r_vals[p], sin_vartheta_vals[p]) - log_h_edge;
    if (log_h_vals[p] < 0.0) {
      return false;
    }
    if (field_config == MagneticFieldConfigs::vertical) {
      break;
    }
  }

  // Calculate vector potential values in torus coordinates
  Real a_varphi_vals[7];
  Real pgas;
  for (int p = 0; p < 7; ++p) {
    Real pgas_over_rho = (gamma_adi-1.0)/gamma_adi * (std::exp(log_h_vals[p])-1.0);
    Real rho = std::pow(pgas_over_rho/k_adi, 1.0/(gamma_adi-1.0)) / rho_peak;
    if (p == 0 && rho < sample_cutoff) {
      return false;
    }
    if (p == 0) {
      pgas = pgas_over_rho * rho;
    }
    if (field_config == MagneticFieldConfigs::vertical) {
      break;
    }
    Real rho_cutoff = std::max(rho-potential_cutoff, static_cast<Real>(0.0));
    a_varphi_vals[p] =
        std::pow(r_vals[p], potential_r_pow) * std::pow(rho_cutoff, potential_rho_pow);
    if (a_varphi_vals[p] == 0.0) {
      return false;
    }
  }

  // Account for tilt
  Real a_theta_vals[7], a_phi_vals[7];
  if (field_config != MagneticFieldConfigs::vertical) {
    for (int p = 0; p < 7; ++p) {
      if (psi != 0.0) {
        Real dvarphi_dtheta = -sin_psi * sin_phi_vals[p] / SQR(sin_vartheta_vals[p]);
        Real dvarphi_dphi = sin_theta_vals[p] / SQR(sin_vartheta_vals[p])
                            * (cos_psi * sin_theta_vals[p]
                               - sin_psi * cos_theta_vals[p] * cos_phi_vals[p]);
        a_theta_vals[p] = dvarphi_dtheta * a_varphi_vals[p];
        a_phi_vals[p] = dvarphi_dphi * a_varphi_vals[p];
      } else {
        a_theta_vals[p] = 0.0;
        a_phi_vals[p] = a_varphi_vals[p];
      }
    }
  }

  // Calculate cell-centered 3-magnetic field
  Real det = (SQR(r_c) + SQR(a) * SQR(cos_theta_vals[0])) * std::abs(sin_theta_vals[0]);
  Real bb1, bb2, bb3;
  if (field_config != MagneticFieldConfigs::vertical) {
    bb1 = 1.0/det * ((a_phi_vals[4]-a_phi_vals[3]) / (theta_p-theta_m)
                     - (a_theta_vals[6]-a_theta_vals[5]) / (phi_p-phi_m));
    bb2 = -1.0/det * (a_phi_vals[2]-a_phi_vals[1]) / (r_p-r_m);
    bb3 = 1.0/det * (a_theta_vals[2]-a_theta_vals[1]) / (r_p-r_m);
  } else {
    Real rr = r_c * std::sin(theta_c);
    Real z = r_c * std::cos(theta_c);
    bb1 = rr * z / det;
    bb2 = -SQR(rr) / (r_c * det);
    bb3 = 0.0;
  }

  // Calculate beta
  Real pmag = CalculateMagneticPressure(bb1, bb2, bb3, r_c, theta_c, phi_c);
  *pbeta = pgas/pmag;
  return true;
}

//----------------------------------------------------------------------------------------
// Function for calculating beta given vector potential
// Inputs:
//   r_m,r_c,r_p: inner, center, and outer radii
//   theta_m,theta_c,theta_p: upper, center, and lower polar angles
//   a_cm,a_cp,a_mc,a_pc: A_phi offset by theta (down,up) and r (down,up)
// Outputs:
//   pbeta: value set to plasma beta at cell center
//   returned value: true if pbeta points to meaningful number (inside torus)
// Notes:
//   references Fishbone & Moncrief 1976, ApJ 207 962 (FM)

bool CalculateBetaFromA(Real r_m, Real r_c, Real r_p,
                        Real theta_m, Real theta_c, Real theta_p,
                        Real a_cm, Real a_cp, Real a_mc, Real a_pc,
                        Real *pbeta) {
  // Calculate trigonometric functions of theta
  Real sin_theta_c = std::sin(theta_c);
  Real cos_theta_c = std::cos(theta_c);

  // Determine if we are in the torus (FM 3.6)
  if (r_m < r_edge) {
    return false;
  }
  Real log_h = LogHAux(r_c, sin_theta_c) - log_h_edge;
  if (log_h < 0.0) {
    return false;
  }

  // Calculate primitives
  Real pgas_over_rho = (gamma_adi-1.0)/gamma_adi * (std::exp(log_h)-1.0);
  Real rho = std::pow(pgas_over_rho/k_adi, 1.0/(gamma_adi-1.0)) / rho_peak;
  Real pgas = pgas_over_rho * rho;

  // Check A_phi
  if (a_cm == 0.0 || a_cp == 0.0 || a_mc == 0.0 || a_pc == 0.0) {
    return false;
  }

  // Calculate 3-magnetic field
  Real det = (SQR(r_c) + SQR(a) * SQR(cos_theta_c)) * std::abs(sin_theta_c);
  Real bb1 = 1.0/det * (a_cp-a_cm) / (theta_p-theta_m);
  Real bb2 = -1.0/det * (a_pc-a_mc) / (r_p-r_m);
  Real bb3 = 0.0;

  // Calculate beta
  Real pmag = CalculateMagneticPressure(bb1, bb2, bb3, r_c, theta_c, 0.0);
  *pbeta = pgas/pmag;
  return true;
}

//----------------------------------------------------------------------------------------
// Function to calculate 1/2 * b^lambda b_lambda
// Inputs:
//   bb1,bb2,bb3: components of 3-magnetic field in Boyer-Lindquist coordinates
//   r,theta,phi: Boyer-Lindquist coordinates
// Outputs:
//   returned value: magnetic pressure

Real CalculateMagneticPressure(Real bb1, Real bb2, Real bb3, Real r, Real theta,
                               Real phi) {
  // Calculate Boyer-Lindquist metric
  Real sin_theta = std::sin(theta);
  Real cos_theta = std::cos(theta);
  Real delta = SQR(r) - 2.0*m*r + SQR(a);
  Real sigma = SQR(r) + SQR(a) * SQR(cos_theta);
  Real g_00 = -(1.0 - 2.0*m*r/sigma);
  Real g_01 = 0.0;
  Real g_02 = 0.0;
  Real g_03 = -2.0*m*a*r/sigma * SQR(sin_theta);
  Real g_11 = sigma/delta;
  Real g_12 = 0.0;
  Real g_13 = 0.0;
  Real g_22 = sigma;
  Real g_23 = 0.0;
  Real g_33 = (SQR(r) + SQR(a) + 2.0*m*SQR(a)*r/sigma * SQR(sin_theta)) * SQR(sin_theta);
  Real g_10 = g_01;
  Real g_20 = g_02;
  Real g_21 = g_12;
  Real g_30 = g_03;
  Real g_31 = g_13;
  Real g_32 = g_23;

  // Calculate 4-velocity
  Real u0, u1, u2, u3;
  CalculateVelocityInTiltedTorus(r, theta, phi, &u0, &u1, &u2, &u3);

  // Calculate 4-magnetic field
  Real b0 = bb1 * (g_10*u0 + g_11*u1 + g_12*u2 + g_13*u3)
            + bb2 * (g_20*u0 + g_21*u1 + g_22*u2 + g_23*u3)
            + bb3 * (g_30*u0 + g_31*u1 + g_32*u2 + g_33*u3);
  Real b1 = 1.0/u0 * (bb1 + b0 * u1);
  Real b2 = 1.0/u0 * (bb2 + b0 * u2);
  Real b3 = 1.0/u0 * (bb3 + b0 * u3);

  // Calculate magnetic pressure
  Real b_sq = g_00*b0*b0 + g_01*b0*b1 + g_02*b0*b2 + g_03*b0*b3
              + g_10*b1*b0 + g_11*b1*b1 + g_12*b1*b2 + g_13*b1*b3
              + g_20*b2*b0 + g_21*b2*b1 + g_22*b2*b2 + g_23*b2*b3
              + g_30*b3*b0 + g_31*b3*b1 + g_32*b3*b2 + g_33*b3*b3;
  return 0.5*b_sq;
}
} // namespace

void inner_boundary_source_function(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> *flux,
  const AthenaArray<Real> &cons_old,const AthenaArray<Real> &cons_half, AthenaArray<Real> &cons,
  const AthenaArray<Real> &prim_old,const AthenaArray<Real> &prim_half,  AthenaArray<Real> &prim, 
  const FaceField &bb_half,const FaceField &bb,
  const AthenaArray<Real> &s_old,const AthenaArray<Real> &s_half, AthenaArray<Real> &s_scalar, 
  const AthenaArray<Real> &r_half,AthenaArray<Real> &prim_scalar){
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





  apply_inner_boundary_condition(pmb,prim,prim_scalar);


}


/* Apply inner "absorbing" boundary conditions */

void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim,AthenaArray<Real> &prim_scalar){


  Real r,th,ph;
  AthenaArray<Real> &g = pmb->ruser_meshblock_data[0], &gi = pmb->ruser_meshblock_data[1];



   for (int k=pmb->ks; k<=pmb->ke; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=pmb->js; j<=pmb->je; ++j) {
      pmb->pcoord->CellMetric(k, j, pmb->is, pmb->ie, g, gi);
#pragma simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {


         GetKSCoordinates(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j),pmb->pcoord->x3v(k), &r, &th, &ph);

          if (r < r_inner_boundary){
              

              //set uu assuming u is zero
              Real gamma = 1.0;
              Real alpha = std::sqrt(-1.0/gi(I00,i));
              Real u0 = gamma/alpha;
              Real uu1 = - gi(I01,i)/gi(I00,i) * u0;
              Real uu2 = - gi(I02,i)/gi(I00,i) * u0;
              Real uu3 = - gi(I03,i)/gi(I00,i) * u0;
              
              prim(IDN,k,j,i) = dfloor;
              prim(IVX,k,j,i) = 0.;
              prim(IVY,k,j,i) = 0.;
              prim(IVZ,k,j,i) = 0.;
              prim(IPR,k,j,i) = pfloor;


              if (NSCALARS>0)prim_scalar(0,k,j,i) = prim(IPR,k,j,i)/std::pow(prim(IDN,k,j,i),gamma_adi);
              if (NSCALARS>1)prim_scalar(1,k,j,i) = ue_to_kappa(ue_over_ug_floor * prim(IPR,k,j,i)/gm1,prim(IDN,k,j,i),ge);
              if (NSCALARS>2)prim_scalar(2,k,j,i) = ue_to_kappa(ue_over_ug_floor * prim(IPR,k,j,i)/gm1,prim(IDN,k,j,i),ge);
              if (NSCALARS>3)prim_scalar(3,k,j,i) = ue_to_kappa(ue_over_ug_floor * prim(IPR,k,j,i)/gm1,prim(IDN,k,j,i),ge);
              
              
              
          }



}}}




}
