#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:49:03 2020
"""
import numpy as np

import tools
import matplotlib.backends.backend_pdf as backend_pdf
import copy

fd_data = "./data/"
tools.mkdir(fd_data)

# Input
fname_in = fd_data + "kf_data.txt"
format_in = ['%f','%f','%f','%f','%f','%f','%f'] 
            # time, pe_ref, pn_ref, ve_ref, vn_ref, pe_meas, pn_meas

# Output
out_pdf = fd_data + "processing.pdf"   # to be renamed using parameters
pdf = backend_pdf.PdfPages(out_pdf)

MODE_BLUNDER_DETECTION = 1     # 1-Use BD; 0-not used
TH_SCALE_BD = 2.5                # Scale factor for BD
SCALE_R_BD  = 100              # Scale factor for reducing weight of BD meas

# Plot 
FS_PLOT = 22
SHIFT_LEGEND_X = 1.6

"""
Load data
"""
da = tools.load_text_file(fname_in, format_in, ' ')
t = da[0]
pe_ref = da[1]
pn_ref = da[2]
ve_ref = da[3]
vn_ref = da[4]
pe_meas = da[5]
pn_meas = da[6]

fig = tools.plot_figure([pe_ref, pe_meas], [pn_ref, pn_meas], 2, 
                       [len(pe_ref),len(pe_meas)], ["b.-", "r*-"], 
                       ["$p_{2D}$ (ref)", "$p_{2D}$ (meas)"], "Position measurement and reference", "East ($m$)", 
                       "North ($m$)", '', 1, 1, 0, [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X)
pdf.savefig(fig)

fig = tools.plot_figure([t, t], [ve_ref, vn_ref], 2,
                        [len(t),len(t)], 
                       ["b.-","r+-"], ["$v_{e}$ (ref)", "$v_{n}$ (meas)"],
                       "Velocity reference", 
                       "Time ($sec$)", "Velocity ($m/s$)", '', 1, 0, 0, 
                       [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X)
pdf.savefig(fig)

"""
Lists for storing results (for result analysis only)
"""
# Store KF results (for result analysis only)
pe_kf = [0.0 for _ in range(len(t))]
pn_kf = [0.0 for _ in range(len(t))]
ve_kf = [0.0 for _ in range(len(t))]
vn_kf = [0.0 for _ in range(len(t))]
be_kf = [0.0 for _ in range(len(t))]
bn_kf = [0.0 for _ in range(len(t))]

# Store uncertainty in KF results (for result analysis only)
sqt_cov_pe_kf = [0.0 for _ in range(len(t))]
sqt_cov_pn_kf = [0.0 for _ in range(len(t))]
sqt_cov_ve_kf = [0.0 for _ in range(len(t))]
sqt_cov_vn_kf = [0.0 for _ in range(len(t))]
sqt_cov_be_kf = [0.0 for _ in range(len(t))]
sqt_cov_bn_kf = [0.0 for _ in range(len(t))]

# Store uncertainty in KF prediction (for result analysis only)
pred_sqt_cov_pe_kf = [0.0 for _ in range(len(t))]
pred_sqt_cov_pn_kf = [0.0 for _ in range(len(t))]
pred_sqt_cov_ve_kf = [0.0 for _ in range(len(t))]
pred_sqt_cov_vn_kf = [0.0 for _ in range(len(t))]
pred_sqt_cov_be_kf = [0.0 for _ in range(len(t))]
pred_sqt_cov_bn_kf = [0.0 for _ in range(len(t))]

# Store uncertainty from Riccati equation (for result analysis only)
riccati_sqt_cov_pe_kf = [0.0 for _ in range(len(t))]
riccati_sqt_cov_pn_kf = [0.0 for _ in range(len(t))]
riccati_sqt_cov_ve_kf = [0.0 for _ in range(len(t))]
riccati_sqt_cov_vn_kf = [0.0 for _ in range(len(t))]
riccati_sqt_cov_be_kf = [0.0 for _ in range(len(t))]
riccati_sqt_cov_bn_kf = [0.0 for _ in range(len(t))]

# Store uncertainty from Riccati equation - offline (for result analysis only)
riccati_sqt_cov_pe_off = [0.0 for _ in range(len(t))]
riccati_sqt_cov_pn_off = [0.0 for _ in range(len(t))]
riccati_sqt_cov_ve_off = [0.0 for _ in range(len(t))]
riccati_sqt_cov_vn_off = [0.0 for _ in range(len(t))]
riccati_sqt_cov_be_off = [0.0 for _ in range(len(t))]
riccati_sqt_cov_bn_off = [0.0 for _ in range(len(t))]

# Store other variables (for result analysis only)
n_obs_each_epoch = 2
inno_kf = [[0.0 for _ in range(len(t))] for _ in range(n_obs_each_epoch)]
                                                              # KF innovation 
inno_norm_kf = [0.0 for _ in range(len(t))]          # Norm of KF innovation 
sqrt_cov_inno_kf = [[0.0 for _ in range(len(t))] for _ in range(n_obs_each_epoch)]
                                    # Square root of covariance KF innovation 
sqrt_cov_inno_norm_kf = [0.0 for _ in range(len(t))]  
                                        # Square rootof norm of KF innovation 

###################### KF Computation ######################
"""
KF parameters
"""
# Two types parameters to set here:
# 1. Parameters for KF computation
#    - The state vector xk (commonly zero vector)
#    - The initial Pk matrix
#    - The Q matrix (if Q is adaptive, move Q setting into KF loop)
#    - The Rk matrix (if Rk is adaptive, move Rk setting into KF loop)
# Note: The term with "k" is in a discrete form

# xk
# [dpe, dpn, dve, dvn, dae, dan]^T, that is,
# the error of pe, pn, ve, vn, ae, an
n_state = 6
xk = np.matrix(np.zeros((n_state, 1)) )  

# P0
sigma_init_pe = 30.0   # Uncertainty of initial position
sigma_init_pn = 30.0
sigma_init_ve = 3.0    # Uncertainty of initial velocity
sigma_init_vn = 3.0
sigma_init_be = 1.0    # Uncertainty of initial accelration
sigma_init_bn = 1.0
Pk = np.matrix( [ [sigma_init_pe**2, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, sigma_init_pn**2, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, sigma_init_ve**2, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, sigma_init_vn**2, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, sigma_init_be**2, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, sigma_init_bn**2]
                 ] )

SCALE_P0 = 1.0       # Scale factor for P0 (for analysis only)
Pk = Pk * SCALE_P0
    
# Q
q_a = 0.1       # Velocity Random Walk (VRW, reflect acceleration white noise,
                # unit: m/s/sqrt(s))  
tau_ba = 10.0   # Correlation time for acceleration Gauss-Markov process
                # unit: s
sigma_ba = 0.1  # Instability (RMS of driving white noise) for 
                # acceleration Gauss-Markov process
                # unit: m/s^2
q_ba = np.sqrt(2 * sigma_ba**2 / tau_ba)
Q = np.matrix( [ [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, q_a**2, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, q_a**2, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, q_ba**2, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, q_ba**2]
                ] )
    
# Parameter for Rk
sigma_pos_meas = 1.0    # uncertainty for position observation, unit: m

# 2. Navigation state (e.g., position, velocity, acceleration) values
#    The KF computation use errors of the states, however,
#    The navigation states are also computed in the KF prediction step  
#    and updated in the KF update step
#    The predicted navigation state valuesZQ are used to compute 
#    the innovation (by comparing with the observations)

# Navigation state [pe, pn, ve, vn]^T values
Xk = np.matrix(np.zeros((n_state, 1)) )

def kf_predict(x_, P_, Qk_, PHI_):
    # Predict state vector
    x_ = PHI * x_   
    
    # Predict covariance
    # P = PHI * P * PHI_t + Qk
    P_ = PHI * P_ * np.transpose(PHI_) + Qk_
    return x_, P_
    

def kf_update(x_, P_, inno_, H_, R_, n_state_):
    # Compute KF gain
    # K = Pk(-) * Ht * ( H * Pk(-) * Ht + R )^-1
    K_ = P_ * np.transpose(H_) * np.linalg.inv(H_ * P_ * np.transpose(H_) + R_)
    
    # Update state vector
    # dx= K * (Z - H * xk(-) ) = K *inno  
    x_ = x_ + K_ * inno_
    
    # Update covariance
    # Pk(+) = (I - K * H) * Pk(-)
    P_ = (np.eye(n_state_) - K_ * H_) * P_
    
    return x_, P_

"""
KF
"""
# Fill F, Q matrices
F = np.matrix( [ [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                 [0.0, 0.0, 0.0, 0.0, -1.0/tau_ba, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, -1.0/tau_ba]] )

G = np.identity(n_state)

# Design matrix Hk
Hk = np.matrix( [ [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0] ] )
   
# Measurement noise matrix Rk (without blunder)
Rk_normal = np.matrix( [ [sigma_pos_meas**2, 0.0],
                        [ 0.0, sigma_pos_meas**2] ] )
Rk = copy.deepcopy(Rk_normal)
    
t_pre = 0.0
for k in range(len(t)):
    t_cur = t[k]
        
    """
    KF Prediction
    """
    # Discretization of matrices 
    if k == 0:
        d_t = t[-1] / len(t)
    else:
        d_t = t_cur - t_pre    
    
    I = np.identity(n_state)
    PHI = I + F*d_t

    # Discretization of matrices 
    # ---- Qk = 0.5 * ( PHI * GQGt + GQGt * PHI_t) * dt -----
    GQGt = G * Q * np.transpose(G)
    Qk = 0.5*(PHI*GQGt + GQGt*np.transpose(PHI)) * d_t
    
    # KF prediction
    xk, Pk = kf_predict(xk, Pk, Qk, PHI) 
    
    # Prediction of navigation state
    Xk = PHI * Xk 
    
    # Store uncertainty in KF prediction (for result analysis only)
    pred_sqt_cov_pe_kf[k] = np.sqrt(Pk[0,0])
    pred_sqt_cov_pn_kf[k] = np.sqrt(Pk[1,1])
    pred_sqt_cov_ve_kf[k] = np.sqrt(Pk[2,2])
    pred_sqt_cov_vn_kf[k] = np.sqrt(Pk[3,3])
    pred_sqt_cov_be_kf[k] = np.sqrt(Pk[4,4])
    pred_sqt_cov_bn_kf[k] = np.sqrt(Pk[5,5])
    
    # Riccati equation solution (for result analysis only)
    if k == 0:
        Pk_Ricatti = copy.deepcopy(Pk)
        Pk_Ricatti_off = copy.deepcopy(Pk)
    else:
        Pk_Ricatti = PHI * Pk_Ricatti * np.transpose(PHI) + Qk \
                     - PHI * Pk_Ricatti * np.transpose(Hk) \
                     * np.linalg.inv(Hk * Pk_Ricatti * np.transpose(Hk) + Rk) \
                     * Hk * Pk_Ricatti * np.transpose(PHI)
        Pk_Ricatti_off = PHI * Pk_Ricatti_off * np.transpose(PHI) + Qk \
                     - PHI * Pk_Ricatti_off * np.transpose(Hk) \
                     * np.linalg.inv(Hk * Pk_Ricatti_off * np.transpose(Hk) + Rk_normal) \
                     * Hk * Pk_Ricatti_off * np.transpose(PHI)
                     
    # Store Riccati equation solutions (for result analysis only)
    riccati_sqt_cov_pe_kf[k] = np.sqrt(Pk_Ricatti[0,0])
    riccati_sqt_cov_pn_kf[k] = np.sqrt(Pk_Ricatti[1,1])
    riccati_sqt_cov_ve_kf[k] = np.sqrt(Pk_Ricatti[2,2])
    riccati_sqt_cov_vn_kf[k] = np.sqrt(Pk_Ricatti[3,3])
    riccati_sqt_cov_be_kf[k] = np.sqrt(Pk_Ricatti[4,4])
    riccati_sqt_cov_bn_kf[k] = np.sqrt(Pk_Ricatti[5,5])
    
    riccati_sqt_cov_pe_off[k] = np.sqrt(Pk_Ricatti_off[0,0])
    riccati_sqt_cov_pn_off[k] = np.sqrt(Pk_Ricatti_off[1,1])
    riccati_sqt_cov_ve_off[k] = np.sqrt(Pk_Ricatti_off[2,2])
    riccati_sqt_cov_vn_off[k] = np.sqrt(Pk_Ricatti_off[3,3])
    riccati_sqt_cov_be_off[k] = np.sqrt(Pk_Ricatti_off[4,4])
    riccati_sqt_cov_bn_off[k] = np.sqrt(Pk_Ricatti_off[5,5])

    """
    KF Update
    """    
    Hkxk = Hk * xk
    
    # Use predicted position - measured position for zk
    zk = np.matrix( [ [ Xk[0,0] - pe_meas[k]],
                      [ Xk[1,0] - pn_meas[k]] ] )
    # Innovation
    inno_k = np.matrix( [ [zk[0,0] - Hkxk[0,0]],
                          [zk[1,0] - Hkxk[1,0]] ] )
    
    # Covariance matrix for innovation
    Cv = Hk * Pk * np.transpose(Hk) + Rk

    # Innovation and covariance for blunder detection 
    norm_inno = 0.0
    th_cv = 0.0
    for j in range(n_obs_each_epoch):
        norm_inno += inno_k[j,0]**2
        th_cv += Cv[j,j]
    norm_inno = np.sqrt(norm_inno)
    th_cv = np.sqrt(th_cv)

    # Blunder Detection
    flag_blunder = 0
    if MODE_BLUNDER_DETECTION:        
        if norm_inno > TH_SCALE_BD * th_cv:
            print("BD in epoch %d, norm_inno=%.2f, th_cv=%.2f"
                  %(k, norm_inno, th_cv) )
            flag_blunder = 1    
    
    # Adjust measurement noise matrix according to blunder detecton
    Rk = copy.deepcopy(Rk_normal)
    if flag_blunder == 1:
        # Increase R when a BD occurs
        Rk = SCALE_R_BD * Rk  
    
    # Store innovation and covariance (for result analysis only) 
    for j in range(n_obs_each_epoch):
        inno_kf[j][k] = inno_k[j,0]
        sqrt_cov_inno_kf[j][k] = np.sqrt(Cv[j,j])
    inno_norm_kf[k] = norm_inno
    sqrt_cov_inno_norm_kf[k] = th_cv 
    
    # KF update
    xk, Pk = kf_update(xk, Pk, inno_k, Hk, Rk, n_state)
    
    """
    Feedback
    """
    # Feedback estimated state error values into navigation states
    # Set state errors to zeros after feedback
    Xk = Xk - xk
    xk = np.matrix(np.zeros((n_state, 1)) )  

    """
    Output results
    """
    pe_kf[k] = Xk[0,0]
    pn_kf[k] = Xk[1,0]
    ve_kf[k] = Xk[2,0]
    vn_kf[k] = Xk[3,0]
    be_kf[k] = Xk[4,0]
    bn_kf[k] = Xk[5,0]
    
    sqt_cov_pe_kf[k] = np.sqrt(Pk[0,0])
    sqt_cov_pn_kf[k] = np.sqrt(Pk[1,1])
    sqt_cov_ve_kf[k] = np.sqrt(Pk[2,2])
    sqt_cov_vn_kf[k] = np.sqrt(Pk[3,3])
    sqt_cov_be_kf[k] = np.sqrt(Pk[4,4])
    sqt_cov_bn_kf[k] = np.sqrt(Pk[5,5])
    
    
    """
    Other variables
    """
    t_pre = t_cur
    
    
###################### End of KF Computation ######################

"""
Plots and statistics
"""
# Veclocity solution
fig = tools.plot_figure([t, t, t, t], [ve_ref, vn_ref, ve_kf, vn_kf], 4,
                        [len(t),len(t), len(t),len(t)], 
                       ["b.-","r+-", "c*-","mo-"], 
                       ["$v_{e}$ (ref)","$v_{n}$ (ref)","$v_{e}$ (kf)","$v_{n} $(kf)"],
                       "Velocity", 
                       "Time ($sec$)", "Velocity ($m/s$)", '', 1, 0, 0, 
                       [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X)
pdf.savefig(fig)

# Position solution
fig = tools.plot_figure([pe_ref, pe_meas, pe_kf], [pn_ref, pn_meas, pn_kf], 3, 
                       [len(pe_ref),len(pe_meas),len(pe_kf)], ["g.-", "b*-", "ro-"], 
                       ["$p_{2D}$ (ref)","$p_{2D}$ (meas)","$p_{2D}$ (kf)"], "Position", "East ($m$)", 
                       "North ($m$)", '', 1, 1, 0, [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X+0.05)
pdf.savefig(fig)

# Acceleration solution
fig = tools.plot_figure([t, t], [be_kf, bn_kf], 2, [len(t),len(t)], 
                       ["c.-","m+-"], ["$a_{e}$","$a_{n}$"], "Acceleration", 
                       "Time ($sec$)", "Acceleration ($m/s^2$)", '', 1, 0, 0, 
                       [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X-0.2)
pdf.savefig(fig)

# Innovation
fig = tools.plot_figure([t, t, t], [inno_kf[0], inno_kf[1], inno_norm_kf], 3, 
                        [len(t),len(t),len(t)], 
                       ["b.-","r+-","g:"], ["$Inno_{e}$","$Inno_{n}$","$Inno_{2D}$"], "KF innovation", 
                       "Time ($sec$)", "Innovation ($m$)", '', 1, 0, 0, 
                       [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X)
pdf.savefig(fig)

# Threshold for blunder detection
th_bd = tools.scale_vector(sqrt_cov_inno_norm_kf, TH_SCALE_BD)

fig = tools.plot_figure([t, t], [inno_norm_kf, th_bd], 2, 
                        [len(t),len(t)], 
                       ["b+-","r-"], ["$Inno$","$Th_{inno}$"], "KF innovation and threshold for BD", 
                       "Time ($sec$)", "Innovation ($m$)", '', 1, 0, 0, 
                       [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X-0.1)
pdf.savefig(fig)

# Riccati equation
sqt_cov_p_kf = tools.norm_of_vectors(sqt_cov_pe_kf, sqt_cov_pn_kf)
pred_sqt_cov_p_kf = tools.norm_of_vectors(pred_sqt_cov_pe_kf, pred_sqt_cov_pn_kf)
riccati_sqt_cov_p_kf = tools.norm_of_vectors(riccati_sqt_cov_pe_kf, riccati_sqt_cov_pn_kf)
riccati_sqt_cov_p_off = tools.norm_of_vectors(riccati_sqt_cov_pe_off, riccati_sqt_cov_pn_off)

sqt_cov_v_kf = tools.norm_of_vectors(sqt_cov_ve_kf, sqt_cov_vn_kf)
pred_sqt_cov_v_kf = tools.norm_of_vectors(pred_sqt_cov_ve_kf, pred_sqt_cov_vn_kf)
riccati_sqt_cov_v_kf = tools.norm_of_vectors(riccati_sqt_cov_ve_kf, riccati_sqt_cov_vn_kf) 
riccati_sqt_cov_v_off = tools.norm_of_vectors(riccati_sqt_cov_ve_off, riccati_sqt_cov_vn_off) 

fig = tools.plot_figure([t, t, t, t], 
                        [sqt_cov_p_kf, pred_sqt_cov_p_kf, 
                         riccati_sqt_cov_p_kf, riccati_sqt_cov_p_off], 4, 
                       [len(t),len(t),len(t),len(t)], 
                       ["b-", "ro-", "g.-", "c+-"],  
                       ["$\sigma_{p_{2D}}$ (KF update)",
                        "$\sigma_{p_{2D}}$ (KF prediction)",
                        "$\sigma_{p_{2D}}$ (Riccati, online)",
                        "$\sigma_{p_{2D}}$ (Riccati, offline)"], 
                       "Position uncertainty", "Time ($s$)", 
                       "Position error ($m$)", '', 1, 0, 0, [], 1, [0, 3], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X+0.3, 
                       y_shift_legend=1.05)
pdf.savefig(fig)

fig = tools.plot_figure([t, t, t, t], 
                        [sqt_cov_v_kf, pred_sqt_cov_v_kf, 
                         riccati_sqt_cov_v_kf, riccati_sqt_cov_v_off], 4, 
                       [len(t),len(t),len(t),len(t)], 
                       ["b-", "ro-", "g.-", "c+-"],  
                       ["$\sigma_{p_{2D}}$ (KF update)",
                        "$\sigma_{p_{2D}}$ (KF prediction)",
                        "$\sigma_{p_{2D}}$ (Riccati, online)",
                        "$\sigma_{p_{2D}}$ (Riccati, offline)"], 
                       "Velocity uncertainty", "Time ($s$)", 
                       "Velocity error ($m/s$)", '', 1, 0, 0, [], 1, [0, 1.5], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X+0.3)
pdf.savefig(fig)

        
# Plot the errors
err_pe_meas = [0.0 for _ in range(len(t))]
err_pn_meas = [0.0 for _ in range(len(t))]
err_p_meas = [0.0 for _ in range(len(t))]

err_pe_kf = [0.0 for _ in range(len(t))]
err_pn_kf = [0.0 for _ in range(len(t))]
err_p_kf = [0.0 for _ in range(len(t))]

err_ve_kf = [0.0 for _ in range(len(t))]
err_vn_kf = [0.0 for _ in range(len(t))]
err_v_kf = [0.0 for _ in range(len(t))]

for k in range(len(t)):
    err_pe_meas[k] = pe_meas[k] - pe_ref[k]
    err_pn_meas[k] = pn_meas[k] - pn_ref[k]
    err_p_meas[k] = np.linalg.norm([err_pe_meas[k], err_pn_meas[k]])
    
    err_pe_kf[k] = pe_kf[k] - pe_ref[k]
    err_pn_kf[k] = pn_kf[k] - pn_ref[k]
    err_p_kf[k] = np.linalg.norm([err_pe_kf[k], err_pn_kf[k]])
    
    err_ve_kf[k] = ve_kf[k] - ve_ref[k]
    err_vn_kf[k] = vn_kf[k] - vn_ref[k]
    err_v_kf[k] = np.linalg.norm([err_ve_kf[k], err_vn_kf[k]])
    
fig = tools.plot_figure([t, t, t, t], [err_pe_meas, err_pn_meas, err_pe_kf, err_pn_kf], 4,
                        [len(t),len(t), len(t),len(t)], 
                       ["b.-","r+-", "c*-","mo-"], 
                       ["$e_{p_{e}}$ (meas)","$e_{p_{n}}$ (meas)","$e_{p_{e}}$ (kf)","$e_{p_{n}}$ (kf)"],
                       "Position errors", 
                       "Time ($sec$)", "Error ($m$)", '', 1, 0, 0, 
                       [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X+0.1)
pdf.savefig(fig)

fig = tools.plot_figure([t, t], [err_ve_kf, err_vn_kf], 2,
                        [len(t),len(t)], 
                       ["c*-","mo-"], ["$e_{v_{e}}$ (kf)","$e_{v_{n}}$ (kf)"],
                       "Velocity errors", 
                       "Time ($sec$)", "Error ($m/s$)", '', 1, 0, 0, 
                       [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X)
pdf.savefig(fig)

fig = tools.plot_figure([t, t, t, t], [err_pe_kf, err_pn_kf, sqt_cov_pe_kf, sqt_cov_pn_kf], 4,
                        [len(t),len(t), len(t),len(t)], 
                       ["c*-","mo-","g.:","y--"], ["$e_{p_{e}}$","$e_{p_{n}}$","$\sigma_{p_{e}}$","$\sigma_{p_{n}}$"],
                       "Position errors and uncertainty from KF", 
                       "Time ($sec$)", "Error ($m$)", '', 1, 0, 0, 
                       [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X-0.2)
pdf.savefig(fig)

fig = tools.plot_figure([t, t, t, t], [err_ve_kf, err_vn_kf, sqt_cov_ve_kf, sqt_cov_vn_kf], 4,
                        [len(t),len(t),len(t),len(t)], 
                       ["c*-","mo-","g.:","y--"], ["$e_{v_{e}}$","$e_{v_{n}}$","$\sigma_{v_{e}}$","$\sigma_{v_{n}}$"],
                       "Velocity errors and uncertainty from KF", 
                       "Time ($sec$)", "Error ($m/s$)", '', 1, 0, 0, 
                       [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X-0.2)
pdf.savefig(fig)
    
# Error statistics
(mean1, std1, rms1, d801, d951, max1) = tools.cal_statics(err_pe_meas)
print("")
print("pe_meas: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))
(mean1, std1, rms1, d801, d951, max1) = tools.cal_statics(err_pn_meas)
print("pn_meas: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))
(mean1, std1, rms1, d801, d951, max1) = tools.cal_statics(err_p_meas)
print("p_meas: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))

print("")
(mean1, std1, rms1, d801, d951, max1) = tools.cal_statics(err_pe_kf)
print("pe_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(err_pn_kf)
print("pn_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(err_p_kf)
print("p_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))

print("")
(mean1, std1, rms1, d801, d951, max1) = tools.cal_statics(err_ve_kf)
print("ve_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m/s"
      %(mean1, rms1, d801, d951, max1))
(mean1, std1, rms1, d801, d951, max1) = tools.cal_statics(err_vn_kf)
print("vn_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m/s"
      %(mean1, rms1, d801, d951, max1))
(mean1, std1, rms1, d801, d951, max1) = tools.cal_statics(err_v_kf)
print("v_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m/s"
      %(mean1, rms1, d801, d951, max1))

"""
Save solution
"""
label_fname = "_n_%d_Q_%.3f_%.2f_%.3f_R_%.3f_BD_%d_%.1f_P0_%.4f"%(
                n_state, q_a, tau_ba, sigma_ba, sigma_pos_meas, MODE_BLUNDER_DETECTION, 
                TH_SCALE_BD, SCALE_P0)
fname_out = fd_data + "solution" + label_fname + ".txt"
           
da_sol = [t, 
          pe_ref, pn_ref, ve_ref, vn_ref, pe_meas, pn_meas,
          pe_kf, pn_kf, ve_kf, vn_kf, be_kf, bn_kf,
          sqt_cov_pe_kf, sqt_cov_pn_kf, sqt_cov_ve_kf, sqt_cov_vn_kf, sqt_cov_be_kf, sqt_cov_bn_kf,
          inno_kf[0], inno_kf[1], inno_norm_kf,
          sqrt_cov_inno_kf[0], sqrt_cov_inno_kf[1], sqrt_cov_inno_norm_kf,
          err_pe_meas, err_pn_meas, err_p_meas,
          err_pe_kf, err_pn_kf, err_p_kf,
          err_ve_kf, err_vn_kf, err_v_kf,
          ]
format_out = ['%.3f',
              '%.3f','%.3f','%.3f','%.3f', '%.3f','%.3f', 
              '%.3f','%.3f','%.3f','%.3f','%.3f','%.3f',
              '%.3f','%.3f','%.3f','%.3f','%.3f','%.3f',
              '%.3f','%.3f','%.3f',
              '%.3f','%.3f','%.3f',
              '%.3f','%.3f','%.3f',
              '%.3f','%.3f','%.3f',
              '%.3f','%.3f','%.3f',
              ] 
tools.write_text_file(fname_out, da_sol, format_out, ' ')
     
"""
Close PDF file and rename it based on parameter setting
"""
pdf.close()

import os
os.rename(fd_data + "processing.pdf", 
          fd_data + "processing" + label_fname + ".pdf")