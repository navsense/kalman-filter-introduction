#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:21:33 2020
"""

import tools
import matplotlib.backends.backend_pdf as backend_pdf


fd_data = "./data/"
tools.mkdir(fd_data)

# Input
fname_tau_001 = fd_data + "solution_n_6_Q_0.100_0.10_0.100_R_1.000_BD_1_2.5_P0_1.0000.txt"
fname_tau_01   = fd_data + "solution_n_6_Q_0.100_1.00_0.100_R_1.000_BD_1_2.5_P0_1.0000.txt"
fname_tau_1  = fd_data + "solution_n_6_Q_0.100_10.00_0.100_R_1.000_BD_1_2.5_P0_1.0000.txt"
fname_tau_10 = fd_data + "solution_n_6_Q_0.100_100.00_0.100_R_1.000_BD_1_2.5_P0_1.0000.txt"

# Output
out_pdf_Q_tau = fd_data + "analyze_Q_tau.pdf"
pdf_Q_tau = backend_pdf.PdfPages(out_pdf_Q_tau)

# Plot 
FS_PLOT = 22
SHIFT_LEGEND_X = 1.5

"""
Load data
"""
t, tau_01_pe_kf, tau_01_pn_kf, tau_01_err_p_kf, pe_ref, pn_ref, \
tau_01_err_p_kf, tau_01_sqt_cov_p_kf  \
                                        = tools.extract_da_from_sol(fname_tau_001)
t, tau_1_pe_kf, tau_1_pn_kf, tau_1_err_p_kf, pe_ref, pn_ref, \
tau_1_err_p_kf, tau_1_sqt_cov_p_kf  \
                                        = tools.extract_da_from_sol(fname_tau_01)
t, tau_10_pe_kf, tau_10_pn_kf, tau_10_err_p_kf, pe_ref, pn_ref, \
tau_10_err_p_kf, tau_10_sqt_cov_p_kf  \
                                        = tools.extract_da_from_sol(fname_tau_1)
t, tau_100_pe_kf, tau_100_pn_kf, tau_100_err_p_kf, pe_ref, pn_ref, \
tau_100_err_p_kf, tau_100_sqt_cov_p_kf  \
                                        = tools.extract_da_from_sol(fname_tau_10)                                      

"""
Plot
"""
fig = tools.plot_figure([tau_01_pe_kf, tau_1_pe_kf, tau_10_pe_kf, tau_100_pe_kf, pe_ref], 
                        [tau_01_pn_kf, tau_1_pn_kf, tau_10_pn_kf, tau_100_pn_kf, pn_ref], 5, 
                       [len(tau_01_pe_kf),len(tau_1_pe_kf),len(tau_10_pe_kf),len(tau_100_pe_kf), len(pe_ref)], 
                       ["co-", "b*-", "rp-", "m+-", "g.-"], 
                       ["$\\tau =0.1 s$", "$\\tau =1 s$",
                        "$\\tau =10 s$", "$\\tau =100 s$", "ref"], 
                       "Position", "East ($m$)", 
                       "North ($m$)", '', 1, 1, 0, [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X+0.1)
pdf_Q_tau.savefig(fig)


fig = tools.plot_figure([t, t, t, t], 
                        [tau_1_err_p_kf, tau_10_err_p_kf, \
                         tau_1_sqt_cov_p_kf, tau_10_sqt_cov_p_kf], 4, 
                       [len(t),len(t),len(t),len(t)], 
                       ["bo-", "r+-", "c-.","m:"],  
                       ["$e_{p}$ ($\\tau =1 s$)",
                        "$e_{p}$ ($\\tau =100 s$)",
                        "$\sigma_{p}$ ($\\tau =1 s$)",
                        "$\sigma_{p}$ ($\\tau =100 s$)",
                        ], 
                       "Position error and uncertainty", "Time ($s$)", 
                       "Position error ($m$)", '', 1, 0, 0, [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X+0.2)
pdf_Q_tau.savefig(fig)

tau_01_cdf_x, tau_01_cdf_y = tools.cal_cdf(tau_01_err_p_kf)
tau_1_cdf_x, tau_1_cdf_y = tools.cal_cdf(tau_1_err_p_kf)
tau_10_cdf_x, tau_10_cdf_y = tools.cal_cdf(tau_10_err_p_kf)
tau_100_cdf_x, tau_100_cdf_y = tools.cal_cdf(tau_100_err_p_kf)

fig = tools.plot_figure([tau_01_cdf_x, tau_1_cdf_x, tau_10_cdf_x, tau_100_cdf_x], 
                       [tau_01_cdf_y, tau_1_cdf_y, tau_10_cdf_y, tau_100_cdf_y], 
                       4, [len(tau_01_cdf_x), len(tau_1_cdf_x),len(tau_10_cdf_x), len(tau_100_cdf_x)], \
                       ["co-", "b*-", "rp-", "m+-"],
                       ["$\\tau =0.1 s$", "$\\tau =1 s$",
                        "$\\tau =10 s$", "$\\tau =100 s$"],  
                       "CDF of position errors", \
                       "Position error ($m$)", "Probability", '', 1, 0, 0, [0,5], 0, [],
                       legend_loc="lower right", fs=FS_PLOT, 
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X-0.2, y_shift_legend=-0.02) 
pdf_Q_tau.savefig(fig)

"""
Statistics
"""
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(tau_01_err_p_kf)
print("\n tau_01_err_p_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(tau_1_err_p_kf)
print("\n tau_1_err_p_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(tau_10_err_p_kf)
print("\n tau_10_err_p_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(tau_100_err_p_kf)
print("\n tau_100_err_p_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))

"""
Close files
"""
pdf_Q_tau.close()