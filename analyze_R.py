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
fname_R_01 = fd_data + "solution_n_6_Q_0.100_10.00_0.100_R_0.100_BD_0_2.5_P0_1.0000.txt"
fname_R_1   = fd_data + "solution_n_6_Q_0.100_10.00_0.100_R_1.000_BD_0_2.5_P0_1.0000.txt"
fname_R_10  = fd_data + "solution_n_6_Q_0.100_10.00_0.100_R_10.000_BD_0_2.5_P0_1.0000.txt"
fname_R_100 = fd_data + "solution_n_6_Q_0.100_10.00_0.100_R_100.000_BD_0_2.5_P0_1.0000.txt"

# Output
out_pdf_R = fd_data + "analyze_R.pdf"
pdf_R = backend_pdf.PdfPages(out_pdf_R)

# Plot 
FS_PLOT = 22
SHIFT_LEGEND_X = 1.5

"""
Load data
"""
t, r_01_pe_kf, r_01_pn_kf, r_01_err_p_kf, pe_ref, pn_ref, \
r_01_err_p_kf, r_01_sqt_cov_p_kf  \
                                        = tools.extract_da_from_sol(fname_R_01)
t, r_1_pe_kf, r_1_pn_kf, r_1_err_p_kf, pe_ref, pn_ref, \
r_1_err_p_kf, r_1_sqt_cov_p_kf  \
                                        = tools.extract_da_from_sol(fname_R_1)
t, r_10_pe_kf, r_10_pn_kf, r_10_err_p_kf, pe_ref, pn_ref, \
r_10_err_p_kf, r_10_sqt_cov_p_kf  \
                                        = tools.extract_da_from_sol(fname_R_10)
t, r_100_pe_kf, r_100_pn_kf, r_100_err_p_kf, pe_ref, pn_ref, \
r_100_err_p_kf, r_100_sqt_cov_p_kf  \
                                        = tools.extract_da_from_sol(fname_R_100)

"""
Plot
"""
fig = tools.plot_figure([r_01_pe_kf, r_1_pe_kf, r_10_pe_kf, r_100_pe_kf, pe_ref], 
                        [r_01_pn_kf, r_1_pn_kf, r_10_pn_kf, r_100_pn_kf, pn_ref], 5, 
                       [len(r_01_pe_kf),len(r_1_pe_kf),len(r_10_pe_kf),len(r_100_pe_kf), len(pe_ref)], 
                       ["c*-", "bo-", "r+-", "mp-", "g.-"], 
                       ["$\sigma_{R}=0.1 m$","$\sigma_{R}= 1 m$","$\sigma_{R}= 10 m$", "$\sigma_{R}= 100 m$", "ref"], 
                       "Position", "East ($m$)", 
                       "North ($m$)", '', 1, 1, 0, [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X+0.1)
pdf_R.savefig(fig)

fig = tools.plot_figure([t, t, t, t], 
                        [r_1_err_p_kf, r_10_err_p_kf, \
                         r_1_sqt_cov_p_kf, r_10_sqt_cov_p_kf], 4, 
                       [len(t),len(t),len(t),len(t)], 
                       ["bo-", "r+-", "c-.","m:"], 
                       ["$e_{p}$ ($\sigma_{R}= 1 m$)",
                        "$e_{p}$ ($\sigma_{R}= 10 m$)",
                        "$\sigma_{p}$ ($\sigma_{R}= 1 m$)",
                        "$\sigma_{p}$ ($\sigma_{R}= 10 m$)",
                        ], 
                       "Position error and uncertainty", "Time ($s$)", 
                       "Position error ($m$)", '', 1, 0, 0, [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X+0.3)
pdf_R.savefig(fig)

r_01_cdf_x, r_01_cdf_y = tools.cal_cdf(r_01_err_p_kf)
r_1_cdf_x, r_1_cdf_y = tools.cal_cdf(r_1_err_p_kf)
r_10_cdf_x, r_10_cdf_y = tools.cal_cdf(r_10_err_p_kf)
r_100_cdf_x, r_100_cdf_y = tools.cal_cdf(r_100_err_p_kf)

fig = tools.plot_figure([r_01_cdf_x, r_1_cdf_x, r_10_cdf_x, r_100_cdf_x], 
                       [r_01_cdf_y, r_1_cdf_y, r_10_cdf_y, r_100_cdf_y], 
                       4, [len(r_01_cdf_x), len(r_1_cdf_x),len(r_10_cdf_x), len(r_100_cdf_x)], \
                       ["c*-", "bo-", "r+-", "mp-"],
                       ["$\sigma_{R}=0.1 m$","$\sigma_{R}= 1 m$","$\sigma_{R}= 10 m$", "$\sigma_{R}= 100 m$"], 
                       "CDF of position errors", \
                       "Position error ($m$)", "Probability", '', 1, 0, 0, [0,5], 0, [],
                       legend_loc="lower right", fs=FS_PLOT, 
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X, y_shift_legend=-0.02) 
pdf_R.savefig(fig)

"""
Statistics
"""
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(r_01_err_p_kf)
print("\n r_01_err_p_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(r_1_err_p_kf)
print("\n r_1_err_p_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(r_10_err_p_kf)
print("\n r_10_err_p_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(r_100_err_p_kf)
print("\n r_100_err_p_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))

"""
Close files
"""
pdf_R.close()