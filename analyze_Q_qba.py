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
fname_qba_0001 = fd_data + "solution_n_6_Q_0.100_10.00_0.001_R_1.000_BD_1_2.5_P0_1.0000.txt"
fname_qba_001   = fd_data + "solution_n_6_Q_0.100_10.00_0.010_R_1.000_BD_1_2.5_P0_1.0000.txt"
fname_qba_01  = fd_data + "solution_n_6_Q_0.100_10.00_0.100_R_1.000_BD_1_2.5_P0_1.0000.txt"
fname_qba_1 = fd_data + "solution_n_6_Q_0.100_10.00_1.000_R_1.000_BD_1_2.5_P0_1.0000.txt"

# Output
out_pdf_Q_qba = fd_data + "analyze_Q_qba.pdf"
pdf_Q_qba = backend_pdf.PdfPages(out_pdf_Q_qba)

# Plot 
FS_PLOT = 22
SHIFT_LEGEND_X = 1.5

"""
Load data
"""
t, qba_0001_pe_kf, qba_0001_pn_kf, qba_0001_err_p_kf, pe_ref, pn_ref, \
qba_0001_err_p_kf, qba_0001_sqt_cov_p_kf  \
                                        = tools.extract_da_from_sol(fname_qba_0001)
t, qba_001_pe_kf, qba_001_pn_kf, qba_001_err_p_kf, pe_ref, pn_ref, \
qba_001_err_p_kf, qba_001_sqt_cov_p_kf  \
                                        = tools.extract_da_from_sol(fname_qba_001)
t, qba_01_pe_kf, qba_01_pn_kf, qba_01_err_p_kf, pe_ref, pn_ref, \
qba_01_err_p_kf, qba_01_sqt_cov_p_kf  \
                                        = tools.extract_da_from_sol(fname_qba_01)
t, qba_1_pe_kf, qba_1_pn_kf, qba_1_err_p_kf, pe_ref, pn_ref, \
qba_1_err_p_kf, qba_1_sqt_cov_p_kf  \
                                        = tools.extract_da_from_sol(fname_qba_1)

"""
Plot
"""
fig = tools.plot_figure([qba_0001_pe_kf, qba_001_pe_kf, qba_01_pe_kf, qba_1_pe_kf, pe_ref], 
                        [qba_0001_pn_kf, qba_001_pn_kf, qba_01_pn_kf, qba_1_pn_kf, pn_ref], 5, 
                       [len(qba_0001_pe_kf),len(qba_001_pe_kf),len(qba_01_pe_kf),len(qba_1_pe_kf), len(pe_ref)], 
                       ["co-", "b*-", "r+-", "mp-", "g.-"], 
                       ["$q_{ba}=0.001 m/s^2$","$q_{ba}= 0.01 m/s^2$",
                        "$q_{ba}= 0.1 m/s^2$", "$q_{ba}= 1 m/s^2$", "ref"], 
                       "Position", "East ($m$)", 
                       "North ($m$)", '', 1, 1, 0, [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X+0.35)
pdf_Q_qba.savefig(fig)


fig = tools.plot_figure([t, t, t, t], 
                        [qba_0001_err_p_kf, qba_01_err_p_kf, \
                         qba_0001_sqt_cov_p_kf, qba_01_sqt_cov_p_kf], 4, 
                       [len(t),len(t),len(t),len(t)], 
                       ["bo-", "r+-", "c-.","m:"],  
                       ["$e_{p}$ ($q_{ba}= 0.001 m/s^2$)",
                        "$e_{p}$ ($q_{ba}= 0.1 m/s^2$)",
                        "$\sigma_{p}$ ($q_{ba}= 0.001 m/s^2$)",
                        "$\sigma_{p}$ ($q_{ba}= 0.1 m/s^2$)",
                        ], 
                       "Position error and uncertainty", "Time ($s$)", 
                       "Position error ($m$)", '', 1, 0, 0, [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X+0.4)
pdf_Q_qba.savefig(fig)


qba_0001_cdf_x, qba_0001_cdf_y = tools.cal_cdf(qba_0001_err_p_kf)
qba_001_cdf_x, qba_001_cdf_y = tools.cal_cdf(qba_001_err_p_kf)
qba_01_cdf_x, qba_01_cdf_y = tools.cal_cdf(qba_01_err_p_kf)
qba_1_cdf_x, qba_1_cdf_y = tools.cal_cdf(qba_1_err_p_kf)

fig = tools.plot_figure([qba_0001_cdf_x, qba_001_cdf_x, qba_01_cdf_x, qba_1_cdf_x], 
                       [qba_0001_cdf_y, qba_001_cdf_y, qba_01_cdf_y, qba_1_cdf_y], 
                       4, [len(qba_0001_cdf_x), len(qba_001_cdf_x),len(qba_01_cdf_x), len(qba_1_cdf_x)], \
                       ["co-", "b*-", "r+-", "mp-"],
                       ["$q_{ba}=0.001 m/s^2$","$q_{ba}= 0.01 m/s^2$",
                        "$q_{ba}= 0.1 m/s^2$", "$q_{ba}= 1 m/s^2$"], 
                       "CDF of position errors", \
                       "Position error ($m$)", "Probability", '', 1, 0, 0, [0,5], 0, [],
                       legend_loc="lower right", fs=FS_PLOT, 
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X-0.2, y_shift_legend=-0.02) 
pdf_Q_qba.savefig(fig)


"""
Statistics
"""
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(qba_0001_err_p_kf)
print("\n qba_0001_err_p_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(qba_001_err_p_kf)
print("\n qba_001_err_p_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(qba_01_err_p_kf)
print("\n qba_01_err_p_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(qba_1_err_p_kf)
print("\n qba_1_err_p_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))

"""
Close files
"""
pdf_Q_qba.close()