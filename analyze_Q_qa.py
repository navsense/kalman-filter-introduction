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
fname_VRW_0001 = fd_data + "solution_n_6_Q_0.001_10.00_0.100_R_1.000_BD_1_2.5_P0_1.0000.txt"
fname_VRW_001   = fd_data + "solution_n_6_Q_0.010_10.00_0.100_R_1.000_BD_1_2.5_P0_1.0000.txt"
fname_VRW_01  = fd_data + "solution_n_6_Q_0.100_10.00_0.100_R_1.000_BD_1_2.5_P0_1.0000.txt"
fname_VRW_1 = fd_data + "solution_n_6_Q_1.000_10.00_0.100_R_1.000_BD_1_2.5_P0_1.0000.txt"

# Output
out_pdf_Q_qa = fd_data + "analyze_Q_qa.pdf"
pdf_Q_qa = backend_pdf.PdfPages(out_pdf_Q_qa)

# Plot 
FS_PLOT = 22
SHIFT_LEGEND_X = 1.5

"""
Load data
"""
t, vrw_0001_pe_kf, vrw_0001_pn_kf, vrw_0001_err_p_kf, pe_ref, pn_ref, \
vrw_0001_err_p_kf, vrw_0001_sqt_cov_p_kf  \
                                        = tools.extract_da_from_sol(fname_VRW_0001)
t, vrw_001_pe_kf, vrw_001_pn_kf, vrw_001_err_p_kf, pe_ref, pn_ref, \
vrw_001_err_p_kf, vrw_001_sqt_cov_p_kf  \
                                        = tools.extract_da_from_sol(fname_VRW_001)
t, vrw_01_pe_kf, vrw_01_pn_kf, vrw_01_err_p_kf, pe_ref, pn_ref, \
vrw_01_err_p_kf, vrw_01_sqt_cov_p_kf  \
                                        = tools.extract_da_from_sol(fname_VRW_01)
t, vrw_1_pe_kf, vrw_1_pn_kf, vrw_1_err_p_kf, pe_ref, pn_ref, \
vrw_1_err_p_kf, vrw_1_sqt_cov_p_kf  \
                                        = tools.extract_da_from_sol(fname_VRW_1)

"""
Plot
"""
fig = tools.plot_figure([vrw_0001_pe_kf, vrw_001_pe_kf, vrw_01_pe_kf, vrw_1_pe_kf, pe_ref], 
                        [vrw_0001_pn_kf, vrw_001_pn_kf, vrw_01_pn_kf, vrw_1_pn_kf, pn_ref], 5, 
                       [len(vrw_0001_pe_kf),len(vrw_001_pe_kf),len(vrw_01_pe_kf),len(vrw_1_pe_kf), len(pe_ref)], 
                       ["co-", "b*-", "r+-", "mp-", "g.-"], 
                       ["$q_a=0.001 m/s/\u221A s$","$q_a= 0.01 m/s/\u221A s$","$q_a= 0.1 m/s/\u221A s$", "$q_a= 1 m/s/\u221A s$", "ref"], 
                       "Position", "East ($m$)", 
                       "North ($m$)", '', 1, 1, 0, [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X+0.3)
pdf_Q_qa.savefig(fig)

fig = tools.plot_figure([t, t, t, t], 
                        [vrw_0001_err_p_kf, vrw_01_err_p_kf, \
                         vrw_0001_sqt_cov_p_kf, vrw_01_sqt_cov_p_kf], 4, 
                       [len(t),len(t),len(t),len(t)], 
                       ["bo-", "r+-", "c-.","m:"],  
                       ["$e_{p}$ ($q_a= 0.001 m/s/\u221A s$)",
                        "$e_{p}$ ($q_a= 0.1 m/s/\u221A s$)",
                        "$\sigma_{p}$ ($q_a= 0.001 m/s/\u221A s$)",
                        "$\sigma_{p}$ ($q_a= 0.1 m/s/\u221A s$)",
                        ], 
                       "Position error and uncertainty", "Time ($s$)", 
                       "Position error ($m$)", '', 1, 0, 0, [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X+0.4)
pdf_Q_qa.savefig(fig)


vrw_0001_cdf_x, vrw_0001_cdf_y = tools.cal_cdf(vrw_0001_err_p_kf)
vrw_001_cdf_x, vrw_001_cdf_y = tools.cal_cdf(vrw_001_err_p_kf)
vrw_01_cdf_x, vrw_01_cdf_y = tools.cal_cdf(vrw_01_err_p_kf)
vrw_1_cdf_x, vrw_1_cdf_y = tools.cal_cdf(vrw_1_err_p_kf)

fig = tools.plot_figure([vrw_0001_cdf_x, vrw_001_cdf_x, vrw_01_cdf_x, vrw_1_cdf_x], 
                       [vrw_0001_cdf_y, vrw_001_cdf_y, vrw_01_cdf_y, vrw_1_cdf_y], 
                       4, [len(vrw_0001_cdf_x), len(vrw_001_cdf_x),len(vrw_01_cdf_x), len(vrw_1_cdf_x)], \
                       ["co-", "b*-", "r+-", "mp-"],
                       ["$q_a=0.001 m/s/\u221A s$","$q_a= 0.01 m/s/\u221A s$","$q_a= 0.1 m/s/\u221A s$", "$q_a= 1 m/s/\u221A s$"], 
                       "CDF of position errors", \
                       "Position error ($m$)", "Probability", '', 1, 0, 0, [0,5], 0, [],
                       legend_loc="lower right", fs=FS_PLOT, 
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X-0.2, y_shift_legend=-0.02) 
pdf_Q_qa.savefig(fig)


"""
Statistics
"""
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(vrw_0001_err_p_kf)
print("\n vrw_0001_err_p_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(vrw_001_err_p_kf)
print("\n vrw_001_err_p_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(vrw_01_err_p_kf)
print("\n vrw_01_err_p_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(vrw_1_err_p_kf)
print("\n vrw_1_err_p_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))

"""
Close files
"""
pdf_Q_qa.close()