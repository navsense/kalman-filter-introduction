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
fname_BD_0 = fd_data + "solution_n_6_Q_0.100_10.00_0.100_R_1.000_BD_0_2.5_P0_1.0000.txt"
fname_BD_1 = fd_data + "solution_n_6_Q_0.100_10.00_0.100_R_1.000_BD_1_2.5_P0_1.0000.txt"

# Output
out_pdf_BD = fd_data + "analyze_BD.pdf"
pdf_BD = backend_pdf.PdfPages(out_pdf_BD)

# Plot 
FS_PLOT = 22
SHIFT_LEGEND_X = 1.5

"""
Load data
"""
t, bd_0_pe_kf, bd_0_pn_kf, bd_0_err_p_kf, pe_ref, pn_ref, \
bd_0_err_p_kf, bd_0_sqt_cov_p_kf  \
                                        = tools.extract_da_from_sol(fname_BD_0)
t, bd_1_pe_kf, bd_1_pn_kf, bd_1_err_p_kf, pe_ref, pn_ref, \
bd_1_err_p_kf, bd_1_sqt_cov_p_kf  \
                                        = tools.extract_da_from_sol(fname_BD_1)

"""
Plot
"""
fig = tools.plot_figure([bd_0_pe_kf, bd_1_pe_kf, pe_ref], 
                        [bd_0_pn_kf, bd_1_pn_kf, pn_ref], 3, 
                       [len(bd_0_pe_kf),len(bd_1_pe_kf),len(pe_ref)], 
                       ["r*-", "bo-","g.-", ], 
                       ["$p_{2D}$ (no BD)","$p_{2D}$ (with BD)", "ref"], 
                       "Position without/with blunder detection", "East ($m$)", 
                       "North ($m$)", '', 1, 1, 0, [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X+0.1)
pdf_BD.savefig(fig)

bd_0_cdf_x, bd_0_cdf_y = tools.cal_cdf(bd_0_err_p_kf)
bd_1_cdf_x, bd_1_cdf_y = tools.cal_cdf(bd_1_err_p_kf)

fig = tools.plot_figure([bd_0_cdf_x, bd_1_cdf_x], 
                       [bd_0_cdf_y, bd_1_cdf_y], 
                       2, [len(bd_0_cdf_x), len(bd_1_cdf_x)], \
                       ["r*-", "bo-"],
                       ["$e_{p}$ (no BD)","$e_{p}$ (with BD)"], 
                       "CDF of position errors", \
                       "Position error ($m$)", "Probability", '', 1, 0, 0, [0,5], 0, [],
                       legend_loc="lower right", fs=FS_PLOT, 
                       if_use_shift=1, y_shift_legend=-0.01) 
pdf_BD.savefig(fig)

"""
Statistics
"""
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(bd_0_err_p_kf)
print("\n bd_0_err_p_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(bd_1_err_p_kf)
print("\n bd_1_err_p_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))

"""
Close files
"""
pdf_BD.close()