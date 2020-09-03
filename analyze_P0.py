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
fname_sf_p_0001  = fd_data + "solution_n_6_Q_0.100_10.00_0.100_R_1.000_BD_1_2.5_P0_0.0001.txt"
fname_sf_p_1   = fd_data + "solution_n_6_Q_0.100_10.00_0.100_R_1.000_BD_1_2.5_P0_1.0000.txt"
fname_sf_p_100  = fd_data + "solution_n_6_Q_0.100_10.00_0.100_R_1.000_BD_1_2.5_P0_100.0000.txt"
fname_sf_p_10000 = fd_data + "solution_n_6_Q_0.100_10.00_0.100_R_1.000_BD_1_2.5_P0_10000.0000.txt"


# Output
out_pdf_P0 = fd_data + "analyze_P0.pdf"
pdf_P0 = backend_pdf.PdfPages(out_pdf_P0)

# Plot 
FS_PLOT = 22
SHIFT_LEGEND_X = 1.5

"""
Load data
"""
t, sf_p_0001_pe_kf, sf_p_0001_pn_kf, sf_p_0001_err_p_kf, pe_ref, pn_ref, \
sf_p_0001_err_p_kf, sf_p_0001_sqt_cov_p_kf  \
                                        = tools.extract_da_from_sol(fname_sf_p_0001)
t, sf_p_1_pe_kf, sf_p_1_pn_kf, sf_p_1_err_p_kf, pe_ref, pn_ref, \
sf_p_1_err_p_kf, sf_p_1_sqt_cov_p_kf  \
                                        = tools.extract_da_from_sol(fname_sf_p_1)
t, sf_p_100_pe_kf, sf_p_100_pn_kf, sf_p_100_err_p_kf, pe_ref, pn_ref, \
sf_p_100_err_p_kf, sf_p_100_sqt_cov_p_kf  \
                                        = tools.extract_da_from_sol(fname_sf_p_100)
t, sf_p_10000_pe_kf, sf_p_10000_pn_kf, sf_p_10000_err_p_kf, pe_ref, pn_ref, \
sf_p_10000_err_p_kf, sf_p_10000_sqt_cov_p_kf  \
                                        = tools.extract_da_from_sol(fname_sf_p_10000)                                      

"""
Plot
"""
fig = tools.plot_figure([sf_p_0001_pe_kf, sf_p_1_pe_kf, sf_p_100_pe_kf, sf_p_10000_pe_kf, pe_ref], 
                        [sf_p_0001_pn_kf, sf_p_1_pn_kf, sf_p_100_pn_kf, sf_p_10000_pn_kf, pn_ref], 5, 
                       [len(sf_p_0001_pe_kf),len(sf_p_1_pe_kf),len(sf_p_100_pe_kf),len(sf_p_10000_pe_kf), len(pe_ref)], 
                       ["co-", "b*-", "rp-", "m+-", "g.-"], 
                       ["$s_{P0} =0.0001$", "$s_{P0} =1$",
                        "$s_{P0} =100$", "$s_{P0} =10000$", "ref"], 
                       "Position", "East ($m$)", 
                       "North ($m$)", '', 1, 1, 0, [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X+0.15)
pdf_P0.savefig(fig)

fig = tools.plot_figure([sf_p_0001_pe_kf, sf_p_1_pe_kf, sf_p_100_pe_kf, sf_p_10000_pe_kf, pe_ref], 
                        [sf_p_0001_pn_kf, sf_p_1_pn_kf, sf_p_100_pn_kf, sf_p_10000_pn_kf, pn_ref], 5, 
                       [len(sf_p_0001_pe_kf),len(sf_p_1_pe_kf),len(sf_p_100_pe_kf),len(sf_p_10000_pe_kf), len(pe_ref)], 
                       ["co-", "b*-", "rp-", "m+-", "g.-"], 
                       ["$s_{P0} =0.0001$", "$s_{P0} =1$",
                        "$s_{P0} =100$", "$s_{P0} =10000$", "ref"], 
                       "Position", "East ($m$)", 
                       "North ($m$)", '', 1, 0, 1, [-2,2], 1, [-2,20], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X+0.15, 
                       legend_loc="lower right", y_shift_legend=-0.02)
pdf_P0.savefig(fig)


fig = tools.plot_figure([t, t, t, t], 
                        [sf_p_0001_sqt_cov_p_kf, sf_p_1_sqt_cov_p_kf, \
                         sf_p_100_sqt_cov_p_kf, sf_p_10000_sqt_cov_p_kf], 4, 
                       [len(t),len(t),len(t),len(t)], 
                       ["co-", "b*-", "rp-", "m+-"],
                       ["$s_{P0} =0.0001$", "$s_{P0} =1$",
                        "$s_{P0} =100$", "$s_{P0} =10000$"],
                       "Position uncertainty", "Time ($s$)", 
                       "Uncertainty ($m$)", '', 1, 0, 0, [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X+0.2)
pdf_P0.savefig(fig)


sf_p_0001_cdf_x, sf_p_0001_cdf_y = tools.cal_cdf(sf_p_0001_err_p_kf)
sf_p_1_cdf_x, sf_p_1_cdf_y = tools.cal_cdf(sf_p_1_err_p_kf)
sf_p_100_cdf_x, sf_p_100_cdf_y = tools.cal_cdf(sf_p_100_err_p_kf)
sf_p_10000_cdf_x, sf_p_10000_cdf_y = tools.cal_cdf(sf_p_10000_err_p_kf)

fig = tools.plot_figure([sf_p_0001_cdf_x, sf_p_1_cdf_x, sf_p_100_cdf_x, sf_p_10000_cdf_x], 
                       [sf_p_0001_cdf_y, sf_p_1_cdf_y, sf_p_100_cdf_y, sf_p_10000_cdf_y], 
                       4, [len(sf_p_0001_cdf_x), len(sf_p_1_cdf_x),len(sf_p_100_cdf_x), len(sf_p_10000_cdf_x)], \
                       ["co-", "b*-", "rp-", "m+-"],
                       ["$s_{P0} =0.0001$", "$s_{P0} =1$",
                        "$s_{P0} =100$", "$s_{P0} =10000$"], 
                       "CDF of position errors", \
                       "Position error ($m$)", "Probability", '', 1, 0, 0, [0,5], 0, [],
                       legend_loc="lower right", fs=FS_PLOT, 
                       if_use_shift=1, x_shift_legend=SHIFT_LEGEND_X-0.2, y_shift_legend=-0.02) 
pdf_P0.savefig(fig)


"""
Statistics
"""
print("")
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(sf_p_0001_err_p_kf)
print("sf_p_0001_err_p_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(sf_p_1_err_p_kf)
print("sf_p_1_err_p_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(sf_p_100_err_p_kf)
print("sf_p_100_err_p_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))
(mean1, std1,  rms1, d801, d951, max1) = tools.cal_statics(sf_p_10000_err_p_kf)
print("sf_p_10000_err_p_kf: mean=%.1f, rms=%.1f, d80=%.1f, d95=%.1f, max=%.1f m"
      %(mean1, rms1, d801, d951, max1))
print("")

"""
Close files
"""
pdf_P0.close()