#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:49:03 2020
"""
import numpy as np

import tools
import matplotlib.backends.backend_pdf as backend_pdf


fd_data = "./data/"
tools.mkdir(fd_data)


# Output
out_pdf = fd_data + "kf_data.pdf"
pdf = backend_pdf.PdfPages(out_pdf)

fname_out = fd_data + "kf_data.txt"
format_out = ['%.3f','%.3f','%.3f','%.3f','%.3f','%.3f','%.3f'] 
            # time, pe_ref, pn_ref, ve_ref, vn_ref, pe_meas, pn_meas
FS_PLOT = 18
D2R = np.pi / 180.0

"""
Generate reference motion data
"""   
T_TOTAL = 100.0  # Total navigation time
T_TURN = 10.0  # Time length for each turn
D_T = 1.0      # Time interval
VEL = 1.0      # Speed

# Real-time states
pe = 0.0
pn = 0.0
ve = 0.0
vn = 0.0

# For recording the states
pe_ref = []
pn_ref = []
theta_ref = []
ve_ref = []
vn_ref = []
t_ref = []

# Set the trajectory
def set_theta_turn(t_cur_, t_start_turn_, theta_start_turn_):
    theta_ = theta_start_turn_ + D2R * 90.0 / T_TURN * (t_cur_ - t_start_turn_)
    return theta_

def set_thera_square(t_cur_):  
    if t_cur_ < 0.25 * T_TOTAL - 0.5 * T_TURN:    # moving toward north
        theta_ = D2R * 0.0
    elif t_cur_ < 0.25 * T_TOTAL + 0.5 * T_TURN:# Turn
        theta_ = set_theta_turn(t_cur_, 0.25 * T_TOTAL - 0.5 * T_TURN, 
                               D2R * 0.0)
    elif t_cur_ < 0.5 * T_TOTAL - 0.5 * T_TURN:   # moving toward east
        theta_ = D2R * 90.0
    elif t_cur_ < 0.5 * T_TOTAL + 0.5 * T_TURN:# Turn
        theta_ = set_theta_turn(t_cur_, 0.5 * T_TOTAL - 0.5 * T_TURN, 
                               D2R * 90.0)
    elif t_cur_ < 0.75 * T_TOTAL- 0.5 * T_TURN:  # moving toward south
        theta_ = D2R * 180.0
    elif t_cur_ < 0.75 * T_TOTAL + 0.5 * T_TURN:# Turn
        theta_ = set_theta_turn(t_cur_, 0.75 * T_TOTAL - 0.5 * T_TURN, 
                               D2R * 180.0)
    else:                        # moving toward west
        theta_ = D2R * 270.0 
        
    theta_ = tools.central_heading(theta_, "rad")
    return theta_ 

def update_vel_pos(pe_, pn_, theta_, v_, dt_):
    ve_ = v_ * np.sin(theta_)
    vn_ = v_ * np.cos(theta_)
    pe_ += ve_ * dt_
    pn_ += vn_ * dt_
    return pe_, pn_, ve_, vn_

def put_pos_vel_t_2_list(pe_, pn_, ve_, vn_, t_, 
                       pe_list_, pn_list_, ve_list_, vn_list_, t_list_):
    pe_list_.append(pe_)
    pn_list_.append(pn_)
    ve_list_.append(ve_)
    vn_list_.append(vn_)
    t_list_.append(t_)
    return pe_list_, pn_list_, ve_list_, vn_list_, t_list_
 
# Simulate reference velocity and position
for i in range(int(T_TOTAL/D_T)):
    t_cur = D_T * i
    pe_ref, pn_ref, ve_ref, vn_ref, t_ref = \
               put_pos_vel_t_2_list(pe, pn, ve, vn, t_cur,
                       pe_ref, pn_ref, ve_ref, vn_ref, t_ref)
               
    theta = set_thera_square(t_cur)
    pe, pn, ve, vn = update_vel_pos(pe, pn, theta, VEL, D_T)
        
    
fig = tools.plot_figure([t_ref, t_ref], [ve_ref, vn_ref], 2, [len(t_ref),len(t_ref)], 
                       ["b.-","r+-"], ["Ve","Vn"], "Reference velocity", 
                       "Time ($sec$)", "Velocity ($m/s$)", '', 1, 0, 
                       0, [], 0, [], fs=FS_PLOT)
pdf.savefig(fig)

#fig = tools.plot_figure([pe_ref], [pn_ref], 1, [len(pe_ref)], ["b.-"], 
#                       ["p"], "Reference Position", "x ($m$)", 
#                       "y ($m$)", '', 1, 1, 0, [], 0, [], fs=FS_PLOT)
#pdf.savefig(fig)
    
"""
Generate observations 
"""
# Add measurement noises
SIGMA_POS_MEAS = 1.0    # Added position noise

pe_meas = [0.0 for _ in range(len(pe_ref))]
pn_meas = [0.0 for _ in range(len(pe_ref))]

for i in range(len(pe_meas)):
    pe_meas[i] = pe_ref[i] + SIGMA_POS_MEAS * np.random.normal()
    pn_meas[i] = pn_ref[i] + SIGMA_POS_MEAS * np.random.normal()
    
fig = tools.plot_figure([pe_ref, pe_meas], [pn_ref, pn_meas], 2, [len(pe_ref),len(pe_meas)], ["b.-", "r*-"], 
                       ["Ref", "Obs"], "Position", "East ($m$)", 
                       "North ($m$)", '', 1, 1, 0, [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=1.1)
pdf.savefig(fig)

# Add blunders
NUM_BLUNDERS = 10
epoch_between_blunder = int (len(pe_ref) / (NUM_BLUNDERS + 2) )
SIGMA_BLUNDER = 5.0

for i in range(len(pe_meas)):
    if i != 0 and i % epoch_between_blunder == 0:
        print("Epoch %d has a blunder"%i)
        pe_meas[i] += SIGMA_BLUNDER * np.random.normal()
        pn_meas[i] += SIGMA_BLUNDER * np.random.normal()
fig = tools.plot_figure([pe_ref, pe_meas], [pn_ref, pn_meas], 2, [len(pe_ref),len(pe_meas)], ["b.-", "r*-"], 
                       ["Ref", "Obs"], "Position", "East ($m$)", 
                       "North ($m$)", '', 1, 1, 0, [], 0, [], fs=FS_PLOT,
                       if_use_shift=1, x_shift_legend=1.1)
pdf.savefig(fig)

"""
Output data
"""
da_out = [t_ref, pe_ref, pn_ref, ve_ref, vn_ref, pe_meas, pn_meas]

tools.write_text_file(fname_out, da_out, format_out, ' ')
    
pdf.close()


    
    
    