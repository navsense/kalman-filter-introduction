#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:32:45 2020

@author: youli
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as backend_pdf


def mkdir(path):
    import os
    path=path.strip()
    path=path.rstrip("/")

    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False
    

def central_heading(heading, type_heading):
    if type_heading=="deg": 
        value_180 = 180.0
    elif type_heading=="rad":
        value_180 = np.pi
    else:
        print ("Error, type_heading!='deg' or 'rad'\n")
        return -1
    while heading>value_180 or heading<=-value_180:
        if heading>value_180:
            heading = heading - value_180*2
        elif heading<=-value_180:
            heading = heading + value_180*2
    return heading  

def cal_rms(vec_):
    rms_ = np.sqrt(np.mean(np.square(vec_)))
    return rms_

def cal_statics(vec):
    range_v = max(vec) - min(vec)
    mean = np.mean(vec)
    median = np.median(vec)
    std = np.std(vec)
    rms = cal_rms(vec)
    
    v1 = np.sort(vec)
    d80 = v1[int(len(v1)*0.8)]
    d90 = v1[int(len(v1)*0.9)]
    d95 = v1[int(len(v1)*0.95)]
#    return (range_v, mean, median, std, rms, d80, d90, d95)
    return (mean, std, rms, d80, d95, np.max(vec))



def plot_figure(da_x, da_y, N, n_i, s_line_lengehd, s_name_legend, s_name_ti, s_name_x, s_name_y, s_path_output, \
                  if_grid, if_axis_equal, if_limit_x, x_limit, if_limit_y, y_limit, if_show=1, \
                  legend_loc="upper right", fs=14, 
                  if_use_shift=0, x_shift_legend=1.1, y_shift_legend=1.0,
                  if_fig_size=0, fig_width=10., fig_hei=7.,
                  if_plot_err_range_=0, da_y_err_range_=[], 
                  color_err_range_="b", color_err_lw_=2, color_err_capsize_=4):
    # Description:
    # da_x and da_y are lists which contain N lists, corresponding to each other
    # s_line_lengehd and s_name_legend are the list of strings, both has N nodes
    # n_i[k] records the lengh of data in for the k-th column of da_x, da_y
    N_x = len(da_x)
    N_y = len(da_y)
    N_name_legend = len(s_name_legend)
    
    if (N_x!=N or N_y!=N or len(n_i)!=N or len(s_line_lengehd)!=N or N_name_legend!=N):
           print ("Error, N=%d, N_x=%d, N_y=%d, N_n_i=%d, N_line_shape=%d, N_name_legend=%d\n" \
                  %(N, N_x, N_y, len(n_i), len(s_line_lengehd), N_name_legend))
           return -1
    if if_fig_size == 1:
        fig = plt.figure(figsize=(fig_width,fig_hei))
    else:   
        fig = plt.figure()
    line = []
    for i in range(0, N):
        if if_plot_err_range_ == 0:
            line.append(plt.plot(da_x[i][0:n_i[i]], da_y[i][0:n_i[i]], s_line_lengehd[i]))
#            line.append(plt.plot(da_x[i][0:n_i[i]], da_y[i][0:n_i[i]], \
#                        s_line_lengehd[i], label=s_name_legend[i]))
        else:
            line.append(plt.errorbar(da_x[i][0:n_i[i]], da_y[i][0:n_i[i]], \
                             fmt=s_line_lengehd[i], 
                             yerr=da_y_err_range_[i][0:n_i[i]],
                             ecolor=color_err_range_, elinewidth=color_err_lw_,
                             capsize=color_err_capsize_))
#    plt.legend(s_name_legend[0:N], prop={'size': 6})
    if if_use_shift == 1:
        plt.legend(s_name_legend[0:N], prop={'size': fs}, loc=legend_loc, bbox_to_anchor=(x_shift_legend, y_shift_legend)) #, 
    else:
        plt.legend(s_name_legend[0:N], prop={'size': fs}, loc=legend_loc) #, 
        
    plt.title(s_name_ti, fontsize=fs)
    plt.xlabel(s_name_x, fontsize=fs)
    plt.ylabel(s_name_y, fontsize=fs)
    plt.tick_params(labelsize=fs)
    
    if if_axis_equal: 
        plt.axis("equal")
    if if_limit_x:
        plt.xlim(x_limit[0], x_limit[1])
    if if_limit_y:
        plt.ylim(y_limit[0], y_limit[1])   
    if if_grid:
        plt.grid(True)
    if if_show:
        plt.show(block=False)
    return (fig)

def write_text_file(fname, da, list_format, delim):
    if not isinstance(da, list):
        print("Error in write_text_file, data is not a list")
        return 
    if len(da) != len(list_format):
        print("Error in write_text_file, len(da) != len(list_format)")
        return  
    n_line = len(da[0])
    for i in range(len(da)):
        if len(da[i]) != n_line:
            print("Error in write_text_file, len(da[i]) != n_line" %i)
            return
        
    fp = open(fname, 'w')
    for i in range(n_line):
        for j in range(len(list_format)):
            fp.write((list_format[j]+delim)   %da[j][i])
        fp.write("\n")
    fp.close()


def remove_blank_head_tail(str1):
    if len(str1)==0:
        return str1
    
    i_start = 0
    i_end = len(str1)
    
    # Remove blank in head
    for p in range(len(str1)):
        if str1[p] == '\t' or str1[p] == ' ':
            i_start = i_start+1
        else:
            break
    
    # Remove blank in tail
    for p in range(len(str1)):
        if str1[-1-p] == '\t' or str1[-1-p] == ' ':
            i_end = i_end-1
        else:
            break
    
    if i_start >= i_end:
        print ('Error! i_start >= i_end in remove_blank_head_tail.')
        return str1
        
    return str1[i_start:i_end]

def load_text_file(fname, list_format, delim, usecols=None):
    if len(delim) > 4:
        print ('Delimiter is too long.')
    if delim in [' ','  ','   ','    ']:
        fid = np.genfromtxt(fname,dtype='str',usecols=usecols).tolist()
    else:
        fid = np.genfromtxt(fname,dtype='str',delimiter=delim,usecols=usecols).tolist()
    
    if len(fid) == 0:
        da_out = []
        return (da_out)
    
    da = []
    if not isinstance(fid, list) :  #has 1 column and 1 row (1 data)
        da = [[fid]]
    elif not isinstance(fid[0], list) and len(list_format) == 1:  #has 1 column
        da = [fid]
    elif not isinstance(fid[0], list) and not len(list_format) == 1:  #has 1 row
        for i in range(len(fid)):
            da.append([fid[i]])
    else:
        da = [[] for _ in range(len(fid[0]))]
        for i in range(len(fid[0])):
            for j in range(len(fid)):
                da[i].append(fid[j][i])
    
    da_out = []
    if len(list_format) > len(da):
        print('Warning in load_text_file(), len(list_format) > len(fid[0]), \
              remaining elements in list_format are not loaded')
        list_format = list_format[:len(da)]
        
    for i in range(len(list_format)):
        if list_format[i]=='%s':
            col1 = [str(j) for j in da[i]]
            if not delim in [' ','  ','   ','    ']:
            # -----------remove blanks for str----------------    
                for k in range(len(col1)):
                    str1 = col1[k]
                    if delim != ' ':
                        col1[k] = remove_blank_head_tail(str1);
        elif list_format[i]=='%d':
            col1 = [int(j) for j in da[i]]
        elif list_format[i]=='%f':
            col1 = [float(j) for j in da[i]]
        else:
            print ('Not valid format.')
        da_out.append(col1)

    return (da_out)

def norm_of_vectors(v1_, v2_):
    n_ = len(v1_)
    v_ = [0.0 for _ in range(n_)]
    for i in range(n_):
        v_[i] = np.linalg.norm([v1_[i], v2_[i]])
    return v_

def scale_vector(v_, c_):
    n_ = len(v_)
    c_v_ = [0.0 for _ in range(n_)]
    for i in range(n_):
        c_v_[i] = v_[i] * c_
    return c_v_

def extract_da_from_sol(fname_sol_):
    
    format_in_sol_ = ['%f',
                  '%f','%f','%f','%f', '%f','%f', 
                  '%f','%f','%f','%f','%f','%f',
                  '%f','%f','%f','%f','%f','%f',
                  '%f','%f','%f',
                  '%f','%f','%f',
                  '%f','%f','%f',
                  '%f','%f','%f',
                  '%f','%f','%f',
                  ] 
    #          [t, 
    #          pe_ref, pn_ref, ve_ref, vn_ref, pe_meas, pn_meas,
    #          pe_kf, pn_kf, ve_kf, vn_kf, be_kf, bn_kf,
    #          sqt_cov_pe_kf, sqt_cov_pn_kf, sqt_cov_ve_kf, sqt_cov_vn_kf, sqt_cov_be_kf, sqt_cov_bn_kf,
    #          inno_kf[0], inno_kf[1], inno_norm_kf,
    #          sqrt_cov_inno_kf[0], sqrt_cov_inno_kf[1], sqrt_cov_inno_norm_kf,
    #          err_pe_meas, err_pn_meas, err_p_meas,
    #          err_pe_kf, err_pn_kf, err_p_kf,
    #          err_ve_kf, err_vn_kf, err_v_kf,
    #          ]

    da_ =  load_text_file(fname_sol_, format_in_sol_, ' ')
    
    t_ = da_[0]
    pe_ref_ = da_[1]; pn_ref_ = da_[2]; ve_ref_ = da_[3]; vn_ref_ = da_[4] 
    pe_meas_ = da_[5]; pn_meas_ = da_[6]
    pe_kf_ = da_[7]; pn_kf_ = da_[8]; ve_kf_ = da_[9]; vn_kf_ = da_[10]; 
    be_kf_ = da_[11]; bn_kf_ = da_[12]
    sqt_cov_pe_kf_ = da_[13]; sqt_cov_pn_kf_ = da_[14]; sqt_cov_ve_kf_ = da_[15]; sqt_cov_vn_kf_ = da_[16]; 
    sqt_cov_be_kf_ = da_[17]; sqt_cov_bn_kf_ = da_[18]
    inno_kf_e_ = da_[19]; inno_kf_n_ = da_[20]; inno_norm_kf_ = da_[21]; 
    sqrt_cov_inno_kf_e_ = da_[22]; sqrt_cov_inno_kf_n_ = da_[23]; sqrt_cov_inno_norm_kf_ = da_[24]
    err_pe_meas_ = da_[25]; err_pn_meas_ = da_[26]; err_p_meas_ = da_[27]; 
    err_pe_kf_ = da_[28]; err_pn_kf_ = da_[29]; err_p_kf_ = da_[30]; 
    err_ve_kf_ = da_[31]; err_vn_kf_ = da_[32]; err_v_kf_ = da_[33]; 
    
    sqt_cov_p_kf_ = norm_of_vectors(sqt_cov_pe_kf_, sqt_cov_pn_kf_)

    
    return t_, pe_kf_, pn_kf_, err_p_kf_, pe_ref_, pn_ref_, \
           err_p_kf_, sqt_cov_p_kf_




def cal_cdf(v_):
    import statsmodels.api as sm # recommended import according to the docs
    ecdf = sm.distributions.ECDF(v_)
    x = np.linspace(min(v_), max(v_))
    y = ecdf(x)
    return (x,y)