# -*- coding: utf-8 -*-
"""
3D adaptive fractionation interpolation
"""

import tkinter as tk
import numpy as np
from scipy.stats import invgamma
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import pandas as pd
from .adaptfx_interpolation_3D import *


if __name__=='__main__':
    # Create a new window with the title "Address Entry Form"
    window = tk.Tk()
    window.title("Adaptive fractionation calculator")
    frm_probdis = tk.Frame(relief=tk.SUNKEN, borderwidth=3)
    frm_probdis.pack()
    
    data = []
    def select_file():
        filetypes = (
            ('csv files', '*.csv'),
            ('All files', '*.*')
        )
    
        filename = fd.askopenfilename(
            title='Open a file',
            initialdir='/',
            filetypes=filetypes)
    
        showinfo(
            title='Selected File',
            message=filename
        )

        ent_file.insert(0,filename)
        data = np.array(pd.read_csv(ent_file.get(),sep = ';'))
        variances = data.var(axis = 1)
        alpha,loc,beta = invgamma.fit(variances,floc = 0)
        ent_alpha.configure(state = 'normal')
        ent_beta.configure(state = 'normal')
        ent_alpha.delete(0, 'end')
        ent_beta.delete(0,'end')
        ent_alpha.insert(0,alpha)
        ent_beta.insert(0,beta)
        ent_alpha.configure(state = 'disabled')
        ent_beta.configure(state = 'disabled')

    def checkbox1():
        if var_radio.get() == 1:
            btn_open.configure(state = 'disabled')
            ent_alpha.configure(state = 'normal')
            ent_beta.configure(state = 'normal')
            ent_file.configure(state = 'disabled')
            ent_mean.configure(state = 'disabled')
            ent_std.configure(state = 'disabled')
        elif var_radio.get() == 2:
            ent_file.configure(state = 'normal')
            btn_open.configure(state = 'normal')
            ent_alpha.configure(state = 'disabled')
            ent_beta.configure(state = 'disabled')
            ent_mean.configure(state = 'disabled')
            ent_std.configure(state = 'disabled')
        elif var_radio.get() == 3:
            ent_mean.configure(state = 'normal')
            ent_std.configure(state = 'normal')
            ent_alpha.configure(state = 'disabled')
            ent_beta.configure(state = 'disabled')
            btn_open.configure(state = 'disabled')
            ent_file.configure(state = 'disabled')
            
    #assign infobutton commands

    def info1():
        lbl_info["text"] = 'Insert the path of your prior patient data in here. \nThis is only needed, if the checkbox for prior data is marked. \nIf not, one can directly insert the hyperparameters below. \nThe file with the prior data must be of the shape n x k,\nwhere each new patient n is on a row and each fraction for patient n is in column k'
    def info2():
        lbl_info["text"] = 'Insert the mean of the sparing factor distribution. \nwith this option the distribution is not updated'
    def info3():
        lbl_info["text"] = 'Insert the standard deviation of the sparing factor distribution. \nwith this option the distribution is not updated'
    def info4():
        lbl_info["text"] = 'Insert the shape parameter for the inverse-gamme distribution.'
    def info5():
        lbl_info["text"] = 'Insert the scale parameter for the inverse-gamme distribution.'

    info_funcs = [info1,info2,info3,info4,info5]   
    info_buttons =["btn_path","btn_mean","btn_std","btn_shae","btn_scale"]
    for idx in range(len(info_funcs)):
        globals()[info_buttons[idx]] = tk.Button(master = frm_probdis,text = '?',command = info_funcs[idx])
        globals()[info_buttons[idx]].grid(row=idx+1,column=4)

    var_radio = tk.IntVar()
    var_radio.set(1)
    hyper_insert = tk.Radiobutton(master = frm_probdis,text = 'hyperparameters',justify = "left",variable = var_radio, value = 1, command = checkbox1)
    hyper_insert.grid(row= 0, column = 0)
    file_insert = tk.Radiobutton(master = frm_probdis,text = 'prior data',justify = "left",variable = var_radio, value = 2, command = checkbox1)
    file_insert.grid(row= 0, column = 1)
    fixed_insert = tk.Radiobutton(master = frm_probdis, text = 'define normal distribution',justify= "left",variable = var_radio,value = 3, command = checkbox1)
    fixed_insert.grid(row= 0, column = 2)

    
    # open button
    lbl_open = tk.Label(master = frm_probdis, text = 'load patient data for prior')
    lbl_open.grid(row = 1, column = 0)
    btn_open = tk.Button(
        frm_probdis,
        text='Open a File',
        command=select_file)
    ent_file = tk.Entry(master=frm_probdis, width=20)
    btn_open.grid(row = 1, column = 1)
    
    lbl_mean = tk.Label(master = frm_probdis, text = 'mean of normal distribution:')
    lbl_mean.grid(row=2,column = 0)
    ent_mean = tk.Entry(master = frm_probdis, width = 30)
    ent_mean.grid(row = 2, column = 1,columnspan = 2)
    
    lbl_std = tk.Label(master = frm_probdis, text = 'std of normal distribution:')
    lbl_std.grid(row=3,column = 0)
    ent_std = tk.Entry(master = frm_probdis, width = 30)
    ent_std.grid(row = 3, column = 1,columnspan = 2)
    
    lbl_alpha = tk.Label(master = frm_probdis, text = "shape of inverse-gamma distribution (alpha):")
    lbl_alpha.grid(row=4,column = 0)
    ent_alpha = tk.Entry(master = frm_probdis, width = 30)
    ent_alpha.grid(row = 4, column = 1,columnspan = 2)
    
    lbl_beta = tk.Label(master = frm_probdis, text = "scale of inverse-gamma distribution (beta):")
    lbl_beta.grid(row=5,column = 0)
    ent_beta = tk.Entry(master = frm_probdis, width = 30)
    ent_beta.grid(row = 5, column = 1,columnspan = 2)
    
    btn_open.configure(state = 'disabled')
    ent_alpha.configure(state = 'normal')
    ent_beta.configure(state = 'normal')
    ent_file.configure(state = 'disabled')
    ent_mean.configure(state = 'disabled')
    ent_std.configure(state = 'disabled')
    ent_alpha.insert(0,"0.6133124926763415")
    ent_beta.insert(0,"0.0004167968394550765")
    
    

    
    # Create a new frame `frm_form` to contain the Label
    # and Entry widgets for entering variable values
    frm_form = tk.Frame(relief=tk.SUNKEN, borderwidth=3)
    # Pack the frame into the window
    frm_form.pack()
    
    frm_buttons = tk.Frame()
    frm_buttons.pack(fill=tk.X, ipadx=5, ipady=5)
    
    frm_output = tk.Frame(relief=tk.SUNKEN, borderwidth = 3)
    frm_output.pack(fill=tk.BOTH, ipadx = 10, ipady = 10)

    #add label and entry for filename
    label = tk.Label(master=frm_form, text='file path of prior patients')
    ent_file = tk.Entry(master=frm_form, width=50)
    label.grid(row=0, column=0, sticky="e")
    ent_file.grid(row=0, column=1)
    
#assign infobutton commands

    def info10():
        lbl_info["text"] = 'Insert the sparing factors that were observed so far.\n The sparing factor of the planning session must be included!.\nThe sparing factors must be separated by spaces e.g.:\n1.1 0.95 0.88\nFor a whole plan 6 sparing factors are needed.'
    def info11():
        lbl_info["text"] = 'Insert the alpha-beta ratio of the tumor tissue.'
    def info12():
        lbl_info["text"] = 'Insert the alpha-beta ratio of the dose-limiting Organ at risk.'
    def info13():
        lbl_info["text"] = 'Insert the maximum dose delivered to the dose-limiting OAR in BED.'
    def info14():
        lbl_info["text"] = 'Insert the prescribed biological effectiv dose to be delivered to the tumor.'
    def info15():
        lbl_info["text"] = 'Insert the accumulated tumor BED so far. (If fraction one, it is zero).'
    def info16():
        lbl_info["text"] = 'Insert the accumulated OAR dose so far. (If fraction one, it is zero).'
    info_funcs = [info10,info11,info12,info13,info14,info15,info16]    
    info_buttons =["btn_sf","btn_abt","btn_abn","btn_OARlimit","btn_tumorlimit","btn_tumorBED","btn_OARBED"]
    
    # List of field labels
    labels = [
        "sparing factors separated by spaces:",
        "alpha-beta ratio of tumor:",
        "alpha-beta ratio of OAR:", 
        "OAR limit:",
        "prescribed tumor dose:",
        "accumulated tumor dose:",
        "accumulated OAR dose:"
    ]
    entries = ["ent_sf","ent_abt","ent_abn","ent_OARlimit","ent_tumorlimit","ent_BED_tumor","ent_BED_OAR"]
    ent_sf, ent_abt, ent_abn, ent_OARlimit,ent_tumorlimit, ent_BED_tumor, ent_BED_OAR = 0,0,0,0,0,0,0
    example_list = ["sparing factors separated by space",10,3,90,72,"only needed if we calculate the dose for a single fraction","only needed if we calculate the dose for a single fraction"]
    # Loop over the list of field labels
    for idx, text in enumerate(labels):
        # Create a Label widget with the text from the labels list
        label = tk.Label(master=frm_form, text=text)
        # Create an Entry widget
        globals()[entries[idx]] = tk.Entry(master=frm_form, width=50)
        
        # Use the grid geometry manager to place the Label and
        # Entry widgets in the row whose index is idx
        label.grid(row=idx, column=0, sticky="e")
        globals()[entries[idx]].grid(row=idx, column=1)
        globals()[entries[idx]].insert(0,f"{example_list[idx]}")
        globals()[info_buttons[idx]] = tk.Button(master = frm_form,text = '?',command = info_funcs[idx])
        globals()[info_buttons[idx]].grid(row=idx,column=2)
        
        
    def compute_plan():
        alpha = float(ent_alpha.get())
        beta = float(ent_beta.get())
        if var_radio.get() != 3:
            fixed_prob = 0
            fixed_mean = 0
            fixed_std = 0
        elif var_radio.get() == 3:
            fixed_prob = 1
            fixed_mean = float(ent_mean.get())
            fixed_std = float(ent_std.get())
        try:
            global lbl_output
            lbl_output.destroy()
        except:
            pass   
        if var.get() == 0:
            try:
                sparing_factors_str = (ent_sf.get()).split()
                sparing_factors = [float(i) for i in sparing_factors_str]
                abt = float(ent_abt.get())
                abn = float(ent_abn.get())
                OAR_limit = float(ent_OARlimit.get())
                tumor_limit = float(ent_tumorlimit.get())
                [tumor_doses,OAR_doses,physical_doses] = whole_plan(sparing_factors,abt,abn,OAR_limit,tumor_limit,alpha,beta,fixed_prob,fixed_mean,fixed_std)
                lbl_output = tk.Frame()
                lbl_output.pack()
                frame = tk.Frame(master = lbl_output, relief = tk.RAISED, borderwidth = 1)
                frame.grid(row = 0,column = 0)
                label= tk.Label(master = frame, text = "fraction number")
                label.pack()
                frame = tk.Frame(master = lbl_output, relief = tk.RAISED, borderwidth = 1)
                frame.grid(row = 0,column = 1)
                label= tk.Label(master = frame, text = "sparing factor")
                label.pack()
                frame = tk.Frame(master = lbl_output, relief = tk.RAISED, borderwidth = 1)
                frame.grid(row = 0,column = 2)
                label= tk.Label(master = frame, text = "physical dose delivered to PTV95")
                label.pack()
                frame = tk.Frame(master = lbl_output, relief = tk.RAISED, borderwidth = 1)
                frame.grid(row = 0,column = 3)
                label= tk.Label(master = frame, text = "BED delivered to tumor")
                label.pack()
                frame = tk.Frame(master = lbl_output, relief = tk.RAISED, borderwidth = 1)
                frame.grid(row = 0,column = 4)
                label= tk.Label(master = frame, text = "BED delivered to OAR")
                label.pack()
                for i in range(1,6):
                    for j in range(5):
                        if j == 0:
                            frame = tk.Frame(master = lbl_output)
                            frame.grid(row = i,column = 0)
                            label = tk.Label(master= frame, text = f"fraction {i}")
                            label.pack()
                        elif j == 1:
                            frame = tk.Frame(master = lbl_output)
                            frame.grid(row = i,column = 1)
                            label = tk.Label(master= frame, text = f" {sparing_factors[i]}")
                            label.pack()
                        elif j == 2:
                            frame = tk.Frame(master = lbl_output)
                            frame.grid(row = i,column = 2)
                            label = tk.Label(master= frame, text = f" {np.round(physical_doses[i-1],2)}")
                            label.pack()    
                        elif j == 3:
                            frame = tk.Frame(master = lbl_output)
                            frame.grid(row = i,column = 3)
                            label = tk.Label(master= frame, text = f" {np.round(tumor_doses[i-1],2)}")
                            label.pack()
                        elif j == 4:
                            frame = tk.Frame(master = lbl_output)
                            frame.grid(row = i,column = 4)
                            label = tk.Label(master= frame, text = f" {np.round(OAR_doses[i-1],2)}")
                            label.pack()
                frame = tk.Frame(master = lbl_output, relief = tk.RAISED, borderwidth = 1)
                frame.grid(row = 6,column = 0)
                label= tk.Label(master = frame, text = "accumulated doses")
                label.pack()
                frame = tk.Frame(master = lbl_output, relief = tk.RAISED, borderwidth = 1)
                frame.grid(row = 6  ,column = 3)
                label = tk.Label(master= frame, text = f" {np.round(np.sum(tumor_doses),2)}")
                label.pack()
                frame = tk.Frame(master = lbl_output, relief = tk.RAISED, borderwidth = 1)
                frame.grid(row = 6  ,column = 4)
                label = tk.Label(master= frame, text = f" {np.sum(OAR_doses)}")
                label.pack()
                
            except ValueError:
                lbl_info["text"] = "please enter correct values. Use the ? boxes for further information."
        else:
            try:
                sparing_factors_str = (ent_sf.get()).split()
                sparing_factors = [float(i) for i in sparing_factors_str]
                abt = float(ent_abt.get())
                abn = float(ent_abn.get())
                OAR_limit = float(ent_OARlimit.get())
                tumor_limit = float(ent_tumorlimit.get())
                BED_tumor = float(ent_BED_tumor.get())
                BED_OAR = float(ent_BED_OAR.get())
                [optimal_dose,total_dose_delivered_tumor,total_dose_delivered_OAR,tumor_dose,OAR_dose] =  value_eval(len(sparing_factors)-1,BED_OAR,BED_tumor,sparing_factors,abt,abn,OAR_limit,tumor_limit,alpha,beta,fixed_prob,fixed_mean,fixed_std)
                lbl_info["text"] = f"The optimal dose for fraction {len(sparing_factors)-1},  = {optimal_dose}\naccumulated dose in tumor = {total_dose_delivered_tumor}\naccumulated dose OAR = {total_dose_delivered_OAR}"
            except ValueError:
                lbl_info["text"] = "please enter correct values. Use the ? boxes for further information."        
    ent_BED_OAR.configure(state = 'disabled') #the standard calculation is a whole plan calculation
    ent_BED_tumor.configure(state = 'disabled')
    def checkbox():
        if var.get() == 0:
            ent_BED_tumor.configure(state = 'disabled')
            ent_BED_OAR.configure(state = 'disabled')
        else:
            ent_BED_tumor.configure(state = 'normal')
            ent_BED_OAR.configure(state = 'normal')
    # Create a new frame `frm_buttons` to contain the compute button
    # whole window in the horizontal direction and has
    # 5 pixels of horizontal and vertical padding.
    frm_buttons = tk.Frame()
    frm_buttons.pack(fill=tk.X, ipadx=5, ipady=5)
            
    btn_compute = tk.Button(master=frm_buttons, text="compute plan", command = compute_plan)
    btn_compute.pack(side=tk.BOTTOM, ipadx=10)
    var = tk.IntVar()
    chk_single_fraction = tk.Checkbutton(master = frm_buttons,text = "Calculate dose only for actual fraction",variable = var,onvalue = 1,offvalue = 0, command=checkbox)
    chk_single_fraction.pack(side = tk.BOTTOM, padx = 10, ipadx = 10)
    # frm_output = tk.Frame(relief=tk.SUNKEN, borderwidth = 3)
    # frm_output.pack(fill=tk.BOTH, ipadx = 10, ipady = 10)
    

    
    
    lbl_info = tk.Label(master = frm_output, text = "There are several default values set. Only the sparing factors have to been inserted.\nThis program might take some minutes to calculate")
    lbl_info.pack()
    lbl_output = tk.Frame()

    
    
    # Start the application
    window.mainloop()
