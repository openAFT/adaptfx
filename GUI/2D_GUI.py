"""
GUI for 2D adaptive fractionation with minimum and maximum dose
"""
# The whole GUI needs to be class like to have a working scrollbar. Update coming soon!
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

import interpol2D_OAR as intOAR
import interpol2D_tumor as inttumor
import numpy as np
import pandas as pd
from scipy.stats import gamma


class VerticalScrolledFrame(tk.Frame):
    """A pure Tkinter scrollable frame that actually works!
    * Use the 'interior' attribute to place widgets inside the scrollable frame
    * Construct and pack/place/grid normally
    * This frame only allows vertical scrolling

    """

    def __init__(self, parent, *args, **kw):
        tk.Frame.__init__(self, parent, *args, **kw)

        # create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL)
        vscrollbar.pack(fill=tk.Y, side=tk.RIGHT, expand=tk.FALSE)
        canvas = tk.Canvas(
            self, bd=0, highlightthickness=0, yscrollcommand=vscrollbar.set, height=1000
        )
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.TRUE)
        vscrollbar.config(command=canvas.yview)
        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.interior = interior = tk.Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior, anchor=tk.NW)

        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.config(width=interior.winfo_reqwidth())

        interior.bind("<Configure>", _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())

        canvas.bind("<Configure>", _configure_canvas)


class GUI2Dextended:
    def __init__(self, master):
        self.master = master
        master.title("2D adaptive fractionation calculator extended")
        self.frame = VerticalScrolledFrame(master)
        self.frame.pack()
        self.frm_probdis = tk.Frame(
            master=self.frame.interior, relief=tk.SUNKEN, borderwidth=3
        )
        self.frm_probdis.pack()
        self.data = []
        self.info_funcs = [self.info1, self.info2, self.info3, self.info4, self.info5]
        self.info_buttons = ["btn_path", "btn_mean", "btn_std", "btn_shae", "btn_scale"]
        for idx in range(len(self.info_funcs)):
            globals()[self.info_buttons[idx]] = tk.Button(
                master=self.frm_probdis, text="?", command=self.info_funcs[idx]
            )
            globals()[self.info_buttons[idx]].grid(row=idx + 1, column=4)
        for idx in range(len(self.info_funcs)):
            globals()[self.info_buttons[idx]] = tk.Button(
                master=self.frm_probdis, text="?", command=self.info_funcs[idx]
            )
            globals()[self.info_buttons[idx]].grid(row=idx + 1, column=4)

        self.var_radio = tk.IntVar()
        self.var_radio.set(1)
        self.hyper_insert = tk.Radiobutton(
            master=self.frm_probdis,
            text="hyperparameters",
            justify="left",
            variable=self.var_radio,
            value=1,
            command=self.checkbox1,
        )
        self.hyper_insert.grid(row=0, column=0)
        self.file_insert = tk.Radiobutton(
            master=self.frm_probdis,
            text="prior data",
            justify="left",
            variable=self.var_radio,
            value=2,
            command=self.checkbox1,
        )
        self.file_insert.grid(row=0, column=1)
        self.fixed_insert = tk.Radiobutton(
            master=self.frm_probdis,
            text="define normal distribution",
            justify="left",
            variable=self.var_radio,
            value=3,
            command=self.checkbox1,
        )
        self.fixed_insert.grid(row=0, column=2)

        # open button
        self.lbl_open = tk.Label(
            master=self.frm_probdis, text="load patient data for prior"
        )
        self.lbl_open.grid(row=1, column=0)
        self.btn_open = tk.Button(
            self.frm_probdis, text="Open a File", command=self.select_file
        )
        self.ent_file = tk.Entry(master=self.frm_probdis, width=20)
        self.btn_open.grid(row=1, column=1)

        self.lbl_mean = tk.Label(
            master=self.frm_probdis, text="mean of normal distribution:"
        )
        self.lbl_mean.grid(row=2, column=0)
        self.ent_mean = tk.Entry(master=self.frm_probdis, width=30)
        self.ent_mean.grid(row=2, column=1, columnspan=2)

        self.lbl_std = tk.Label(
            master=self.frm_probdis, text="std of normal distribution:"
        )
        self.lbl_std.grid(row=3, column=0)
        self.ent_std = tk.Entry(master=self.frm_probdis, width=30)
        self.ent_std.grid(row=3, column=1, columnspan=2)

        self.lbl_alpha = tk.Label(
            master=self.frm_probdis, text="shape of gamma distribution (alpha):"
        )
        self.lbl_alpha.grid(row=4, column=0)
        self.ent_alpha = tk.Entry(master=self.frm_probdis, width=30)
        self.ent_alpha.grid(row=4, column=1, columnspan=2)

        self.lbl_beta = tk.Label(
            master=self.frm_probdis, text="scale of gamma distribution (beta):"
        )
        self.lbl_beta.grid(row=5, column=0)
        self.ent_beta = tk.Entry(master=self.frm_probdis, width=30)
        self.ent_beta.grid(row=5, column=1, columnspan=2)

        self.btn_open.configure(state="disabled")
        self.ent_alpha.configure(state="normal")
        self.ent_beta.configure(state="normal")
        self.ent_file.configure(state="disabled")
        self.ent_mean.configure(state="disabled")
        self.ent_std.configure(state="disabled")
        self.ent_alpha.insert(0, "2.468531215126044")
        self.ent_beta.insert(0, "0.02584178910588476")

        # produce master with extra option like number of fractions.
        self.frm_extras = tk.Frame(
            master=self.frame.interior, relief=tk.SUNKEN, borderwidth=3
        )
        self.frm_extras.pack()

        self.lbl_fractions = tk.Label(
            master=self.frm_extras, text="Total number of fractions"
        )
        self.lbl_fractions.grid(row=0, column=0)
        self.ent_fractions = tk.Entry(master=self.frm_extras, width=30)
        self.ent_fractions.grid(row=0, column=1, columnspan=2)
        self.ent_fractions.insert(0, "5")
        self.btn_infofrac = tk.Button(
            master=self.frm_extras, text="?", command=self.infofrac
        )
        self.btn_infofrac.grid(row=0, column=3)

        self.lbl_mindose = tk.Label(master=self.frm_extras, text="minimum dose")
        self.lbl_mindose.grid(row=1, column=0)
        self.ent_mindose = tk.Entry(master=self.frm_extras, width=30)
        self.ent_mindose.grid(row=1, column=1, columnspan=2)
        self.ent_mindose.insert(0, "0")
        self.btn_mindose = tk.Button(
            master=self.frm_extras, text="?", command=self.infomin
        )
        self.btn_mindose.grid(row=1, column=3)

        self.lbl_maxdose = tk.Label(master=self.frm_extras, text="maximum dose")
        self.lbl_maxdose.grid(row=2, column=0)
        self.ent_maxdose = tk.Entry(master=self.frm_extras, width=30)
        self.ent_maxdose.grid(row=2, column=1, columnspan=2)
        self.ent_maxdose.insert(0, "22.3")
        self.btn_maxdose = tk.Button(
            master=self.frm_extras, text="?", command=self.infomin
        )
        self.btn_maxdose.grid(row=2, column=3)
        # Create a new frame `frm_form` to contain the Label
        # and Entry widgets for entering variable values
        self.frm_form = tk.Frame(
            master=self.frame.interior, relief=tk.SUNKEN, borderwidth=3
        )
        # Pack the frame into the master
        self.frm_form.pack()

        self.frm_buttons = tk.Frame(master=self.frame.interior)
        self.frm_buttons.pack(fill=tk.X, ipadx=5, ipady=5)

        self.frm_output = tk.Frame(
            master=self.frame.interior, relief=tk.SUNKEN, borderwidth=3
        )
        # add label and entry for filename
        self.label = tk.Label(master=self.frm_form, text="file path of prior patients")
        self.ent_file = tk.Entry(master=self.frm_form, width=50)
        self.label.grid(row=0, column=0, sticky="e")
        self.ent_file.grid(row=0, column=1)
        self.info_funcs = [
            self.info10,
            self.info11,
            self.info12,
            self.info13,
            self.info14,
            self.info15,
        ]
        self.info_buttons = [
            "self.btn_sf",
            "self.btn_abt",
            "self.btn_abn",
            "self.btn_OARlimit",
            "self.btn_tumorlimit",
            "self.btn_BED",
        ]
        # List of field labels
        self.labels = [
            "sparing factors separated by spaces:",
            "alpha-beta ratio of tumor:",
            "alpha-beta ratio of OAR:",
            "OAR limit:",
            "prescribed tumor dose:",
            "accumulated BED in tumor or OAR (depending on the tracked organ):",
        ]
        self.ent_sf = tk.Entry(master=self.frm_form, width=50)
        self.lbl_sf = tk.Label(master=self.frm_form, text=self.labels[0])
        self.example_list = [
            "sparing factors separated by space",
            10,
            3,
            90,
            72,
            "only needed if we calculate the dose for a single fraction",
        ]
        self.lbl_sf.grid(row=0, column=0, sticky="e")
        self.ent_sf.grid(row=0, column=1)
        self.ent_sf.insert(0, f"{self.example_list[0]}")
        self.btn_sf = tk.Button(
            master=self.frm_form, text="?", command=self.info_funcs[0]
        )
        self.btn_sf.grid(row=0, column=2)

        self.ent_abt = tk.Entry(master=self.frm_form, width=50)
        self.lbl_abt = tk.Label(master=self.frm_form, text=self.labels[1])
        self.lbl_abt.grid(row=1, column=0, sticky="e")
        self.ent_abt.grid(row=1, column=1)
        self.ent_abt.insert(0, f"{self.example_list[1]}")
        self.btn_abt = tk.Button(
            master=self.frm_form, text="?", command=self.info_funcs[1]
        )
        self.btn_abt.grid(row=1, column=2)

        self.ent_abn = tk.Entry(master=self.frm_form, width=50)
        self.lbl_abn = tk.Label(master=self.frm_form, text=self.labels[2])
        self.lbl_abn.grid(row=2, column=0, sticky="e")
        self.ent_abn.grid(row=2, column=1)
        self.ent_abn.insert(0, f"{self.example_list[2]}")
        self.btn_abn = tk.Button(
            master=self.frm_form, text="?", command=self.info_funcs[2]
        )
        self.btn_abn.grid(row=2, column=2)

        self.ent_OARlimit = tk.Entry(master=self.frm_form, width=50)
        self.lbl_OARlimit = tk.Label(master=self.frm_form, text=self.labels[3])
        self.lbl_OARlimit.grid(row=3, column=0, sticky="e")
        self.ent_OARlimit.grid(row=3, column=1)
        self.ent_OARlimit.insert(0, f"{self.example_list[3]}")
        self.btn_OARlimit = tk.Button(
            master=self.frm_form, text="?", command=self.info_funcs[3]
        )
        self.btn_OARlimit.grid(row=3, column=2)

        self.ent_tumorlimit = tk.Entry(master=self.frm_form, width=50)
        self.lbl_tumorlimit = tk.Label(master=self.frm_form, text=self.labels[4])
        self.lbl_tumorlimit.grid(row=4, column=0, sticky="e")
        self.ent_tumorlimit.grid(row=4, column=1)
        self.ent_tumorlimit.insert(0, f"{self.example_list[4]}")
        self.btn_tumorlimit = tk.Button(
            master=self.frm_form, text="?", command=self.info_funcs[4]
        )
        self.btn_tumorlimit.grid(row=4, column=2)

        self.ent_BED = tk.Entry(master=self.frm_form, width=50)
        self.lbl_BED = tk.Label(master=self.frm_form, text=self.labels[5])
        self.lbl_BED.grid(row=5, column=0, sticky="e")
        self.ent_BED.grid(row=5, column=1)
        self.ent_BED.insert(0, f"{self.example_list[5]}")
        self.btn_BED = tk.Button(
            master=self.frm_form, text="?", command=self.info_funcs[5]
        )
        self.btn_BED.grid(row=5, column=2)

        self.ent_BED.configure(
            state="disabled"
        )  # the standard calculation is a whole plan calculation
        self.ent_tumorlimit.configure(state="disabled")

        self.btn_compute = tk.Button(
            master=self.frm_buttons, text="compute plan", command=self.compute_plan
        )
        self.btn_compute.pack(side=tk.BOTTOM, ipadx=10)
        self.var = tk.IntVar()
        self.chk_single_fraction = tk.Checkbutton(
            master=self.frm_buttons,
            text="Calculate dose only for actual fraction",
            variable=self.var,
            onvalue=1,
            offvalue=0,
            command=self.checkbox,
        )
        self.chk_single_fraction.pack(side=tk.BOTTOM, padx=10, ipadx=10)
        self.var_OAR = tk.IntVar()
        self.chk_OARswitch = tk.Checkbutton(
            master=self.frm_buttons,
            text="Calculate optimal plan by minimizing OAR and aiming on prescribed tumor dose",
            variable=self.var_OAR,
            onvalue=1,
            offvalue=0,
            command=self.OAR_enable,
        )
        self.chk_OARswitch.pack(side=tk.BOTTOM, padx=10, ipadx=10)

        self.lbl_info = tk.Label(
            master=self.frm_output,
            text="There are several default values set. Only the sparing factors have to been inserted.\nThis program might take some minutes to calculate",
        )
        self.lbl_info.pack()
        self.frm_output.pack(fill=tk.BOTH, ipadx=10, ipady=10)

    def select_file(self):
        filetypes = (("csv files", "*.csv"), ("All files", "*.*"))

        filename = fd.askopenfilename(
            title="Open a file", initialdir="/", filetypes=filetypes
        )

        showinfo(title="Selected File", message=self.filename)

        self.ent_file.insert(0, filename)
        self.data = np.array(pd.read_csv(self.ent_file.get(), sep=";"))
        self.stds = self.data.std(axis=1)
        self.alpha, self.loc, self.beta = gamma.fit(self.stds, floc=0)
        self.ent_alpha.configure(state="normal")
        self.ent_beta.configure(state="normal")
        self.ent_alpha.delete(0, "end")
        self.ent_beta.delete(0, "end")
        self.ent_alpha.insert(0, self.alpha)
        self.ent_beta.insert(0, self.beta)
        self.ent_alpha.configure(state="disabled")
        self.ent_beta.configure(state="disabled")

    def checkbox1(self):
        if self.var_radio.get() == 1:
            self.btn_open.configure(state="disabled")
            self.ent_alpha.configure(state="normal")
            self.ent_beta.configure(state="normal")
            self.ent_file.configure(state="disabled")
            self.ent_mean.configure(state="disabled")
            self.ent_std.configure(state="disabled")
        elif self.var_radio.get() == 2:
            self.ent_file.configure(state="normal")
            self.btn_open.configure(state="normal")
            self.ent_alpha.configure(state="disabled")
            self.ent_beta.configure(state="disabled")
            self.ent_mean.configure(state="disabled")
            self.ent_std.configure(state="disabled")
        elif self.var_radio.get() == 3:
            self.ent_mean.configure(state="normal")
            self.ent_std.configure(state="normal")
            self.ent_alpha.configure(state="disabled")
            self.ent_beta.configure(state="disabled")
            self.btn_open.configure(state="disabled")
            self.ent_file.configure(state="disabled")

    # assign infobutton commands

    def info1(self):
        self.lbl_info[
            "text"
        ] = "Insert the path of your prior patient data in here. \nThis is only needed, if the checkbox for prior data is marked. \nIf not, one can directly insert the hyperparameters below. \nThe file with the prior data must be of the shape n x k,\nwhere each new patient n is on a row and each fraction for patient n is in column k"

    def info2(self):
        self.lbl_info[
            "text"
        ] = "Insert the mean of the sparing factor distribution. \nwith this option the distribution is not updated"

    def info3(self):
        self.lbl_info[
            "text"
        ] = "Insert the standard deviation of the sparing factor distribution. \nwith this option the distribution is not updated"

    def info4(self):
        self.lbl_info[
            "text"
        ] = "Insert the shape parameter for the inverse-gamme distribution."

    def info5(self):
        self.lbl_info[
            "text"
        ] = "Insert the scale parameter for the inverse-gamme distribution."

    def infofrac(self):
        self.lbl_info[
            "text"
        ] = "Insert the number of fractions to be delivered to the patient. \n5 fractions is set a standard SBRT treatment."

    def infomin(self):
        self.lbl_info[
            "text"
        ] = "Insert the minimal physical dose that shall be delivered to the PTV95 in one fraction."

    def infomax(self):
        self.lbl_info[
            "text"
        ] = "Insert the maximal physical dose that shall be delivered to the PTV95 in one fraction."

    # assign infobutton commands

    def info10(self):
        self.lbl_info[
            "text"
        ] = "Insert the sparing factors that were observed so far.\n The sparing factor of the planning session must be included!.\nThe sparing factors must be separated by spaces e.g.:\n1.1 0.95 0.88\nFor a whole plan 6 sparing factors are needed."

    def info11(self):
        self.lbl_info["text"] = "Insert the alpha-beta ratio of the tumor tissue."

    def info12(self):
        self.lbl_info[
            "text"
        ] = "Insert the alpha-beta ratio of the dose-limiting Organ at risk."

    def info13(self):
        self.lbl_info[
            "text"
        ] = "Insert the maximum dose delivered to the dose-limiting OAR in BED."

    def info14(self):
        self.lbl_info[
            "text"
        ] = "Insert the prescribed biological effectiv dose to be delivered to the tumor."

    def info15(self):
        self.lbl_info[
            "text"
        ] = "Insert the accumulated BED so far. (If fraction one, it is zero)."

    def checkbox(self):
        if self.var.get() == 0:
            self.ent_BED.configure(state="disabled")
        else:
            self.ent_BED.configure(state="normal")

    def OAR_enable(self):
        if self.var_OAR.get() == 0:
            self.ent_OARlimit.configure(state="normal")
            self.ent_tumorlimit.configure(state="disabled")
        else:
            self.ent_OARlimit.configure(state="disabled")
            self.ent_tumorlimit.configure(state="normal")

    def compute_plan(self):
        number_of_fractions = int(self.ent_fractions.get())
        alpha = float(self.ent_alpha.get())
        beta = float(self.ent_beta.get())
        min_dose = float(self.ent_mindose.get())
        max_dose = float(self.ent_maxdose.get())
        try:
            global lbl_output
            self.lbl_output.destroy()
        except:
            pass
        if self.var_radio.get() != 3:
            fixed_prob = 0
            fixed_mean = 0
            fixed_std = 0
        if self.var_radio.get() == 3:
            fixed_prob = 1
            fixed_mean = float(self.ent_mean.get())
            fixed_std = float(self.ent_std.get())
        if self.var.get() == 0:
            try:
                alpha = float(self.ent_alpha.get())
                beta = float(self.ent_beta.get())
                sparing_factors_str = (self.ent_sf.get()).split()
                sparing_factors = [float(i) for i in sparing_factors_str]
                abt = float(self.ent_abt.get())
                abn = float(self.ent_abn.get())
                OAR_limit = float(self.ent_OARlimit.get())
                if self.var_OAR.get() == 0:
                    [tumor_doses, OAR_doses, physical_doses] = inttumor.whole_plan(
                        number_of_fractions,
                        sparing_factors,
                        abt,
                        abn,
                        alpha,
                        beta,
                        OAR_limit,
                        min_dose,
                        max_dose,
                        fixed_prob,
                        fixed_mean,
                        fixed_std,
                    )

                    self.lbl_output = tk.Frame(master=self.frame.interior)
                    self.lbl_output.pack()
                    frame = tk.Frame(
                        master=self.lbl_output, relief=tk.RAISED, borderwidth=1
                    )
                    frame.grid(row=0, column=0)
                    label = tk.Label(master=frame, text="fraction number")
                    label.pack()
                    frame = tk.Frame(
                        master=self.lbl_output, relief=tk.RAISED, borderwidth=1
                    )
                    frame.grid(row=0, column=1)
                    label = tk.Label(master=frame, text="sparing factor")
                    label.pack()
                    frame = tk.Frame(
                        master=self.lbl_output, relief=tk.RAISED, borderwidth=1
                    )
                    frame.grid(row=0, column=2)
                    label = tk.Label(
                        master=frame, text="physical dose delivered to PTV95"
                    )
                    label.pack()
                    frame = tk.Frame(
                        master=self.lbl_output, relief=tk.RAISED, borderwidth=1
                    )
                    frame.grid(row=0, column=3)
                    label = tk.Label(master=frame, text="BED delivered to tumor")
                    label.pack()
                    frame = tk.Frame(
                        master=self.lbl_output, relief=tk.RAISED, borderwidth=1
                    )
                    frame.grid(row=0, column=4)
                    label = tk.Label(master=frame, text="BED delivered to OAR")
                    label.pack()
                    for i in range(1, number_of_fractions + 1):
                        for j in range(5):
                            if j == 0:
                                frame = tk.Frame(master=self.lbl_output)
                                frame.grid(row=i, column=0)
                                label = tk.Label(master=frame, text=f"fraction {i}")
                                label.pack()
                            elif j == 1:
                                frame = tk.Frame(master=self.lbl_output)
                                frame.grid(row=i, column=1)
                                label = tk.Label(
                                    master=frame, text=f" {sparing_factors[i]}"
                                )
                                label.pack()
                            elif j == 2:
                                frame = tk.Frame(master=self.lbl_output)
                                frame.grid(row=i, column=2)
                                label = tk.Label(
                                    master=frame,
                                    text=f" {np.round(physical_doses[i-1],2)}",
                                )
                                label.pack()
                            elif j == 3:
                                frame = tk.Frame(master=self.lbl_output)
                                frame.grid(row=i, column=3)
                                label = tk.Label(
                                    master=frame,
                                    text=f" {np.round(tumor_doses[i-1],2)}",
                                )
                                label.pack()
                            elif j == 4:
                                frame = tk.Frame(master=self.lbl_output)
                                frame.grid(row=i, column=4)
                                label = tk.Label(
                                    master=frame, text=f" {np.round(OAR_doses[i-1],2)}"
                                )
                                label.pack()
                    frame = tk.Frame(
                        master=self.lbl_output, relief=tk.RAISED, borderwidth=1
                    )
                    frame.grid(row=number_of_fractions + 1, column=0)
                    label = tk.Label(master=frame, text="accumulated doses")
                    label.pack()
                    frame = tk.Frame(
                        master=self.lbl_output, relief=tk.RAISED, borderwidth=1
                    )
                    frame.grid(row=number_of_fractions + 1, column=3)
                    label = tk.Label(
                        master=frame, text=f" {np.round(np.sum(tumor_doses),2)}"
                    )
                    label.pack()
                    frame = tk.Frame(
                        master=self.lbl_output, relief=tk.RAISED, borderwidth=1
                    )
                    frame.grid(row=number_of_fractions + 1, column=4)
                    label = tk.Label(master=frame, text=f" {np.sum(OAR_doses)}")
                    label.pack()
                elif self.var_OAR.get() == 1:
                    prescribed_tumor_dose = float(self.ent_tumorlimit.get())
                    [tumor_doses, OAR_doses, physical_doses] = intOAR.whole_plan(
                        number_of_fractions,
                        sparing_factors,
                        alpha,
                        beta,
                        prescribed_tumor_dose,
                        abt,
                        abn,
                        min_dose,
                        max_dose,
                        fixed_prob,
                        fixed_mean,
                        fixed_std,
                    )
                    self.lbl_output = tk.Frame(master=self.frame.interior)
                    self.lbl_output.pack()
                    frame = tk.Frame(
                        master=self.lbl_output, relief=tk.RAISED, borderwidth=1
                    )
                    frame.grid(row=0, column=0)
                    label = tk.Label(master=frame, text="fraction number")
                    label.pack()
                    frame = tk.Frame(
                        master=self.lbl_output, relief=tk.RAISED, borderwidth=1
                    )
                    frame.grid(row=0, column=1)
                    label = tk.Label(master=frame, text="sparing factor")
                    label.pack()
                    frame = tk.Frame(
                        master=self.lbl_output, relief=tk.RAISED, borderwidth=1
                    )
                    frame.grid(row=0, column=2)
                    label = tk.Label(
                        master=frame, text="physical dose delivered to PTV95"
                    )
                    label.pack()
                    frame = tk.Frame(
                        master=self.lbl_output, relief=tk.RAISED, borderwidth=1
                    )
                    frame.grid(row=0, column=3)
                    label = tk.Label(master=frame, text="BED delivered to tumor")
                    label.pack()
                    frame = tk.Frame(
                        master=self.lbl_output, relief=tk.RAISED, borderwidth=1
                    )
                    frame.grid(row=0, column=4)
                    label = tk.Label(master=frame, text="BED delivered to OAR")
                    label.pack()
                    for i in range(1, number_of_fractions + 1):
                        for j in range(5):
                            if j == 0:
                                frame = tk.Frame(master=self.lbl_output)
                                frame.grid(row=i, column=0)
                                label = tk.Label(master=frame, text=f"fraction {i}")
                                label.pack()
                            elif j == 1:
                                frame = tk.Frame(master=self.lbl_output)
                                frame.grid(row=i, column=1)
                                label = tk.Label(
                                    master=frame, text=f" {sparing_factors[i]}"
                                )
                                label.pack()
                            elif j == 2:
                                frame = tk.Frame(master=self.lbl_output)
                                frame.grid(row=i, column=2)
                                label = tk.Label(
                                    master=frame,
                                    text=f" {np.round(physical_doses[i-1],2)}",
                                )
                                label.pack()
                            elif j == 3:
                                frame = tk.Frame(master=self.lbl_output)
                                frame.grid(row=i, column=3)
                                label = tk.Label(
                                    master=frame,
                                    text=f" {np.round(tumor_doses[i-1],2)}",
                                )
                                label.pack()
                            elif j == 4:
                                frame = tk.Frame(master=self.lbl_output)
                                frame.grid(row=i, column=4)
                                label = tk.Label(
                                    master=frame, text=f" {np.round(OAR_doses[i-1],2)}"
                                )
                                label.pack()
                                frame = tk.Frame(
                                    master=self.lbl_output,
                                    relief=tk.RAISED,
                                    borderwidth=1,
                                )
                    frame.grid(row=number_of_fractions + 1, column=0)
                    label = tk.Label(master=frame, text="accumulated doses")
                    label.pack()
                    frame = tk.Frame(
                        master=self.lbl_output, relief=tk.RAISED, borderwidth=1
                    )
                    frame.grid(row=number_of_fractions + 1, column=3)
                    label = tk.Label(
                        master=frame, text=f" {np.round(np.sum(tumor_doses),2)}"
                    )
                    label.pack()
                    frame = tk.Frame(
                        master=self.lbl_output, relief=tk.RAISED, borderwidth=1
                    )
                    frame.grid(row=number_of_fractions + 1, column=4)
                    label = tk.Label(
                        master=frame, text=f" {np.round(np.sum(OAR_doses),1)}"
                    )
                    label.pack()
            except ValueError:
                self.lbl_info[
                    "text"
                ] = "please enter correct values\nsparing factors have to been inserted with space in between. No additional brackets needed."
        else:
            try:
                sparing_factors_str = (self.ent_sf.get()).split()
                sparing_factors = [float(i) for i in sparing_factors_str]
                abt = float(self.ent_abt.get())
                abn = float(self.ent_abn.get())
                OAR_limit = float(self.ent_OARlimit.get())
                BED = float(self.ent_BED.get())
                if self.var_OAR.get() == 0:
                    [
                        Values,
                        policy,
                        actual_value,
                        actual_policy,
                        dose_delivered_OAR,
                        dose_delivered_tumor,
                        total_dose_delivered_OAR,
                        actual_dose_delivered,
                    ] = inttumor.value_eval(
                        len(sparing_factors) - 1,
                        number_of_fractions,
                        BED,
                        sparing_factors,
                        alpha,
                        beta,
                        abt,
                        abn,
                        OAR_limit,
                        min_dose,
                        max_dose,
                        fixed_prob,
                        fixed_mean,
                        fixed_std,
                    )
                    self.lbl_info[
                        "text"
                    ] = f"Optimal dose for fraction {len(sparing_factors)-1} = {actual_dose_delivered}\ndelivered tumor BED = {dose_delivered_tumor}\ndelivered OAR BED = {dose_delivered_OAR}"
                elif self.var_OAR.get() == 1:
                    prescribed_tumor_dose = float(self.ent_tumorlimit.get())
                    [
                        policy,
                        sf,
                        physical_dose,
                        tumor_dose,
                        OAR_dose,
                    ] = intOAR.value_eval(
                        len(sparing_factors) - 1,
                        number_of_fractions,
                        BED,
                        sparing_factors,
                        alpha,
                        beta,
                        prescribed_tumor_dose,
                        abt,
                        abn,
                        min_dose,
                        max_dose,
                        fixed_prob,
                        fixed_mean,
                        fixed_std,
                    )
                    self.lbl_info[
                        "text"
                    ] = f"The optimal dose for fraction {len(sparing_factors)-1},  = {physical_dose}\n dose delivered to tumor = {tumor_dose}\n dose delivered OAR = {OAR_dose}"
            except ValueError:
                self.lbl_info[
                    "text"
                ] = "please enter correct values\nsparing factors have to been inserted with space in between. No additional brackets needed."


if __name__ == "__main__":
    root = tk.Tk()
    GUI = GUI2Dextended(root)
    # Start the application
    root.mainloop()
