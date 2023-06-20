# -*- coding: utf-8 -*-
import argparse
import json
import adaptfx as afx
import adaptsim as afs
import numpy as np
import sys
nme = __name__

class MC_object():
    """
    Reinforcement Learning class to check instructions
    of calculation, invoke keys and define
    calculation settings from file
    """
    def __init__(self, model_filename):
        model = afx.RL_object(model_filename)
        with open(model.filename, 'r') as f:
            read_in = f.read()
        raw_simulation_dict = json.loads(read_in)

        try:
            algorithm_simulation = raw_simulation_dict['algorithm_simulation']
        except KeyError as algo_err:
            afx.aft_error(f'{algo_err} key missing in: "{model.filename}"', nme)
        else:
            afx.aft_message_info('algorithm simulation:', algorithm_simulation, nme, 1)

        try: # check if simulation_keys exists and is a dictionnary
            raw_keys = raw_simulation_dict['keys_simulation']
        except KeyError:
            afx.aft_error(f'"keys_simulation" is missing in : "{model.filename}"', nme)
        else:
            simulation_dict = afx.key_reader(afs.KEY_DICT_SIM, afs.ALL_SIM_DICT, raw_keys, 'sim')
            afx.aft_message_dict('simulation', simulation_dict, nme)

        self.algorithm = model.algorithm
        self.filename = model.filename
        self.basename = model.basename
        self.log = model.log
        self.log_level = model.log_level
        self.keys_model = model.keys
        self.settings = model.settings
        self.algorithm_simulation = algorithm_simulation
        self.keys_simulation = afx.DotDict(simulation_dict)


    def simulate(self):
        plot_sets = afs.RCPARAMS
        if self.settings.usetex:
            # if CLI specifies global use of latex
            plot_sets["text.usetex"] = True
        elif self.keys_simulation.usetex == True:
            plot_sets["text.usetex"] = True
        plot_sets["figure.figsize"] = self.keys_simulation.figsize
        plot_sets["font.size"] = self.keys_simulation.fontsize
        n_frac = self.keys_model.number_of_fractions
        # case listing for all options
        if self.algorithm_simulation == 'histogram':
            n_patients = self.keys_simulation.n_patients 
            mu = self.keys_simulation.fixed_mean_sample
            std = self.keys_simulation.fixed_std_sample
            plans = np.zeros(n_patients)
            for i in range(n_patients):
                self.keys_model.sparing_factors = list(np.random.normal(mu , std, n_frac + 1))
                output = afx.multiple(self.algorithm, self.keys_model, self.settings)
                # output.oar_sum output.tumor_sum
                n_frac_used = np.count_nonzero(~np.isnan(output.physical_doses))
                plans[i] = n_frac_used
            end_plot = afs.plot_hist(plans, n_frac, plot_sets)

        if self.algorithm_simulation == 'NEW':
            # Set parameters
            u = 0  # mean of normal distribution
            sigma = 1  # standard deviation of normal distribution
            shape = 2  # shape parameter of gamma distribution
            scale = 3  # scale parameter of gamma distribution
            n = 6  # number of samples in each array
            m = 10  # number of arrays to generate

            # Vectorized implementation
            u_mean = np.random.normal(u, sigma, size=(m, 1))
            std = np.random.gamma(shape, scale, size=(m, 1))
            samples = np.random.normal(u_mean, std, size=(m, n))

            for i in range(m):
                self.keys_model.sparing_factors = samples[i]
                output = afx.multiple(self.algorithm, self.keys_model, self.settings)
            
        elif self.algorithm_simulation == 'fraction':
            # plot applied dose, sparing factor dependent on fraction
            c_list = self.keys_simulation.c_list
            n_c = len(c_list)
            mu = self.keys_model.fixed_mean
            std = self.keys_model.fixed_std
            sf_list = self.keys_model.sparing_factors
            c_dose_array = np.zeros((n_c, n_frac))
            oar_sum_array = np.zeros((n_c))
            for i, c in enumerate(self.keys_simulation.c_list):
                self.keys_model.c = c
                output = afx.multiple(self.algorithm, self.keys_model, self.settings)
                c_dose_array[i] = output.tumor_doses
                oar_sum_array[i] = output.oar_sum
            self.c_dose_list = c_dose_array
            end_plot = afs.plot_dose(self.c_dose_list, sf_list, n_frac, c_list, oar_sum_array,
                mu, std, plot_sets)

        elif self.algorithm_simulation == 'single_state':
            # plot policy, values or remaining number of fraction for one state
            out = afx.multiple(self.algorithm, self.keys_model, self.settings)
            if self.settings.plot_policy:
                end_plot = afs.plot_val_single(out.policy.sf, out.policy.states, out.policy.val,
                out.policy.fractions, self.keys_simulation.plot_index, r'Policy $\pi$ in BED$_{10}$', 'turbo', plot_sets)
            if self.settings.plot_values:
                end_plot = afs.plot_val_single(out.value.sf, out.value.states, out.value.val,
                out.value.fractions, self.keys_simulation.plot_index, r'Value $v$', 'viridis', plot_sets)
            if self.settings.plot_remains:
                end_plot = afs.plot_val_single(out.remains.sf, out.remains.states, out.remains.val,
                out.remains.fractions, self.keys_simulation.plot_index, r'Expected Remaining Number $\varepsilon$', 'plasma', plot_sets)

        elif self.algorithm_simulation == 'all_state':
            # plot all policy, values or remaining number of fraction except for the last
            out = afx.multiple(self.algorithm, self.keys_model, self.settings)
            if self.settings.plot_policy:
                end_plot = afs.plot_val_all(out.policy.sf, out.policy.states, out.policy.val,
                out.policy.fractions, r'Policy $\pi$ in BED$_{10}$ [Gy]', 'turbo', plot_sets)
            if self.settings.plot_values:
                end_plot = afs.plot_val_all(out.value.sf, out.value.states, out.value.val,
                out.value.fractions, r'Value $v$', 'viridis', plot_sets)
            if self.settings.plot_remains:
                end_plot = afs.plot_val_all(out.remains.sf, out.remains.states, out.remains.val,
                out.remains.fractions, r'Expected Remaining Number $\varepsilon$', 'plasma', plot_sets)

        elif self.algorithm_simulation == 'single_distance':
            selec = self.keys_simulation.data_selection
            row_hue = self.keys_simulation.data_row_hue
            plot_sets["axes.linewidth"] = 1.3
            data_sf = afs.data_reader(self.keys_simulation.data_filepath, selec[0], selec[1])
            end_plot = afs.plot_single(data_sf, 'Distance', 'sparing_factor',
                row_hue, r'$w$ [cm]', r'$\delta$', True, 'Set2', plot_sets)

        elif self.algorithm_simulation == 'single_patient':
            selec = self.keys_simulation.data_selection
            row_hue = self.keys_simulation.data_row_hue
            plot_sets["axes.linewidth"] = 1.3
            data_sf = afs.data_reader(self.keys_simulation.data_filepath, selec[0], selec[1], selec[2], selec[3])
            end_plot = afs.plot_single(data_sf, 'Patient', 'sparing_factor',
                row_hue, 'Patient', r'$\delta$', False, 'Set2', plot_sets)

        elif self.algorithm_simulation == 'grid_distance':
            selec = self.keys_simulation.data_selection
            row_hue = self.keys_simulation.data_row_hue
            plot_sets["axes.linewidth"] = 1.3
            data_sf = afs.data_reader(self.keys_simulation.data_filepath, selec[0], selec[1], selec[2], selec[3])
            end_plot = afs.plot_grid(data_sf, 'Distance', 'sparing_factor',
                'Patient', row_hue, r'$w$ [cm]', r'$\delta$', 'colorblind', plot_sets)

        elif self.algorithm_simulation == 'grid_fraction':
            selec = self.keys_simulation.data_selection
            row_hue = self.keys_simulation.data_row_hue
            plot_sets["axes.linewidth"] = 1.2
            data_sf = afs.data_reader(self.keys_simulation.data_filepath, selec[0], selec[1], selec[2], selec[3])
            end_plot = afs.plot_twin_grid(data_sf, 'Fraction', 'sparing_factor',
                'Distance', 'Patient', row_hue, r'Fraction $t$', r'$\delta$', r'$w$ [cm]', 'colorblind', plot_sets)
        elif end_plot == None:
            # catch case where no algorithm simulation is specified
            afx.aft_error(f'No such simulation: "{self.algorithm_simulation}"', nme)

        if self.keys_simulation.save:
            afx.save_plot(self.basename, end_plot)
        else:
            afx.show_plot()

def main():
    """
    CLI interface to invoke the RL class
    """
    start = afx.timing()
    parser = argparse.ArgumentParser(
        description='Patient Monte Carlo Simulation to test adaptive fractionation'
    )
    parser.add_argument(
        '-f',
        '--filenames',
        metavar='',
        help='input adaptive fractionation instruction filename(s)',
        type=str,
        nargs='*'
    )
    parser.add_argument(
        '-t',
        '--usetex',
        help='matplotlib usetex parameter flag',
        action='store_true',
        default=False
    )
    # In case there is no input show help
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    for single_file in args.filenames:
        sim = MC_object(single_file)
        sim.settings.usetex = args.usetex
        afx.aft_message(f'start session for {single_file}', nme, 1)
        sim.simulate()
        afx.timing(start)
        afx.aft_message(f'finish session for {single_file}', nme, 1)


if __name__ == '__main__':
    main()
