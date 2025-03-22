# Copyright (c) 2022 - 2025 Jan Tünnermann. All rights reserved.
# This work is licensed under the terms of the MIT license.  
# For a copy, see <https://opensource.org/licenses/MIT>.

import pymc as pm
import pytensor.tensor as at
import os
import numpy as np
from matplotlib.pylab import plt
import arviz as az
from tqdm import tqdm
from .data import aggregate, get_end, get_type_names, get_transistion_length, objective_frequency
import pandas as pd
from IPython.core.display import display, HTML

import pytensor.tensor as at

import pytensor.tensor as at

def adaptation_curve(
    cycle_index,
    phases,
    adaptation=1,
    shift=0,
    bias=0,
    upper_limit=1,
    lower_limit=1,
):
    """Computes an adaptation curve with that defines transitions between phases.

    Args:
        cycle_index (TensorVariable): 
            A series of integers representing the cycle index, starting at zero.
        phases (dict): 
            A dictionary defining plateau and transition phases.
        adaptation (float, optional): 
            Adaptation parameter τ (default is 1).
        shift (float, optional): 
            Shift parameter δ, offsetting `cycle_index` (default is 0).
        bias (float, optional): 
            Bias parameter β (default is 0).
        upper_limit (float, optional): 
            Controls the max positive value, e.g. if the positive plateau
            is at only 0.75 compared to the negative, set upper_limit to 0.75

    Returns:
        TensorVariable: The computed adaptation curve function.
    """
    # Define scale factor
    sf = 6  # Hard coded scaling factor so that values are in a nicer range
    sf_pos = sf * ((upper_limit * 2) - 1)
    sf_neg = -sf * ((lower_limit * 2) - 1)

    # Adjust time index
    t = cycle_index - shift  
    func = at.where(t < 0, sf_neg, 0)  # Initialize function, -sf before first phase

    # Get phase start times and type names
    phase_starts = sorted(phases.keys())
    tn = get_type_names(phases)  

    # Compute plateau and transition phases
    for i, ps in enumerate(phase_starts[:-1]):
        next_phase = phase_starts[i + 1]
        phase_label = phases[ps]
        active = at.ge(t, ps) * at.lt(t, next_phase)  # Boolean mask for phase range

        if tn[0] + ' Plateau' in phase_label:
            func += active * sf_neg  

        elif tn[1] + ' Plateau' in phase_label:
            func += active * sf_pos  

        elif tn[0] + ' to ' + tn[1] + ' Transition' in phase_label:
            tl_inv = 1 / (next_phase - ps)  # Precompute inverse length for efficiency
            func += active * ((t - ps) * tl_inv * (sf_pos - sf_neg) + sf_neg)

        elif tn[1] + ' to ' + tn[0] + ' Transition' in phase_label:
            tl_inv = 1 / (next_phase - ps)  
            func += active * ((-(t - ps)) * tl_inv * (sf_pos - sf_neg) + sf_pos)

    # Handle the final phase
    last_phase_label = phases[phase_starts[-2]]
    func += at.ge(t, phase_starts[-1]) * (sf_neg if tn[0] + ' Plateau' in last_phase_label or 'to ' + tn[0] in last_phase_label else sf_pos)

    return pm.invlogit(func * adaptation + bias)




def get_model(phases,
              data,
              upper_limit=1,
              lower_limit=1,
              custom_priors={},
              custom_links=None,
              silent=False,
):
    """Sets up a PyMC model from an adaptation curve defined by phases.

    Constructs a hierarchical Bayesian model in PyMC based on the provided 
    adaptation curve phases and dataset. Allows customization of prior 
    distributions and link functions.

    Args:
        phases (list): 
            A list of dictionaries defining the adaptation curve phases 
            (see TODO for details).
        data (pd.DataFrame): 
            A Pandas DataFrame containing the required columns (see TODO for details).
        custom_priors (dict, optional): 
            A dictionary specifying PyMC distribution definitions as strings, 
            which are evaluated to set up priors. Defaults to an empty dictionary.
        custom_links (list, optional): 
            Defines link functions between group- and participant-level parameters. 
            Defaults to `None`, leading to identity links for all parameters except 
            adaptation, which is linked via an exponential function.
        silent (bool, optional): 
            If True, suppresses status outputs. Defaults to `False`.

    Returns:
        pm.Model: 
            A fully specified PyMC model ready for sampling.
    """
    links = {'adaptation': pm.math.exp,
             'shift': lambda x: x, 'bias': lambda x: x}
    if custom_links is not None:
        for link in custom_links:
            links[link] = custom_links[link]

    data = aggregate(data)

    participant_idx, participants = data['Participant_ID'].factorize(sort=True)
    condition_idx, conditions = data['Condition_Name'].factorize()
    coords = {
        'Participant': participants,
        'Condition': conditions
    }

    with pm.Model(coords=coords) as model:

        selections = data.Decline_First_Selection.values.astype(int)
        repetitions = data.Repetitions.values.astype(int)
        cycle_idx = data.Cycle_Idx.values.astype(int)
        starting_plateau_idx = data.Starting_Plateau_Idx.values.astype(int)

        if not 'Starting_Plateau_Idx' in data.columns:
            starting_plateau_idx = [0] * len(data)

        shift_mu_sigma = np.mean(get_transistion_length(phases))/8
        shift_sigma_dist_b = 1/(np.mean(get_transistion_length(phases))/16)
        if not silent:
            if 'shift_mu' not in custom_priors:
                print(f"Auto setting shift_mu to Normal(0, %f) where "  % shift_mu_sigma
                      + "SD is 1/8 of the transition length")
            if 'shift_sigma' not in custom_priors:
                print(("Auto setting  shift_sigma to Gamma(α=3, β=%.2f),"
                       + "so that its mean is %.2f") % (
                    shift_sigma_dist_b, 3/shift_sigma_dist_b))

        cy_idx = pm.MutableData("cycle_idx", cycle_idx)
        co_idx = pm.MutableData("condition_idx", condition_idx)
        p_idx = pm.MutableData("participant_idx", participant_idx)
        sp_idx = pm.MutableData("starting_plateau_idx", starting_plateau_idx)

        if len(conditions) > 1:

            adaptation_mu_log = pm.Normal(
                'adaptation_µ_log', mu=-1, sigma=1, dims=('Condition')),
            adaptation_sigma_dist = pm.Gamma.dist(
                mu=0.5, sigma=1.5, shape=len(conditions))
            adaptation_chol, _, adaptation_sigma = pm.LKJCholeskyCov(
                'adaptation_chol_cov', n=len(conditions), eta=1,
                sd_dist=adaptation_sigma_dist, compute_corr=True)
            adaptation_effects_z = pm.Normal(
                'adaptation_effects_z', mu=0, sigma=1,
                dims=('Participant', 'Condition'))
            adaptation_effects = pm.Deterministic(
                'adaptation_effects', at.dot(adaptation_chol, adaptation_effects_z.T).T)
            adaptation = pm.Deterministic(
                'adaptation', links['adaptation'](adaptation_mu_log + adaptation_effects))

            shift_mu = pm.Normal(
                'shift_µ', 0, shift_mu_sigma,
                dims=('Condition'))
            shift_sigma_dist = pm.Exponential.dist(
                shift_sigma_dist_b, shape=len(conditions))
            shift_chol, _, _ = pm.LKJCholeskyCov(
                'shift_chol_cov', n=len(conditions), eta=1,
                sd_dist=shift_sigma_dist, compute_corr=True)
            shift_effects_z = pm.Normal(
                'shift_effect_z', mu=0, sigma=1,
                dims=('Participant', 'Condition'))
            shift_effects = pm.Deterministic(
                'shift_effects', at.dot(shift_chol, shift_effects_z.T).T)
            shift = pm.Deterministic(
                'shift',  shift_mu + shift_effects)

            bias_mu = pm.Normal('bias_µ', 0, 1)
            bias_sigma = pm.HalfCauchy('bias_σ', 0.1)

        else:  # We have only one condition
            default_priors = {
                'adaptation_mu_log': "pm.Normal('adaptation_µ_log', mu=-1, sigma=1, dims=['Condition'])",
                'adaptation_sigma_log': "pm.Gamma('adaptation_σ_log', mu=1.5, sigma=0.5)",
                'shift_mu': "pm.Normal('shift_µ', 0, shift_mu_sigma, dims=['Condition'])",
                'shift_sigma':  "pm.Gamma('shift_σ', alpha=3, beta=shift_sigma_dist_b, dims=['Condition'])",
                'bias_mu': "pm.Normal('bias_µ', 0, 1)",
                'bias_sigma': "pm.HalfCauchy('bias_σ', 0.1)"
            }
            up = {}
            for variable in default_priors:
                if variable in custom_priors:
                    up[variable] = eval(custom_priors[variable])
                else:
                    up[variable] = eval(default_priors[variable])

            adaptation_z = pm.Normal(
                'adaptation_z', 0, 1,dims=('Participant', 'Condition'))
            adaptation = pm.Deterministic(
                'adaptation',  links['adaptation'](up['adaptation_mu_log']
                                + adaptation_z * up['adaptation_sigma_log']))

            shift_z = pm.Normal(
                'shift_z', 0, 1, dims=('Participant', 'Condition'))
            shift = pm.Deterministic(
                'shift',  up['shift_mu'] + shift_z * up['shift_sigma'])

        bias_z = pm.Normal('bias_z', 0, 1, dims=('Participant'))
        bias = pm.Deterministic(
            'bias',  up['bias_mu'] + bias_z * up['bias_sigma'])

        # Map bias to the starting plateaus
        bias_ = at.stack((bias, -bias), axis=0)

        # Back-transform adaptation
        #Back-transform adaptation
        if links['adaptation'] == pm.math.exp:
            adaptation_mu = pm.Deterministic(
                'adaptation_µ', pm.math.exp(up['adaptation_mu_log']
                                            + up['adaptation_sigma_log']**2/2))

            adaptation_sigma = pm.Deterministic(
                'adaptation_σ', pm.math.exp(up['adaptation_mu_log']
                                            + 0.5 * up['adaptation_sigma_log']**2)
                * pm.math.sqrt(pm.math.exp(up['adaptation_sigma_log']**2)-1))

        p_selected = pm.Deterministic(
            'p_selected',  adaptation_curve(cy_idx, phases,
                                            adaptation[p_idx, co_idx],
                                            shift[p_idx, co_idx],
                                            bias_[sp_idx, p_idx],
                                            upper_limit=[upper_limt, lower_limit][sp_idx],
                                            lower_limit=[upper_limt, lower_limit][-sp_idx]
                                           )
        )


        y = pm.Binomial('y', p=p_selected, n=repetitions, observed=selections)

        return model


def get_samples(model,
                samples=2000,
                tune=1000,
                target_accept=0.75,
                file=None,
                seed=0,
                thin=None,
                silent=False
):
    """Generates posterior samples from a PyMC model.

    Draws samples using the NUTS sampler and stores the prior predictive 
    and posterior predictive distributions in the trace. If an existing 
    trace file is provided, samples are loaded from disk.

    Args:
        model (Model): 
            A PyMC model created using `get_model(...)`.
        samples (int, optional): 
            Number of samples to be drawn per chain after tuning (default is 2000).
        tune (int, optional): 
            Number of samples used for tuning the NUTS sampler (default is 1000).
        target_accept (float, optional): 
            Target acceptance rate for the NUTS sampler, in the range [0,1] (default is 0.75).
        seed (int, optional): 
            Seed for the random number generator (default is 0).
        thin (int, optional): 
            Keep only every nth sample to thin the trace (default is None).
        silent (bool, optional): 
            If True, suppresses status outputs (default is False).

    Returns:
        Trace: 
            The sampled posterior trace containing posterior, prior predictive, 
            and posterior predictive distributions.
    """
    with model:
        if file is None or not os.path.isfile(str(file)):
            print('Sampling posterior')
            trace = pm.sample(samples, tune=tune,
                              target_accept=target_accept, random_seed=seed)

            if thin is not None:
                trace = trace.sel(draw=slice(0, None, thin))

            print('Sampling prior predictive')
            trace.extend(pm.sample_prior_predictive(random_seed=seed))
            print('Sampling posterior predictive')
            trace.extend(pm.sample_posterior_predictive(
                trace, random_seed=seed))
            if file is not None:
                print('Saving trace to file')
                trace.to_netcdf(file)
        else:
            if not silent:
                print('Loading samples from disk! Delete/rename the existing file ' +
                      'or change "file" argument in get_samples ' +
                      'if you want to fit anew instead of loading.')
            trace = az.from_netcdf(file)
    return trace


def interactive(phases, static=False, limit=1):
    """Creates an interactive adaptation curve widget for Jupyter notebooks.

    Generates an adaptation curve visualization with interactive sliders 
    for adjusting parameters. If `static` is set to True, a non-interactive 
    version is created for exporting (e.g., saving a notebook as a PDF).

    Args:
        phases (dict): 
            Dictionary defining the phases (see TODO for specification).
        static (bool, optional): 
            If True, generates a static version without sliders (default is False).

    Returns:
        Widget or Plot: 
            An interactive widget if `static=False`, otherwise a static plot.
    """
    from ipywidgets import interact, widgets
    from IPython.display import clear_output
    end = get_end(phases)
    t = np.linspace(0, end-1, end)

    predicted_sp0 = None
    predicted_sp1 = None

    def update(adaptation, shift, bias):
        global predicted
        clear_output(wait=True)
        fig = plt.figure(figsize=(14.5, 6))
        ax = fig.add_subplot(1, 1, 1)
        predicted_sp0 = adaptation_curve(t, phases, adaptation, shift, bias, upper_limit=limit)
        predicted_sp1 = 1 - adaptation_curve(t, phases, adaptation, shift, bias, lower_limit=limit)
        obj_frequency, labels = objective_frequency(t, phases, limit)
        line_sp0, = plt.plot(t, predicted_sp0.eval(), color='blue', linewidth=3)
        line_sp1, = plt.plot(t, predicted_sp1.eval(), color='green', linewidth=3)
        plt.plot(t, obj_frequency, color='black', linewidth=3)
        plt.axhline(0, color='black')
        plt.ylim(0, 1)
        for tx in t:
            plt.axvline(x=tx, linestyle='-', linewidth=0.5, color='gray')
        plt.xticks(t, rotation=90)
        ax.set_xticklabels(labels)
    tl = np.mean(get_transistion_length(phases))

    if static == False:
        interact(update, adaptation=widgets.FloatSlider(min=0, max=10, value=1, step=0.001,
                                                        description="Adaptation"),
                 shift=widgets.FloatSlider(
                     min=-tl, max=tl, value=0, step=0.1, description="Shift"),
                 bias=widgets.FloatSlider(
                     min=-5.0, max=5.0, value=0, step=0.001, description="Bias")
                 )
    else:
        update(1, 0, 0)
        print('[Here would be sliders for adaptation, shift, and '
              + 'bias in the interactive notebook]')


def get_table(trace, var_names=['~selected']):
    """Generates a summary table from an ArviZ inference data object.

    Extracts key statistics from the posterior samples in the trace. 
    The `static` argument allows inclusion or exclusion of specific 
    parameters.

    Args:
        trace (InferenceData): 
            An ArviZ inference data object containing posterior samples.
        static (list, optional): 
            List of parameter names to include/exclude. Defaults to 
            `['~selected']`, which excludes the latent selection probabilities 
            at each cycle index `t`.

    Returns:
        DataFrame: 
            A summary table with key statistics from the posterior samples.
    """
    summary = az.summary(trace, var_names=var_names, filter_vars='like')
    return summary


def get_diagnostic(trace, var_names=['~selected']):
    """Generates a diagnostics table from an ArviZ inference data object.

    Computes diagnostic statistics for the posterior samples, such as 
    convergence metrics. The `static` argument allows inclusion or exclusion 
    of specific parameters to manage trace size.

    Args:
        trace (InferenceData): 
            An ArviZ inference data object containing posterior samples.
        static (list, optional): 
            List of parameter names to include/exclude. Defaults to 
            `['~selected']`, which excludes the latent selection probabilities 
            at each cycle index to reduce trace size.

    Returns:
        DataFrame: 
            A diagnostics table with convergence statistics for the posterior samples.
    """
    table = get_table(trace, var_names)
    diag = pd.DataFrame(table[['ess_bulk', 'ess_tail', 'r_hat']].describe())
    return diag
