# Copyright (c) 2022 Jan Tünnermann. All rights reserved.
# This work is licensed under the terms of the MIT license.  
# For a copy, see <https://opensource.org/licenses/MIT>.

import aesara as at
import pymc as pm
import aesara.tensor as at
import os
import numpy as np
from matplotlib.pylab import plt
import arviz as az
from tqdm import tqdm
from .data import aggregate, get_end, get_type_names, get_transistion_length, objective_frequency
import pandas as pd
from IPython.core.display import display, HTML

def adaptation_curve(
    cycle_index,
    phases,
    adaptation=1,
    shift=0,
    bias=0
):
    R"""
    Implementation of the adaptation curve.

    Parameters
    ----------
    cycle_index: series of integers
        The cycle_index starting at zero.  
    phases: dictionary
        Plateau and transition phases is the format specifed here: TODO.
    adaptation: float
        Parameter adaptation τ; defaults to 1
    shift: float
        Parameter shift δ; defaults to 0
    bias: float
        Parameter bias β; defaults to 0
    """
    tn = get_type_names(phases)
    sf = 6  # Scale factor
    t = cycle_index - shift

    func = at.lt(t, 0) * (sf * -1)

    phase_starts = sorted(phases.keys())

    for i, ps in enumerate(phase_starts[:-1]):
        if tn[0] + ' Plateau' in phases[ps]:
            func = func + at.ge(t, ps) * at.lt(t, phase_starts[i+1]) * -sf
        elif tn[1] + ' Plateau' in phases[ps]:
            func = func + at.ge(t, ps) * at.lt(t, phase_starts[i+1]) * +sf
        elif tn[0] + ' to ' + tn[1] + ' Transition' in phases[ps]:
            tl = phase_starts[i+1] - ps  # + 1 # Transition lenght
            # + 1# Position on the transition relative to center
            tp = t - ps - (tl/2)
            func = func + (at.ge(t, ps) * at.lt(t, phase_starts[i+1])
                                        * (1/(tl+1)) * (tp+0.5) * (2*sf))
        elif tn[1] + ' to ' + tn[0] + ' Transition' in phases[ps]:
            tl = phase_starts[i+1] - ps  # + 1# Transition lenght
            # + 1 # Position on the transition relative to center
            tp = t - ps - (tl/2)
            func = func + (at.ge(t, ps) * at.lt(t, phase_starts[i+1])
                                        * (-1/(tl+1)) * (tp+0.5) * (2*sf))
    if (tn[0] + ' Plateau' in phases[phase_starts[-2]]
            or 'to ' + tn[0] in phases[phase_starts[-2]]):
        func = func + at.ge(t, phase_starts[-1]) * sf * -1
    else:
        func = func + at.ge(t, phase_starts[-1]) * sf * 1

    return pm.invlogit(func * adaptation + bias)


def get_model(phases,
              data,
              custom_priors={},
              custom_links=None,
              silent=False
):
    R"""
    Setup a PyMC model from an adaptation curve defined
    via the pahses as documented here: TODO

    Parameters
    ----------
    phases: list
        A list with dictionaries as described here: TODO.
    data: pandas DataFrame
        Dataframe with columns as described here: TODO.
    custom_priors: dictionary
        Dictionary with PyMC distributions definitions as 
        a string which is evaluated to set up the priors;
        defaults to empty dictionary
    custom_links: list
        Link function between group and participant level;
        defaults to None, which leads to identity links for
        all parameters except adaptation, which gets linked
        via an exponential function
    silent: bool
        If true, status outputs are surpressed; defaults to
        False
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
                print("Auto setting shift_mu to Normal(0, %f) where "
                      + "SD is 1/8 of the transition length" % (shift_mu_sigma))
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
                                            bias_[sp_idx, p_idx]))

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
    R"""
    Sample from the model to generate the posterior. Also stores the
    prior predictive and posterior predictive distribution in the trace.
    The samples are loaded from disk if an existing trace file is porived.

    Parameters
    ----------
    model: PyMC model created with get_model(...)
    samples: int
        Number of samples to be drawn in each chain after tuning;
        defaults to 2000
    tune: int
        Samples used to tune the NUTS sampler; defaults to 1000
    target_accept: float (in range 0..1)
        Target accept rate for the NUTS sampler; defaults to 0.75
    seed: int
        Seed for the random number generator; defaults to 0
    thin: int
        Keep only every nth sample to thin the trace; defaults
        to None
    silent: bool
        If true, status outputs are surpressed;
        defaults to False
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


def interactive(phases, static=False):
    R"""
    Interactive adaptation curve widget for jupyter notebooks.

    Parameters
    ----------
    phases: dictionary
        Dictionary with the phases as specified here TODO
    static: bool
        If true, a static version without sliders is 
        produced (e.g. for saving a ntoebook to PDF)
    """
    from ipywidgets import interact, widgets
    from IPython.display import clear_output
    end = get_end(phases)
    t = np.linspace(0, end-1, end)

    predicted = None

    def update(adaptation, shift, bias):
        global predicted
        clear_output(wait=True)
        fig = plt.figure(figsize=(14.5, 6))
        ax = fig.add_subplot(1, 1, 1)
        predicted = adaptation_curve(t, phases, adaptation, shift, bias)
        obj_frequency, labels = objective_frequency(t, phases)
        line, = plt.plot(t, predicted.eval(), color='blue', linewidth=3)
        plt.plot(t, obj_frequency, color='black', linewidth=3)
        plt.axhline(0.5, color='black')
        plt.ylim(0, 1)
        for tx in t:
            plt.axvline(x=tx, linestyle='-', linewidth=0.5, color='gray')
        plt.xticks(t, rotation=90)
        ax.set_xticklabels(labels)
    tl = np.mean(get_transistion_length(phases))

    if static == False:
        interact(update, adaptation=widgets.FloatSlider(min=0, max=10, value=1, step=0.01,
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
    R"""
    Get summary table.

    Parameters
    ----------
    trace: trace (ArviZ inference data object)
        Trace with the posterior samples
    static: list
        List of parameter names to include/exclude; defaults to
        ['~selected'], which excludes the latent selection probabilities
        at each cycle index to keep trace size in check.
    """
    summary = az.summary(trace, var_names=var_names, filter_vars='like')
    return summary


def get_diagnostic(trace, var_names=['~selected']):
    R"""
    Get diagnostics table.

    Parameters
    ----------
    trace: trace (ArviZ inference data object)
        Trace with the posterior samples
    static: list
        List of parameter names to include/exclude; defaults to
        ['~selected'], which excludes the latent selection probabilities
        at each cycle index to keep trace size in check.
    """
    table = get_table(trace, var_names)
    diag = pd.DataFrame(table[['ess_bulk', 'ess_tail', 'r_hat']].describe())
    return diag
