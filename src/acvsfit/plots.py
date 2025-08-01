# Copyright (c) 2022 - 2025 Jan Tünnermann. All rights reserved.
# This work is licensed under the terms of the MIT license.  
# For a copy, see <https://opensource.org/licenses/MIT>.

import pymc as pm
import numpy as np
import arviz as az
import acvsfit
from .acvsfit import get_type_names, adaptation_curve, get_transistion_length
from .data import aggregate, unaggregate, get_end
import seaborn as sns
from matplotlib.pylab import plt
from matplotlib.patches import Rectangle
from IPython.display import display, Markdown, Latex
from arviz.plots.plot_utils import calculate_point_estimate
from arviz.rcparams import rcParams
from arviz import hdi
from IPython.display import display, HTML

def plot_empirical_curves(data, phases, colors=None,
                          label_every_nth=1, spmode='tangle',
                          points_only=False, ax=None):
    
    data = aggregate(data)

    tn = get_type_names(phases)
    cs = data.Condition_Name.unique()
    sps = data.Starting_Plateau_Idx.unique().astype(int)
    
    if colors == None:
        colors = {}
        for c in cs:
            colors[c] = ('blue', 'purpel')

    if ax is None:
        f, axs = plt.subplots(len(cs), 1,sharex=True, figsize = (14,3*len(cs)))
    else:
        axs = ax
    if not hasattr(axs,'__len__'):
        axs = [axs] # For compatibiliy with list access
    
    d = data
    d['Cycle State'] = data['Cycle_Idx']
    d[str(tn[0]) + ' Selections'] = data['Decline_First_Selection']
    d[str(tn[1]) + ' Selections'] = data['Decline_First_Selection']
    
    if spmode=='tangle':
        d[str(tn[0]) + ' Selections'] = 1 - d[str(tn[0]) + ' Selections'] / data['Repetitions']
        plotlabels =  [tn[i] + ' starting plateau' for i in range(0,2)]
        ylabel = 'Proportion\n' + tn[1] + ' Selections'
    else:
        plotlabels = [tn[i] + ' selected, ' + tn[i] +' starting plateau' for i in range(0,2)]
        ylabel = 'Proportion\n decline-first selections'
       
    d[str(tn[1]) + ' Selections'] = d[str(tn[1]) + ' Selections'] / data['Repetitions']

    obj_freq, labels = acvsfit.objective_frequency(np.arange(get_end(phases)), phases)
    labels_ = [labels[i] if i%label_every_nth==0 else '' for i in range(0, len(labels))]                                                           

    for c,condition in enumerate(cs):
        
        axs[c].plot(obj_freq, color="gainsboro")
        axs[c].axhline(y=0.5, color="gainsboro", linestyle='--')
        
        #if draw_transition_center is not None:
        #    axs[c].axhline(y=draw_transition_center,
        #                   color="gainsboro", linestyle='--')
        
        for il,l in enumerate(labels[:-1]): 
            if 'P' in l and 'P' in labels[il+1]:
                axs[c].add_patch(Rectangle((il, 0), 1, 1,
                            alpha=0.2,
                            facecolor="gainsboro"))
            
        for s in sps:
            dsp = d[(d.Condition_Name==condition) &
                    (d.Starting_Plateau_Idx==s)].reset_index()
            if not points_only:
                sns.lineplot(data=dsp, x='Cycle State',
                    y=tn[s] + ' Selections', ax=axs[c],
                    linestyle='dashed', label=plotlabels[s],
                    marker='x', color=colors[condition][s])
            else:
                sns.lineplot(data=dsp, x='Cycle State', y=tn[s] + ' Selections',
                    ax=axs[c], linestyle='', errorbar='se', label= plotlabels[s],
                    err_style='bars', marker='o', color=colors[condition][s])
            
            axs[c].legend(loc='upper right')
            axs[c].set_ylabel(ylabel)
            axs[c].set_title(condition)
            axs[c].set_xticks(np.arange(get_end(phases)))
            axs[c].set_xticklabels(labels_, rotation=90)

    sns.despine(bottom = False, left = False)

    
    
def plot_empirical_participant_curves(data,
                                      phases,
                                      colors=None,
                                      columns=6,
                                      spmode = 'tangle',
                                      connected=True): 

    data = unaggregate(data) 
    
    tn = get_type_names(phases)
    cs = data.Condition_Name.unique()
    sps = data.Starting_Plateau_Idx.unique().astype(int)
    
    if colors == None:
        colors = {}
        for c in cs:
            colors[c] = ('blue', 'purple')
    d=data
    d['Cycle Index'] = data['Cycle_Idx']
    d['Participant'] = data['Participant_ID']
    d['Starting Plateau'] = d['Starting_Plateau_Idx']
    
    
    if 'Decline_First_Selection' not in d:
        d.loc[d['Starting_Plateau_Idx'] == 0, 'Proportion Selected'] = d['Selection']
        d.loc[d['Starting_Plateau_Idx'] == 1, 'Proportion Selected'] = 1 - d['Selection']
    else:
        d['Proportion Selected'] =  d['Decline_First_Selection']
       
    if spmode == 'tangle':
        d.loc[d['Starting_Plateau_Idx'] == 0, 'Proportion Selected'] = 1 - d['Proportion Selected']
        
    kwargs={}
    if not connected:
        kwargs={'marker' : 'o', 'linestyle' : ''}
        
    for c,condition in  enumerate(cs):
        dsp = d[(d.Condition_Name==condition)].reset_index()
        sns.relplot(data=dsp, kind='line', x='Cycle Index', y='Proportion Selected',
                  col="Participant", col_wrap=columns, hue='Starting Plateau',
                  palette=colors[condition], height=1.6, aspect=1.5, **kwargs)
        plt.suptitle(condition)

def plot_prior_simulations(model, phases, fix_adaptation=None,
                          fix_shift=None, fix_bias=None, upper_limit=1, simulations=100, ax=None):
    
   
    
    with model:
        trace = pm.sample_prior_predictive()
        
    display(trace)
    
    if ax is None:
        f, ax=plt.subplots(1)

    
    end =  get_end(phases)
    ci = np.linspace(0, end-1, end)
   
    if fix_adaptation is None:
        adaptation = trace.prior['adaptation'].values[0,-simulations:,0,0].reshape(-1,1)
    else:
        adaptation = fix_adaptation
    if fix_shift is None:
        shift = trace.prior['shift'].values[0,-simulations:,0,0].reshape(-1,1)
    else:
        shift = fix_shift

    if fix_bias is None:
        vals = trace.prior['bias'].values
        if vals.ndim == 3:
            bias = vals.reshape[0,-simulations:,0](-1,1)
        else:
            bias = vals.reshape[0,-simulations:,0,0](-1,1)
    else:
        bias = fix_bias

    adaptation_curves = adaptation_curve(ci, phases, adaptation=adaptation, shift=shift, bias=bias, upper_limit=upper_limit).eval()
    ax.plot(adaptation_curves.T)
    
def plot_priors(model,
                seed=0,
                axs=None,
                labels=('A', 'B', 'C')):
    R"""
    Plot prior predictive distributions for the parameters.
    adaptation τ, shift δ, and bias β. on the participant level.
    Parameters
    ----------
    model: A (pymc) model generated with :func:`~acvsfit.get_model.
    axs: None, matplotlib axes object. 
        If None, a new one will be created.
    labels: A list of strings (typically one character) to
        be used to label the plot panels. Defaults to ('A', 'B', 'C')
    """
        
    with model:
        trace = pm.sample_prior_predictive(random_seed=seed)
    
    if axs is None:
        f, axs = plt.subplots(1,3, figsize=(10,3))
    else:
        if axs.shape != (1,3):
            raise RuntimeError("Shape of the axis provided to the axs argument must be (1,3)")
    with az.rc_context({"stats.ci_prob": 0.99}):
        az.plot_density(trace.prior["adaptation"][0,:,0,0], ax=axs[0], point_estimate=None)
        axs[0].set_title('Adaptation $\\tau$')
        axs[0].set_xlim(0,10)
        az.plot_density(trace.prior["shift"][0,:,0,0], ax=axs[1], point_estimate=None)
        axs[1].set_title('Shift $\delta$')
        vals = trace.prior["bias"].values
        if vals.ndim == 3:
            az.plot_density(trace.prior["bias"][0,:,0], ax=axs[2], point_estimate=None)
        else:
            az.plot_density(trace.prior["bias"][0,:,0,0], ax=axs[2], point_estimate=None)
        axs[2].set_title('Bias $\\beta$')

    
    for i, label in enumerate(labels):
        axs.flatten()[i].text(-0.08, 1.08, label, transform=axs.flatten()[i].transAxes,
          fontsize=16, fontweight='bold', va='top', ha='right')
    

    
def plot_prior_simulations_quartet(model,
                                   phases,
                                   seed=0,
                                   simulations=100,
                                   upper_limit=1,
                                   axs=None,
                                   labels=('A', 'B', 'C', 'D')):
    
    with model:
        trace = pm.sample_prior_predictive(random_seed=seed)
    
    if axs is None:
        f, axs = plt.subplots(2,2, figsize=(10,6))
    else:
        if axs.shape != (2,2):
            raise RuntimeError("Shape of the axis provided to the axs argument must be (2,2)")
            
    end =  get_end(phases)
    ci = np.linspace(0, end-1, end)
   
    adaptation=trace.prior['adaptation'].values[0,-simulations:,0,0].reshape(-1,1)
    shift=trace.prior['shift'].values[0,-simulations:,0,0].reshape(-1,1)
    vals = trace.prior['bias'].values
    if vals.ndim == 3:
        bias=vals[0,-simulations:,0].reshape(-1,1)
    else:
        bias=vals[0,-simulations:,0,0].reshape(-1,1)

    acs = adaptation_curve(ci, phases, adaptation=adaptation,
                           shift=shift, bias=bias, upper_limit=upper_limit).eval().T
    acs_adaptation = adaptation_curve(ci, phases, adaptation=adaptation,
                                      shift=0, bias=0, upper_limit=upper_limit).eval().T
    acs_shift = adaptation_curve(ci, phases, adaptation=1,
                                 shift=shift, bias=0, upper_limit=upper_limit).eval().T
    acs_bias = adaptation_curve(ci, phases, adaptation=0,
                                shift=0, bias=bias, upper_limit=upper_limit).eval().T
    axs[0,0].plot(acs)
    axs[0,0].set_title('All parameters')
    axs[0,0].set_ylim(0, 1)
    axs[0,1].plot(acs_adaptation)
    axs[0,1].set_title('Adaptation τ')
    axs[0,1].set_ylim(0, 1)
    axs[1,0].plot(acs_shift)
    axs[1,0].set_title('Shift δ')
    axs[1,0].set_ylim(0, 1)
    axs[1,1].plot(acs_bias)
    axs[1,1].set_title('Bias β')
    axs[1,1].set_ylim(0, 1)
    
    for i, label in enumerate(labels):
        axs.flatten()[i].text(-0.08, 1.08, label, transform=axs.flatten()[i].transAxes,
          fontsize=16, fontweight='bold', va='top', ha='right')
    
    #plt.tight_layout()    
    plt.subplots_adjust(wspace=0.15)
    
    
    

    
def plot_examples(model, phases,
                  adaptation_tau = [0, 0.1, 0.3, 1, 10], 
                  shift_delta = 'auto',
                  bias_beta = [-5, -2, 0, 2, 5],
                  fixed_values = {'tau' : [1, 0.05], 'beta' : [0,0], 'delta' : [0,0]},
                  upper_limit=1,
                  axs=None,
                  labels=('A', 'B', 'C'),
                  fignum='1',
                  caption='notebook-default',
                  ylabel='Proportion\n Decline-first selections',
                  save=''):
    
    if axs is None:
        f, axs = plt.subplots(1,3, figsize=(14,3))
    else:
        if axs.shape != (1,3):
            raise RuntimeError("Shape of the axis provided to the axs argument must be (1,2)")
            
    if shift_delta == 'auto':
        tl = np.mean(get_transistion_length(phases))
        shift_delta = [0, tl*0.10, tl*0.20, tl*0.30, tl*0.40]
    
    end = get_end(phases)
    ci = np.linspace(0, end -1, end)

    for i,tau in enumerate(adaptation_tau):
        delta = fixed_values['delta'][0]
        beta = fixed_values['beta'][0]
        acs = adaptation_curve(ci, phases, adaptation=tau,
                               shift=delta, bias=beta, upper_limit=upper_limit).eval().T
        axs[0].plot(acs, label = 'τ = %.2f (δ = %.2f, β = %.2f)'%(tau,delta,beta))
    axs[0].set_title('Adaptation τ')
    legend = axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.65), ncol= 1)
    legend.get_frame().set_alpha(None)
    
    #legend.get_frame().set_facecolor((0, 0, 1, 0.1))
    for delta in shift_delta:
        tau = fixed_values['tau'][0]
        beta = fixed_values['beta'][1]
        acs = adaptation_curve(ci, phases, adaptation=tau,
                               shift=delta, bias=beta, upper_limit=upper_limit).eval().T
        axs[1].plot(acs, label = 'δ = %.2f (τ = %.2f, β = %.2f)'%(delta,tau,beta))
    axs[1].set_title('Shift δ')
    legend = axs[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.65), ncol= 1)
    legend.get_frame().set_alpha(None)
        
    for beta in bias_beta:
        delta = fixed_values['delta'][1]
        tau = fixed_values['tau'][1]
        acs = adaptation_curve(ci, phases, adaptation=tau,
                               shift=delta, bias=beta, upper_limit=upper_limit).eval().T
        axs[2].plot(acs, label = 'β = %.2f (τ = %.2f, δ = %.2f)'%(beta,tau,delta))
        axs[2].set_title('Bias β')
    legend = axs[2].legend(loc="lower center", bbox_to_anchor=(0.5, -0.65), ncol= 1)
    legend.get_frame().set_alpha(None)
   
    for i, label in enumerate(labels):
        axs.flatten()[i].text(-0.08, 1.08, label, transform=axs.flatten()[i].transAxes,
          fontsize=16, fontweight='bold', va='top', ha='right')
    
    #plt.tight_layout()
    axs[0].set_ylabel(ylabel)
    plt.subplots_adjust(wspace=0.15)
    
    if save != '':
        plt.savefig(save, bbox_inches='tight')
    
    if caption=='notebook-default':
        if axs is not None:
            display(f)
        display(Markdown(
        ("***Figure %s.*** *Visualizations of the adaptation curve with exemplary parameters." 
        + " In each panel, the value of one parameter is varied while the others are held constant."
        + " **%s** Different values for adaptation τ."
        + " **%s** Different values for shift δ."
        + " **%s** Different values for bias β."
        + " The curves depict the predicted proportion of selections of the target"
        + " that shares the feature with the large distractor set in the starting"
        + " plateau.*")
        %(fignum, labels[0], labels[1], labels[2])
        ))
        plt.close()
    
    
def plot_trace(trace, title='', **kwargs):
    az.plot_trace(trace, 
                  var_names=[v for v in list(trace.posterior.data_vars) 
                            if not 'corr' in v and not 'selected' in v], **kwargs)
    plt.gcf().suptitle(title, size=18)

    correlations = [v for v in list(trace.posterior.data_vars) if 'corr' in v ]
    for c in correlations:
        matrix_size = trace.posterior.dims[c + '_dim_0']
        pairs = []
        for i in range(0, matrix_size):
            for j in range(0, matrix_size):
                if i != j and (j,i) not in pairs:
                    pairs.append((i,j))
        for p in pairs:
            dims = {}
            dims[c + '_dim_0'] = p[0]
            dims[c + '_dim_1'] = p[1]
            az.plot_trace(trace, var_names=[c], filter_vars='like', coords=dims)
            plt.title('Correlation between ' + str(p))
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
   
   

def plot_participant_parameter_posterior(trace,
                                         data,
                                         parameter,
                                         colors=None,
                                         ax=None):
    
    data = unaggregate(data)
    
    if colors == None:
        colors = {}
        for c in cs:
            colors[c] = ('blue', 'purple')
            
    
    color_list = [colors[c][0] for c in data.Condition_Name.unique()]
    conditions_count = trace.posterior[parameter].shape[-1]
    participant_count = data.Participant_ID.nunique()
    trace_list = []

    if len(trace.posterior[parameter].shape) > 3:
        for i in range(0, conditions_count):
            trace_list.append(
                trace.posterior[parameter][:,:,:,i]
            )
    else:
        trace_list = trace
    #    color_list = ('gray')
        
    if ax is None:
        f,ax = plt.subplots(figsize=(12,2*participant_count))
    az.plot_forest(trace_list,
                   var_names=[parameter], ax=ax,
                   combined=True, colors=color_list)
    
    #if len(trace.posterior[parameter].shape) > 3:
    #label_list = list(data.Condition_Name.unique())
    #label_list.reverse()
    #ax.legend(label_list)

    legend_handles = [
        mlines.Line2D([], [], color=color, label=cond, linewidth=2)
        for cond, color in colors.items()
    ]
    ax.legend(handles=legend_handles)                                 

    participants_list = list(data.Participant_ID.unique())
    participants_list.reverse()
    ax.set_yticklabels(participants_list, ha='left')
    plt.tight_layout()
    ax.set_title(pp(parameter))
    
def plot_participant_posteriors(trace,
                                data,
                                parameters=['adaptation', 'shift', 'bias'],
                                colors=None,
                                ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 3, figsize=(14, data.Participant_ID.nunique()/5))

        for pi,parameter in enumerate(parameters):
            plot_participant_parameter_posterior(trace, data, parameter,
                                             colors, ax=ax[pi])
        
        
def plot_group_posteriors(trace,
                               data,
                               colors=None,
                               ax=None,
                               save=''):
    
    parameters = ['adaptation', 'shift', 'bias']
    parameter_names = ['Adaptation $τ_µ$', 'Shift $δ_µ$', 'Bias $β_µ$']
    
    cs = data.Condition_Name.nunique()
    
    if ax is None:
        f, axs = plt.subplots(cs, 3, figsize=(14, 2 * cs), sharex='col')
    else:
        axs = ax
    
    if cs == 1:
        axs = np.expand_dims(axs, axis=0) # For compatibiliy with list access
    

    for pi,p in enumerate(parameters):
        if len(trace.posterior[p].shape) > 3:
            for ci,c in enumerate(data.Condition_Name.unique()):
                az.plot_posterior(trace.posterior[p + '_µ'][:,:,ci],
                                  ax=axs[ci,pi], color=colors[c][0],
                                  ref_val=0, ref_val_color='gray',
                                  lw=4, alpha=0.75)
                if ci==0:
                    axs[ci,pi].set_title(parameter_names[pi])
                else:
                    axs[ci,pi].set_title(None)
                axs[ci, 0].set_ylabel(c)
        else:
            az.plot_posterior(trace.posterior[p + '_µ'][:,:], ax=axs[0,pi],
                              ref_val=0, ref_val_color='gray',
                              color=colors[c][0], lw=4, alpha=0.75)
            axs[0,pi].set_title(parameter_names[pi])
            for i in range(1,data.Condition_Name.nunique()):
                axs[i,pi].axis('off')
            axs[0, 0].set_ylabel(c, labelpad=25)
    plt.tight_layout()
    if save != '':
        plt.savefig(save, bbox_inches='tight')
    
def plot_participant_ppc(trace, data, phases, colors=None,
                         hdi_samples=200, spmode='tangle',
                         marker='o'):
    
    R"""
    Plot posterior predictive check on the participant level.
    ----------
    tarce: An ArviZ inference data object that does include the 
        posterior predictive group. 
    data: The (pandas) input data frame TODO: Which columns are required
    colors: A dictionary that maps condition names to tupels with primary
        and secondary colors. See TODO. Defaults to None, which in turn
        sets blue and purple as primary and secondary colors for all condition.
    hdi_sample: Number of posterior predictive samples used at each
        data point to estimate the HDI. Defaults to 200. Large values
        might lead to very slow plotting.
    """
    
    data_agg = acvsfit.aggregate(data)
    #data_agg['Cycle_Idx'] = ni
    cs = data_agg.Condition_Name.unique()
    sps = data_agg.Starting_Plateau_Idx.unique().astype(int)

    if colors == None:
        colors = {}
        for c in cs:
            colors[c] = ('blue', 'purple')
                                           
                                               
    tn = get_type_names(phases)

    d=data_agg
    
    d['Cycle Index'] = d['Cycle_Idx']
    d['Participant'] = d['Participant_ID']
    d['Starting Plateau'] = d['Starting_Plateau_Idx']
    

    if spmode == 'tangle':
        ylabel = 'Proportion\n' + tn[1] + ' selected'
        d[ylabel] = 1 - (d['Decline_First_Selection'] / d['Repetitions'])
        d.loc[d['Starting_Plateau_Idx']==1, ylabel] = 1 - d[ylabel]
    else:
        ylabel = 'Proportion\n decline-first selected'
        d[ylabel] = d['Decline_First_Selection'] / d['Repetitions']
    
    
    
    
    samples = trace.posterior_predictive.y.to_numpy()[0,-hdi_samples:,:] / d['Repetitions'].values
    d = d.loc[d.index.repeat(hdi_samples)]
    d['y'] = samples.T.flatten()
    if spmode=='tangle':
        d.loc[d['Starting_Plateau_Idx']==0, 'y'] = 1 - d.loc[d['Starting_Plateau_Idx']==0, 'y']

    for c,condition in enumerate(cs):
        dsp = d[(d.Condition_Name==condition)].reset_index()
        g = sns.relplot(data=dsp, kind='line', x='Cycle Index', y='y',
                      col="Participant", col_wrap=6, hue='Starting Plateau',
                      palette=colors[condition], height=1.6, aspect=1.5,
                      errorbar=lambda x: az.hdi(np.array(x)), err_style='band')

        g.map_dataframe(sns.scatterplot, x='Cycle Index', y=ylabel,
                       hue='Starting Plateau', palette=colors[condition], marker=marker)
        plt.suptitle(condition)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
def plot_group_ppc(trace, data, phases, colors=None,
                                          hdi_samples=1000, spmode = 'tangle',
                                          labels=[], ax=None, save='', **kwarg):
    
    R"""
    Plot posterior predictive check on the group level.
    ----------
    tarce: An ArviZ inference data object that does include the 
        posterior predictive group. 
    data: The (pandas) input data frame TODO: Which columns are required
    colors: A dictionary that maps condition names to tupels with primary
        and secondary colors. See TODO. Defaults to None, which in turn
        sets blue and purple as primary and secondary colors for all condition.
    hdi_sample: Number of posterior predictive samples used at each
        data point to estimate the HDI. Defaults to 200. Large values
        might lead to very slow plotting.
    """
    
    tn = get_type_names(phases)
    cs = data.Condition_Name.unique()
    sps = data.Starting_Plateau_Idx.unique().astype(int)
    
    if colors is None:
        colors = {}
        for c in cs:
            colors[c] = ('blue', 'purple')

    if ax is None:
        f, axs = plt.subplots(len(cs), 1,sharex=True, figsize = (14,3*len(cs)))
    else:
        axs = ax
    if not hasattr(axs,'__len__'):
        axs = [axs] # For compatibiliy with list access
    
    plot_empirical_curves(data, phases, colors, spmode=spmode, 
                          points_only=True, ax=axs, **kwarg)
    
    
    samples = az.extract_dataset(trace, group='posterior_predictive', var_names='y',
                       num_samples=hdi_samples).values #[0,:,:]

    da = acvsfit.aggregate(data)
    n = (da.Condition_Name.nunique()
        * da.Starting_Plateau_Idx.nunique()
        * da.Cycle_Idx.nunique())

    cycle_list = np.array([])
    mode_list = np.array([])
    low_hdi_list = np.array([])
    high_hdi_list = np.array([])
    sp_idx_list = np.array([])
    cn_list = np.array([])

    for c,cn in enumerate(da.Condition_Name.unique()):
        for sp in da.Starting_Plateau_Idx.unique().astype(int):
            for ci in da.Cycle_Idx.unique():
                cmask = da.Condition_Name==cn
                spmask = da.Starting_Plateau_Idx==sp
                cimask = da.Cycle_Idx==ci

                reps = da[cmask&spmask&cimask].Repetitions.values
                if len(reps) > 0:

                    samples_rel = samples[cmask&spmask&cimask].T/reps

                    if sp==0 and spmode=='tangle':
                        samples_rel = 1-samples_rel                        

                    mean_over_ps = np.mean(samples_rel, axis=0)
                    mode_ = mean_over_ps.mean() 
                    # calculate_point_estimate(point_estimate=rcParams["plot.point_estimate"],
                    # values=mean_over_ps)
                    hdi_ = hdi(np.array(mean_over_ps), ci_prob=0.95)
                    cycle_list = np.append(cycle_list, ci)
                    mode_list = np.append(mode_list, mode_)
                    low_hdi_list = np.append(low_hdi_list, hdi_[0])
                    high_hdi_list = np.append(high_hdi_list, hdi_[1])
                    sp_idx_list = np.append(sp_idx_list, sp)
                    cn_list = np.append(cn_list, cn)


    cs = da.Condition_Name.unique()
    sps = da.Starting_Plateau_Idx.unique().astype(int)

    for c,condition in enumerate(da.Condition_Name.unique()):
        for sp in [0,1]:
            mode_curve = mode_list[(sp_idx_list==sp) & (cn_list==condition)]
            low_hdi_curve = low_hdi_list[(sp_idx_list==sp) & (cn_list==condition)]
            high_hdi_curve = high_hdi_list[(sp_idx_list==sp) & (cn_list==condition)]
            mode_curve_idx = cycle_list[(sp_idx_list==sp) & (cn_list==condition)]

            axs[c].plot(mode_curve_idx, mode_curve,
                     color=colors[condition][sp])

            axs[c].fill_between(mode_curve_idx, low_hdi_curve, high_hdi_curve,
                             color=colors[condition][sp], alpha=0.05)
            #da['Proportion \n Decline-First Selections'] = da['Decline_First_Selection'] / da['Repetitions']
            #d = da[(da['Starting_Plateau_Idx']==sp) & (da['Condition_Name']==condition)]
            
        axs[c].legend(loc='lower right')
        if len(labels) > c:
            axs[c].text(-0.08, 1.08, labels[c], transform=axs[c].transAxes,
                             fontsize=16, fontweight='bold', va='top', ha='right')
        
    if save != '':
        plt.savefig(save, bbox_inches='tight')
            
def plot_priors_vs_posteriors(trace, color, condition_name=None, ax_lims={}):
    f,ax = plt.subplots(2,3, figsize=(16,4))
    f2,ax2 = plt.subplots(1)
    for x, para in enumerate(['adaptation', 'shift', 'bias']):
        for y, moment in enumerate(['µ', 'σ']):
            if para == 'adaptation':
                name =  para + '_' + moment + '_log'
            else:
                name =  para + '_' + moment
            az.plot_dist_comparison(trace, var_names=name,
                ax=np.array([[ax2, ax2, ax[y,x]]]),
                posterior_kwargs={'plot_kwargs' :
                      {'color' : color, 'linewidth' : 4 }},
                prior_kwargs={'plot_kwargs' :
                      {'color' : color,'linestyle' : '--', 'linewidth' : 4}})
            if (para + '_' + moment) in ax_lims:
                l = ax_lims[para + '_' + moment]
                ax[y,x].set_xlim(l[0],l[1])
            ax[y,x].get_legend().remove()
            ax[y,x].set_title(name)
            ax[y,x].set_xlabel(None)
    f.suptitle(condition_name)
    plt.close()
    plt.tight_layout()
    

def display_side_by_side(dfs:list, captions:list):
    """
        https://stackoverflow.com/a/57832026
        Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
    """
    output = ""
    combined = dict(zip(captions, dfs))
    for caption, df in combined.items():
        output += df.style.set_table_attributes("style='display:inline'").set_caption(caption)._repr_html_()
        output += "\xa0\xa0\xa0"
    display(HTML(output))

def pp(name):
    """
        pp -- 'prettyfy parameter name'. Helper function to turn
        parameter names from the PyMC model into nice names for 
        plot titles, etc.
    """
    translations = {
        'adaptation': 'Adaptation $\\tau$',
        'shift': 'Shift $\delta$',
        'bias' : 'Bias $\\beta$',
    }
    return translations[name]
