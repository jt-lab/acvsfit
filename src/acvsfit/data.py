import pandas as pd
import json 
from matplotlib.pylab import plt
import numpy as np

def aggregate(data):
    R"""
    Aggregate trial by trial selection data to count over
    cycle indices. 

    Parameters
    ----------
    data: pandas DataFrame
        Dataframe as described here: TODO
    """
    if 'Repetitions' in data:
        return data
    
    if 'Decline_First_Selection' not in data:
        data.loc[data['Starting_Plateau_Idx'] == 0, 'Decline_First_Selection'] = data['Selection']
        data.loc[data['Starting_Plateau_Idx'] == 1, 'Decline_First_Selection'] = 1 - data['Selection']
    data = data.groupby(['Cycle_Idx', 'Starting_Plateau_Idx',
              'Condition_Name',  'Participant_ID'])\
        .agg(
             Decline_First_Selection = pd.NamedAgg(column='Decline_First_Selection', aggfunc='sum'),
             Repetitions = pd.NamedAgg(column='Decline_First_Selection', aggfunc='count')
            )\
            .reset_index()
    
    data = data[data.Repetitions!=0]
    
    return data

def unaggregate(data):
    R"""
    Unaggregate count over cycle indices data to trial by trial selection data to. 

    Parameters
    ----------
    data: pandas DataFrame
        Dataframe as described here: TODO
    """

    if not 'Repetitions' in data:
        return data
    
    new_data = pd.DataFrame()
    
    for j,row in data.iterrows():
        for i in range(0, int(row['Repetitions'])):
            sel = 1 if i < row['Decline_First_Selection'] else 0
            if row['Starting_Plateau_Idx'] == 1:
                sel = 1-sel
            record = {
                'Participant_ID' : row['Participant_ID'],
                'Condition_Name' : row['Condition_Name'],
                'Cycle_Idx' : row['Cycle_Idx'],
                'Starting_Plateau_Idx' : row['Starting_Plateau_Idx'],
                'Selection' : sel,
            }
            new_data = new_data.append(record, ignore_index=True)
    new_data.reset_index(drop=True)
    return new_data


def get_end(phases):
    R"""
    Returns the end of the cycle

    Parameters
    ----------
    phases: Dictionary
        Dictionary with the phases as described here: TODO
    """
    return sorted(phases.keys())[-1]

def load_data(filename):
    R"""
    Load dataset. Right now this function simply wraps pandas.read_csv.
    In the future checks concerning the presence of certain columns 
    will be added.

    Parameters
    ----------
    filename: string
        Path to the file
    """
    return pd.read_csv(filename)

def load_phases(filename, plot=True):
    R"""
    Load phases from a .json file. 

    Parameters
    ----------
    filename: string
        Path to the file
    plot: bool
        If true, a representation of the phases will be plottet
    """
    indata = json.load(open(filename, 'r'))
    outdata = {int(k): indata[k] for k in indata}
    
    tn = get_type_names(outdata)
    
    f,ax = plt.subplots(1, figsize=(16, 1))
    
    if plot==True:
        end =  get_end(outdata)
        t = np.linspace(0, end-1, end)
  
        obj_frequency, labels = objective_frequency(t, outdata)
        
        plt.plot(t, obj_frequency, color = 'black' , linewidth = 3)
        plt.axhline(0.5, color = 'black')
        plt.ylim(0, 1)
    
        for tx in t:
            plt.axvline(x=tx, linestyle='-', linewidth=0.5, color='gray')
            plt.xticks(t, rotation=90)
            ax.set_xticklabels(labels)
        ax.set_title('Loaded ' + filename + ' with cycle structure:')
        ax.set_yticks([0,1])
        ax.set_yticklabels([tn[0] + ' Plateau', tn[1] + ' Plateau'])

    return outdata
    
def get_type_names(phases):
    R"""
    Get the names of the two stimulus types

    Parameters
    ----------
    phases: Dictionary
        Dictionary with the phases as described here: TODO
    """
    for k in phases:
        if 'Transition' in phases[k]:
            return phases[k].split(' Transition')[0].split(' to ')

def get_transistion_length(phases):
    R"""
    Returns a vector with the length of all transitions.

    Parameters
    ----------
    phases: Dictionary
        Dictionary with the phases as described here: TODO
    
    Returns
    -------
    Numpy array with the transision lenghts
    """
    tls = []
    phases_starts = sorted(phases.keys())
    for i,p in enumerate(phases_starts):
        if 'Transition' in phases[p]:
            tls.append(phases_starts[i+1] - phases_starts[i])
    return(np.array(tls))

def objective_frequency(cycle_index, phases):
    R"""
    Function which illustrates the objective proportion
    of one distractor type.

    Parameters
    ----------
    cycle_index: Series of integers; zero-based
    phases: Dictionary
        Dictionary with the phases as described here: TODO
 
    Returns
    -------
    Objective curve
    Tick labels
    """
    tn = get_type_names(phases)
    t = cycle_index
    labels = []
    func = (t < 0) * 0
    
    phase_starts = sorted(phases.keys())

    for i,ps in enumerate(phase_starts[:-1]):
        if tn[0] + ' Plateau' in phases[ps]:
            func = func + (t >= ps)  * (t < phase_starts[i+1])  * 0
        elif tn[1] + ' Plateau' in phases[ps]:
            func = func + (t >= ps) * (t < phase_starts[i+1])  * 1
        elif tn[0] + ' to ' + tn[1] + ' Transition' in phases[ps]:
            tl = phase_starts[i+1] - ps + 1# Transition lenght
            tp = t - ps + 1 #- (tl/2) # Position on the transition relative to center
            func = func +  ((t >= ps) * (t < phase_starts[i+1]) 
                                        * (1/tl) * tp)
        elif tn[1] + ' to ' + tn[0] + ' Transition' in phases[ps]:
            tl = phase_starts[i+1] - ps + 1 # Transition lenght
            tp = t - ps - tl +1 # Position on the transition relative to center
            func = func + ((t >= ps) * (t < phase_starts[i+1]) 
                                        * (-1/tl) * tp)
        # Fill label vector
        
        if 'Plateau' in phases[ps]:
            labels.extend(['P%d'%(j+1) for j in range(0, phase_starts[i+1]-ps)])
        elif 'Transition' in phases[ps]:
            labels.extend(['T%d'%(j+1) for j in range(0, phase_starts[i+1]-ps)])
 
    func = func + (t >= phase_starts[-1]) * 0 
    
    return func, labels
   