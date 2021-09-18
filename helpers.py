import scipy.stats as ss
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Functions to be used in main file

def zodiac_sign(day, month):
    signs = { 1: 'Saggitarius', 2: 'Capricorn', 3: 'Aquarius', 4: 'Pisces', 5: 'Aries', 6: 'Taurus',
7: 'Gemini', 8: 'Cancer', 9: 'Leo', 10: 'Virgo', 11: 'Libra', 12: 'Scorpio'}
    if month == 12: 
        return signs[1] if (day < 22) else signs[2]
    if month == 1:
        return signs[2] if (day < 20) else signs[3]
    if month == 2: 
        return signs[3] if (day < 19) else signs[4]
    if month == 3: 
        return signs[4] if (day < 21) else signs[5]
    if month == 4: 
        return signs[5] if (day < 20) else signs[6]
    if month == 5: 
        return signs[6] if (day < 21) else signs[7]
    if month == 6: 
        return signs[7] if (day < 21) else signs[8]
    if month == 7: 
        return signs[8] if (day < 23) else signs[9]
    if month == 8: 
        return signs[9] if (day < 23) else signs[10]
    if month == 9: 
        return signs[10] if (day < 23) else signs[11]
    if month == 10: 
        return signs[11] if (day < 23) else signs[12]
    if month == 11: 
        return signs[12] if (day < 22) else signs[1]

def position_replacer(position):
    defense = ['LWB', 'LB', 'CB', 'RB', 'RWB']
    midcamp = ['LM', 'CM', 'CDM', 'CAM', 'RM', 'LW', 'RW']
    offense = ['LF', 'ST', 'CF', 'RF']
    for pos in position:
        if pos == 'GK':
            return 'GoalKeeper'
        if pos in defense:
            return 'Defense'
        if pos in midcamp:
            return 'Midcamp'
        if pos in offense:
            return 'Offense'

def cramers_V(var1, var2):
    crosstab = np.array(pd.crosstab(var1, var2, rownames=None, colnames=None))
    chi2 = ss.chi2_contingency(crosstab)[0]
    n = np.sum(crosstab)
    phi2 = chi2 / n
    r, k = crosstab.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def cramers_application(dataframe):
    rows = []
    for var1 in dataframe:
        col = []
        for var2 in dataframe:
            cramers = cramers_V(dataframe[var1], dataframe[var2])
            col.append(round(cramers, 2))
        rows.append(col)
    return rows

def heatmap_builder(dataframe):
    mask = np.zeros_like(dataframe, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style("white"):
        ax = sns.heatmap(dataframe, mask=mask, vmin=0, vmax=1, square=True)
        plt.show()