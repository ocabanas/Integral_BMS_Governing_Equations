import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
from copy import deepcopy
import pynumdiff
import matplotlib.gridspec as gridspec
import pickle
from matplotlib.ticker import ScalarFormatter,StrMethodFormatter
try:
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/Arial.ttf')
    prop = font_manager.FontProperties(fname='/usr/share/fonts/truetype/msttcorefonts/Arial.ttf')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = prop.get_name()
except Exception as e:
    print('Could not load Arial font')
    print(e)

colors={
    'data': 'grey',
    'gt':'k',
    'fd':'#7fc97f',
    'sd':'#fdc086',
    'I-BMS':'#bdc9e1',
    'FD-BMS':'#74a9cf',
    'SD-BMS':'#0570b0',
    'WS':'#fdae6b',
    'IS':'#e6550d',
#    'A-BMS':
    'Log':'#b96902',
    'Gom':'#9a0eea'
}




fig2 = plt.figure(figsize=(18.3/2.54, 15/2.54))
gs = gridspec.GridSpec(2, 2)

with open('Logistic/learnability.pkl','rb') as file:
    data_store=pickle.load(file=file)

s_array=data_store['s_array']
ode_lernability=data_store['ode_lernability']
ode_lernability_err=data_store['ode_lernability_err']
fit_lernability=data_store['fit_lernability']
fit_lernability_err=data_store['fit_lernability_err']
smooth_lernability=data_store['smooth_lernability']
smooth_lernability_err=data_store['smooth_lernability_err']
true_best_error_ode_final=data_store['true_best_error_ode_final']
true_best_error_fit_final=data_store['true_best_error_fit_final']
true_best_error_smth_final=data_store['true_best_error_smth_final']
data_best_error_ode_final=data_store['data_best_error_ode_final']
data_best_error_fit_final=data_store['data_best_error_fit_final']
data_best_error_smth_final=data_store['data_best_error_smth_final']
data_best_error_ode_final_err=data_store['data_best_error_ode_final_err']
data_best_error_fit_final_err=data_store['data_best_error_fit_final_err']
data_best_error_smth_final_err=data_store['data_best_error_smth_final_err']

ax=fig2.add_subplot(gs[0,0])
ax.errorbar(s_array,ode_lernability,ode_lernability_err,c=colors['I-BMS'],marker='o',ms=4.,label='I-BMS',lw=1)
ax.errorbar(s_array,fit_lernability,fit_lernability_err,c=colors['FD-BMS'],marker='s',ms=4.,label='FD-BMS',lw=1)
ax.errorbar(s_array,smooth_lernability,smooth_lernability_err,c=colors['SD-BMS'],marker='D',ms=4.,label='SD-BMS',lw=1)
#ax.legend(loc='best',frameon=False)
ax.set_xscale('log')
ax.set_xlim(left=0.)
#ax.set_xlabel('Observational noise, $\sigma$')
ax.set_ylabel('Learnability')
ax.set_xticklabels([])

ax=fig2.add_subplot(gs[1,0])
ax.errorbar(s_array,np.divide(data_best_error_ode_final,s_array),np.divide(data_best_error_ode_final_err,s_array),c=colors['I-BMS'],marker='o',ms=4.,label='I-BMS',lw=1)
ax.errorbar(s_array,np.divide(data_best_error_fit_final,s_array),np.divide(data_best_error_fit_final_err,s_array),c=colors['FD-BMS'],marker='s',ms=4.,label='FD-BMS',lw=1)
ax.errorbar(s_array,np.divide(data_best_error_smth_final,s_array),np.divide(data_best_error_smth_final_err,s_array),c=colors['SD-BMS'],marker='D',ms=4.,label='SD-BMS',lw=1)

#ax.legend(loc='best',frameon=False)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([0,1])
ax.set_ylim([0,10])
ax.set_xlabel('Observational noise, $\sigma$')
ax.set_ylabel('Normalized error, RSME/$\sigma$')
ax.yaxis.set_major_formatter(ScalarFormatter())
#ax.yaxis.set_minor_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))


with open('Lotka-Volterra/learnability.pkl','rb') as file:
    data_store = pickle.load(file=file)
s_array=data_store['s_array']
ode_lernability=data_store['ode_lernability']
ode_lernability_err=data_store['ode_lernability_err']
fit_lernability=data_store['fit_lernability']
fit_lernability_err=data_store['fit_lernability_err']
smooth_lernability=data_store['smooth_lernability']
smooth_lernability_err=data_store['smooth_lernability_err']
data_best_error_ode_final=data_store['data_best_error_ode_final']
data_best_error_fit_final=data_store['data_best_error_fit_final']
data_best_error_smth_final=data_store['data_best_error_smth_final']
data_best_error_ode_final_err=data_store['data_best_error_ode_final_err']
data_best_error_fit_final_err=data_store['data_best_error_fit_final_err']
data_best_error_smth_final_err=data_store['data_best_error_smth_final_err']


ax=fig2.add_subplot(gs[0,1])
ax.errorbar(s_array,ode_lernability,ode_lernability_err,c=colors['I-BMS'],marker='o',ms=4.,label='I-BMS',lw=1)
ax.errorbar(s_array,fit_lernability,fit_lernability_err,c=colors['FD-BMS'],marker='s',ms=4.,label='FD-BMS',lw=1)
ax.errorbar(s_array,smooth_lernability,smooth_lernability_err,c=colors['SD-BMS'],marker='D',ms=4.,label='SD-BMS',lw=1)
#ax.legend(loc='best',frameon=False)
ax.set_xscale('log')
ax.set_xlim(left=0.)
#ax.set_xlabel('Observational noise, $\sigma$')
#ax.set_ylabel('Learnability')
ax.set_xticklabels([])

ax=fig2.add_subplot(gs[1,1])
ax.errorbar(s_array,np.divide(data_best_error_ode_final,s_array),np.divide(data_best_error_ode_final_err,s_array),c=colors['I-BMS'],marker='o',ms=4.,label='I-BMS',lw=1)
ax.errorbar(s_array,np.divide(data_best_error_fit_final,s_array),np.divide(data_best_error_fit_final_err,s_array),c=colors['FD-BMS'],marker='s',ms=4.,label='FD-BMS',lw=1)
ax.errorbar(s_array,np.divide(data_best_error_smth_final,s_array),np.divide(data_best_error_smth_final_err,s_array),c=colors['SD-BMS'],marker='D',ms=4.,label='SD-BMS',lw=1)

#ax.legend(loc='best',frameon=False)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([0,7])
ax.set_ylim([0,20.])
ax.set_xlabel('Observational noise, $\sigma$')
#ax.set_ylabel('Normalized error, RSME/$\sigma$')
ax.yaxis.set_major_formatter(ScalarFormatter())
#ax.yaxis.set_minor_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
handles,labels = ax.get_legend_handles_labels()
plt.legend(handles, labels, frameon=False, ncol=len(handles))


#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).axes[0].set(position=[0.1349, 0.5894, 0.3523, 0.35])
plt.figure(1).axes[0].text(0.85, 0.85, 'A', transform=plt.figure(1).axes[0].transAxes, fontsize=14., weight='bold')  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[1].set(position=[0.1349, 0.1752, 0.3523, 0.35])
plt.figure(1).axes[1].text(0.85, 0.85, 'C', transform=plt.figure(1).axes[1].transAxes, fontsize=14., weight='bold')  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).axes[2].set(position=[0.5835, 0.5894, 0.3523, 0.35])
plt.figure(1).axes[2].text(0.85, 0.85, 'B', transform=plt.figure(1).axes[2].transAxes, fontsize=14., weight='bold')  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[3].legend(loc=(-0.8348, -0.4336), frameon=False, ncols=3)
plt.figure(1).axes[3].set(position=[0.5835, 0.1752, 0.3523, 0.35])
plt.figure(1).axes[3].text(0.85, 0.85, 'D', transform=plt.figure(1).axes[3].transAxes, fontsize=14., weight='bold')  # id=plt.figure(1).axes[3].texts[0].new
plt.figure(1).text(0.2812, 0.9687, 'Logistic', transform=plt.figure(1).transFigure, fontsize=14., weight='bold')  # id=plt.figure(1).texts[0].new
plt.figure(1).text(0.7076, 0.9687, 'Lotka-Volterra', transform=plt.figure(1).transFigure, fontsize=14., weight='bold')  # id=plt.figure(1).texts[1].new
#% end: automatic generated code from pylustrator
plt.show()
fig2.savefig(filename='fig2_learnability.pdf',dpi=300,format='pdf',bbox_inches='tight')