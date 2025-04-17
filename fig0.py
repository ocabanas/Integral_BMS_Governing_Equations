import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
from copy import deepcopy
import pynumdiff
import matplotlib.gridspec as gridspec
import pickle
from matplotlib.ticker import ScalarFormatter,StrMethodFormatter
import pylustrator
pylustrator.start()
try:
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/Arial.ttf')
    prop = font_manager.FontProperties(fname='/usr/share/fonts/truetype/msttcorefonts/Arial.ttf')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = prop.get_name()
except Exception as e:
    print('Could not load Arial font')
    print(e)
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['font.size'] = 10

colors={
    'data': 'grey',
    'gt':'k',
    'fd':'#7fc97f',
    'sd':'#fdc086',
    'I-BMS':'#64B5F6',
    'FD-BMS':'#1976D2',
    'SD-BMS':'#0D47A1',
    'WS':'#fed976',
    'IS':'orangered'
#    'A-BMS':
#    'Log':
#    'Gom':
}

data=pd.read_pickle('./Logistic/noise_data/0.005_0.pkl')

t=data.t.to_numpy()
y=data.B.values

B=data.B.values
h=data.t.to_numpy()[1]-data.t.to_numpy()[0]
gradiend=np.array([(B[1]-B[0])/h]+[(B[i+1]-B[i-1])/(2*h) for i in range(1,len(B)-1)]+[(B[len(B)-1]-B[len(B)-2])/h])
#Smooth derivate
par = [2,21,21]
h=data.t.to_numpy()[1]-data.t.to_numpy()[0]
x_hat, dxdt_hat = pynumdiff.linear_model.polydiff(data.B.to_numpy(), h, par, options=None)
smooth=dxdt_hat
# true logistic gideline
def d_logistic(x):
    #return a*x*(1.-b*x)
    return x*(1.-x)
x_values=np.linspace(0,1,20)
d_values=d_logistic(x_values)

def logistic(x):
    #return a*x*(1.-b*x)
    return 1./(1.+np.exp(6.-x))
ground_truth=logistic(t+6.)

#fig0, axs0 = plt.subplots(1, 2, sharey=True,figsize=(8,5),gridspec_kw=gs_logitic_low)
# Remove vertical space between Axes
fig, ax = plt.subplots(3,4)
# Plot each graph, and manually set the y tick values
ax[0,0].plot(t+6., y,'.-',c=colors['data'],label='x(t)',lw=.7,ms=3.)
ax[0,0].plot(t+6., ground_truth,colors['gt'],label='ground truth',lw=.7)
ax[0,0].set_ylabel('x')
ax[0,0].set_xlabel('t')
ax[0,0].tick_params(axis='both', which='major')

#axs0.legend(loc='best',frameon=False,fontsize=10)

ax[0,1].scatter(gradiend, y, marker='.',c=colors['fd'],label='$\dot x$ finite diff.',alpha=1)
ax[0,1].scatter(smooth, y,marker='.',c=colors['sd'],label='$\dot x$smooth',alpha=1)
ax[0,1].plot(d_values,x_values,c=colors['gt'],label='ground truth',lw=.7)
ax[0,1].set_xlabel(r'$\dot{x}$')
ax[0,1].set_ylabel(None)
#axs0[1].legend(loc='best',frameon=False,fontsize=10)
ax[0,1].tick_params(axis='both', which='major')
ax[0,1].yaxis.set_tick_params(labelleft=False)
##################################################################
# Logistic high noise
data=pd.read_pickle('Logistic/noise_data/0.1_0.pkl')

t=data.t.to_numpy()
y=data.B.values

B=data.B.values
h=data.t.to_numpy()[1]-data.t.to_numpy()[0]
gradiend=np.array([(B[1]-B[0])/h]+[(B[i+1]-B[i-1])/(2*h) for i in range(1,len(B)-1)]+[(B[len(B)-1]-B[len(B)-2])/h])
#Smooth derivate
par = [2,21,21]
h=data.t.to_numpy()[1]-data.t.to_numpy()[0]
x_hat, dxdt_hat = pynumdiff.linear_model.polydiff(data.B.to_numpy(), h, par, options=None)
smooth=dxdt_hat
# true logistic gideline
def d_logistic(x):
    #return a*x*(1.-b*x)
    return x*(1.-x)
x_values=np.linspace(0,1,20)
d_values=d_logistic(x_values)


# Plot each graph, and manually set the y tick values
ax[0,2].plot(t+6., y,'.-',c=colors['data'],ls='-',label='x(t)',lw=.7,ms=3.)
ax[0,2].plot(t+6., ground_truth,c=colors['gt'],label='ground truth',lw=.7)
ax[0,2].set_xlabel('t')
ax[0,2].tick_params(axis='both', which='major')
#axs0.legend(loc='best',frameon=False,fontsize=10)


ax[0,3].scatter(gradiend, y, marker='.',c=colors['fd'],label='$\dot x$ finite diff.',alpha=1)
ax[0,3].scatter(smooth, y,marker='.',c=colors['sd'],label='$\dot x$smooth',alpha=1)
ax[0,3].plot(d_values,x_values,color=colors['gt'],label='ground truth',lw=.7)
ax[0,3].set_xlabel(r'$\dot{x}$')
#axs1[1].legend(loc='best',frameon=False,fontsize=10)
ax[0,3].set_xlim([-0.5,0.5])
ax[0,3].tick_params(axis='both', which='major')
ax[0,3].yaxis.set_tick_params(labelleft=False)

#####################################################
# Lotka-Low noise
data=pd.read_csv('./Lotka-Volterra/noise_data/0.1_0.csv')

t=data.t.to_numpy()
x=data.x.values
y=data.y.values

h=t[1]-t[0]
x_grad=np.array([(x[1]-x[0])/h]+[(x[i+1]-x[i-1])/(2*h) for i in range(1,len(x)-1)]+[(x[len(x)-1]-x[len(x)-2])/h])
y_grad=np.array([(y[1]-y[0])/h]+[(y[i+1]-y[i-1])/(2*h) for i in range(1,len(y)-1)]+[(y[len(x)-1]-y[len(y)-2])/h])

x_sm = data.dx.to_numpy()
y_sm = data.dy.to_numpy()

# true lotka gideline
import scipy

def dx_LV(x,y):
    return x*(0.1-0.02*y)
def dy_LV(x,y):
    #return a*x*(1.-b*x)
    return y*(0.02*x-0.4)
h=0.5
time= np.arange(0,80,h)
#sol=Euler(ode,y0=[10.,5.], h=h ,t_eval=t)
#print(sol)
def ode(t,y):
    return [dx_LV(*y),dy_LV(*y)]
sol=scipy.integrate.solve_ivp(ode, [time[0],time[-1]], y0=[10.,5.], method='RK45', t_eval=time)
print(sol)
t_x=sol.y[0]
t_y=sol.y[1]

true_dx=dx_LV(t_x,t_y)
true_dy=dy_LV(t_x,t_y)

# Plot each graph, and manually set the y tick values
ax[1,0].plot(time, x,'.-',c=colors['data'],label='x(t)',lw=.7,ms=3.)
ax[1,0].plot(time, t_x,c=colors['gt'],label='ground truth',lw=.7)
ax[1,0].set_ylabel('x')
ax[1,0].set_xlabel('t')
ax[1,0].tick_params(axis='both', which='major')
#axs0.legend(loc='best',frameon=False,fontsize=10)



# Plot each graph, and manually set the y tick values
ax[2,0].plot(time, y,'.-',c=colors['data'],label='y(t)',lw=.7,ms=3.)
ax[2,0].plot(time, t_y,c=colors['gt'],label='ground truth',lw=.7)
ax[2,0].set_ylabel('y')
ax[2,0].set_xlabel('t')
ax[2,0].tick_params(axis='both', which='major')
#axs0.legend(loc='best',frameon=False,fontsize=10)




ax[1,1].scatter(x_grad, x, marker='.',c=colors['fd'],label=r'$\dot{x}$ finite diff.',alpha=1)
ax[1,1].scatter(x_sm, x,marker='.',c=colors['sd'],label='$\dot{x}$ smooth',alpha=1)
ax[1,1].plot(true_dx, t_x,c=colors['gt'],label='ground truth',lw=.7)
ax[1,1].set_xlabel(r'$\dot{x}$')
ax[1,1].tick_params(axis='both', which='major')
ax[1,1].yaxis.set_tick_params(labelleft=False)


ax[2,1].scatter(y_grad, y, marker='.',c=colors['fd'],label='$\dot{y}$ finite diff.',alpha=1)
ax[2,1].scatter(y_sm, y,marker='.',c=colors['sd'],label='$\dot{y}$ smooth',alpha=1)
ax[2,1].plot(true_dy, t_y,c=colors['gt'],label='ground truth',lw=.7)
ax[2,1].set_xlabel(r'$\dot{y}$')
#leg=axs2[1].legend(loc='best',frameon=True,fontsize=10,framealpha=0.8,facecolor='white',markerfirst=False)
#leg.get_frame().set_edgecolor('b')
#leg.get_frame().set_linewidth(0.0)
ax[2,1].tick_params(axis='both', which='major')
ax[2,1].yaxis.set_tick_params(labelleft=False)

############################################################################
# Lotka High noise

data=pd.read_csv('./Lotka-Volterra/noise_data/2.0_0.csv')

t=data.t.to_numpy()
x=data.x.values
y=data.y.values

h=t[1]-t[0]
x_grad=np.array([(x[1]-x[0])/h]+[(x[i+1]-x[i-1])/(2*h) for i in range(1,len(x)-1)]+[(x[len(x)-1]-x[len(x)-2])/h])
y_grad=np.array([(y[1]-y[0])/h]+[(y[i+1]-y[i-1])/(2*h) for i in range(1,len(y)-1)]+[(y[len(x)-1]-y[len(y)-2])/h])

x_sm = data.dx.to_numpy()
y_sm = data.dy.to_numpy()



# Plot each graph, and manually set the y tick values
ax[1,2].plot(time, x,'.-',c=colors['data'],label='x(t)',lw=.7,ms=3.)
ax[1,2].plot(time, t_x,c=colors['gt'],label='ground truth',lw=.7)
ax[1,2].set_ylabel('x')
ax[1,2].set_xlabel('t')
ax[1,2].tick_params(axis='both', which='major')
#axs0.legend(loc='best',frameon=False,fontsize=10)



# Plot each graph, and manually set the y tick values
ax[2,2].plot(time, y,'.-',c=colors['data'],label='y(t)',lw=.7,ms=3.)
ax[2,2].plot(time, t_y,c=colors['gt'],label='ground truth',lw=.7)
ax[2,2].set_ylabel('y')
ax[2,2].set_xlabel('t')
ax[2,2].tick_params(axis='both', which='major')
#axs0.legend(loc='best',frameon=False,fontsize=10)
handles,labels = ax[2,2].get_legend_handles_labels()



ax[1,3].scatter(x_grad, x, marker='.',c=colors['fd'],label=r'$\dot{x}$ finite diff.',alpha=1)
ax[1,3].scatter(x_sm, x,marker='.',c=colors['sd'],label='$\dot{x}$ smooth',alpha=1)
ax[1,3].plot(true_dx, t_x,c=colors['gt'],label='ground truth',lw=.7)
ax[1,3].set_xlabel(r'$\dot{x}$')
ax[1,3].tick_params(axis='both', which='major')
ax[1,3].yaxis.set_tick_params(labelleft=False)
a,b = ax[1,3].get_legend_handles_labels()
handles+=a
labels+=b

ax[2,3].scatter(y_grad, y, marker='.',c=colors['fd'],label='$\dot{y}$ finite diff.',alpha=1)
ax[2,3].scatter(y_sm, y,marker='.',c=colors['sd'],label='$\dot{y}$ smooth',alpha=1)
ax[2,3].plot(true_dy, t_y,c=colors['gt'],label='ground truth',lw=.7)
ax[2,3].set_xlabel(r'$\dot{y}$')
#leg=axs2[1].legend(loc='best',frameon=True,fontsize=10,framealpha=0.8,facecolor='white',markerfirst=False)
#leg.get_frame().set_edgecolor('b')
#leg.get_frame().set_linewidth(0.0)
ax[2,3].tick_params(axis='both', which='major')
ax[2,3].yaxis.set_tick_params(labelleft=False)
labels = ['x(t),y(t)', 'ground truth', '$\\dot{x}$,$\\dot{y}$ finite diff.', '$\\dot{x}$,$\\dot{y}$ smooth']
print(labels)
print(handles)
plt.legend(handles[:-1], labels, frameon=False, ncol=len(labels),loc=(-3.461, -0.7455))




#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).set_size_inches(18.290000/2.54, 12.190000/2.54, forward=True)
plt.figure(1).axes[0].set(position=[0.1458, 0.6867, 0.1684, 0.1979])
plt.figure(1).axes[0].text(0.05, 0.85, 'B', transform=plt.figure(1).axes[0].transAxes, fontsize=14., weight='bold')  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[1].set(position=[0.3155, 0.6866, 0.1684, 0.1979], yticks=[])
plt.figure(1).axes[2].set(position=[0.5518, 0.6866, 0.1684, 0.1979])
plt.figure(1).axes[2].text(0.05, 0.85, 'C', transform=plt.figure(1).axes[2].transAxes, fontsize=14., weight='bold')  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[3].set(position=[0.7215, 0.6867, 0.1684, 0.1979], yticks=[])
plt.figure(1).axes[4].set(position=[0.1475, 0.3864, 0.1684, 0.1979], xlabel='')
plt.figure(1).axes[4].text(0.05, 0.85, 'D', transform=plt.figure(1).axes[4].transAxes, fontsize=14., weight='bold')  # id=plt.figure(1).axes[4].texts[0].new
plt.figure(1).axes[4].get_xaxis().get_label().set(text='')
plt.figure(1).axes[5].set(position=[0.3155, 0.3864, 0.1684, 0.1979], xlabel='', yticks=[])
plt.figure(1).axes[5].get_xaxis().get_label().set(text='')
plt.figure(1).axes[6].set(position=[0.5518, 0.3864, 0.1684, 0.1979], xlabel='')
plt.figure(1).axes[6].text(0.05, 0.85, 'E', transform=plt.figure(1).axes[6].transAxes, fontsize=14., weight='bold')  # id=plt.figure(1).axes[6].texts[0].new
plt.figure(1).axes[6].get_xaxis().get_label().set(text='')
plt.figure(1).axes[6].get_yaxis().get_label().set(text='')
plt.figure(1).axes[7].set(position=[0.7215, 0.3864, 0.1684, 0.1979], xlabel='', yticks=[])
plt.figure(1).axes[7].get_xaxis().get_label().set(text='')
plt.figure(1).axes[8].set(position=[0.1475, 0.1329, 0.1684, 0.1979])
plt.figure(1).axes[8].text(0.05, 0.85, 'F', transform=plt.figure(1).axes[8].transAxes, fontsize=14., weight='bold')  # id=plt.figure(1).axes[8].texts[0].new
plt.figure(1).axes[9].set(position=[0.3155, 0.1329, 0.1684, 0.1979], yticks=[])
plt.figure(1).axes[10].set(position=[0.5519, 0.1329, 0.1684, 0.1979], ylabel='')
plt.figure(1).axes[10].text(0.05, 0.85, 'G', transform=plt.figure(1).axes[10].transAxes, fontsize=14., weight='bold')  # id=plt.figure(1).axes[10].texts[0].new
plt.figure(1).axes[10].get_yaxis().get_label().set(text='')
plt.figure(1).axes[11].set(position=[0.7215, 0.1329, 0.1684, 0.1979])
plt.figure(1).text(0.0441, 0.7265, 'Logistic', transform=plt.figure(1).transFigure, fontsize=11., weight='bold', fontname='Arial', rotation=90.)  # id=plt.figure(1).texts[0].new
plt.figure(1).text(0.0458, 0.2554, 'Lotka-Volterra', transform=plt.figure(1).transFigure, fontsize=11., weight='bold', fontname='Arial', rotation=90.)  # id=plt.figure(1).texts[1].new
plt.figure(1).text(0.6373, 0.9228, 'High noise regime', transform=plt.figure(1).transFigure, )  # id=plt.figure(1).texts[2].new
plt.figure(1).text(0.2371, 0.9228, 'Low noise regime', transform=plt.figure(1).transFigure, )  # id=plt.figure(1).texts[3].new
#% end: automatic generated code from pylustrator
plt.show()
fig.savefig(filename='fig0_data_panels.pdf',dpi=300,format='pdf',bbox_inches='tight')