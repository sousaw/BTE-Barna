import numpy as np
import matplotlib.pyplot as plt
import copy 
#from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

data_s = []
data_jy_s = []

for i in range(0,10):
    data_s.append(np.loadtxt("temperature_300K_run_"+str(i)+".csv",delimiter=","))
    data_jy_s.append(np.loadtxt("jy_300K_run_"+str(i)+".csv",delimiter=","))

data_s = np.array(data_s)
data_jy_s = np.array(data_jy_s)

data = np.mean(data_s,axis=0)
data_jy = np.mean(data_jy_s,axis=0)


nboxes = 400

dy = 2.5

x = np.linspace(0.,nboxes*dy,400)

T = data[-1,1:nboxes+1]
j = data_jy[-1,1:nboxes+1]

I = copy.deepcopy(j)

I[0:nboxes//2] *= 0.335 * 1.0e-9
I[nboxes//2:nboxes] *= 1.001 * 1.0e-9

maxI = max(I)
minI = min(I)

FS = 16

fig, axs = plt.subplots(nrows=3, ncols=1,sharex=True)

plt.gcf().subplots_adjust(left = 0.20)

axs[0].axvline(x=500,c = "black")
axs[0].plot(x,T)
axs[0].set_ylabel(r"T [K]",fontsize=FS)
print(T[200]-T[199])

axs[0].tick_params(axis='both', which='major', labelsize=FS-2)
axs[0].tick_params(axis='both', which='minor', labelsize=FS-2)
axs[0].set_ylim(298,302)

#ins = inset_axes(axs[0], 
                    #width="20%", # width = 30% of parent_bbox
                    #height="30%", # height : 1 inch
                    #loc=1)
#x_ = []
#T_ = []

#for ii,jj in zip(x,T):
    #if (ii < 510 and ii > 490):
        #x_.append(ii)
        #T_.append(jj)
#ins.plot(x_,T_)

axins = axs[0].inset_axes([0.75, 0.50, 0.2, 0.4])
axins.set_xlim(495,505)
axins.set_ylim(299.45,299.7)
axins.plot(x,T)

axins.tick_params(axis='both', which='major', labelsize=FS-2)
axins.tick_params(axis='both', which='minor', labelsize=FS-2)

axs[1].axvline(x=500,c = "black")
axs[1].plot(x,j)




ymin = 0.9*min(np.abs(j))
ymax = 1.1*max(np.abs(j))
nbdiv = 4
xTicks = []
xticklabels = []
xx = np.linspace(ymin,ymax,nbdiv)
for i in range(0,nbdiv):
  xTicks.append(xx[i])
  printstr = '{:.2e}'.format(xx[i])
  ls = printstr.split('e')
  xticklabels.append((ls[0]+r' $\times 10^{'  + str(int(ls[1])) + '}$'))

axs[1].set_yticks(xTicks)
axs[1].set_yticklabels(xticklabels)

axs[1].set_ylabel(r"J $\mathrm{\left[\frac{J}{s \cdot m^{2}}\right]}$",fontsize=FS)

axs[1].tick_params(axis='both', which='major', labelsize=FS-2)
axs[1].tick_params(axis='both', which='minor', labelsize=FS-2)



axs[2].axvline(x=500,c = "black")
axs[2].plot(x,I)
axs[2].set_ylim(0.95*minI,1.05*maxI)
axs[2].set_ylabel(r"I/L $\mathrm{\left[\frac{J}{s\cdot m}\right]}$",fontsize=FS)
axs[2].set_xlabel(r"y [nm]",fontsize=FS)
axs[2].text(25,0.787,"Graphene",fontsize=FS)
axs[2].text(510,0.787,"hBN-encapsulated\nGraphene",fontsize=FS-2)
axs[2].tick_params(axis='both', which='major', labelsize=FS-2)
axs[2].tick_params(axis='both', which='minor', labelsize=FS-2)


plt.xlim(min(x),max(x))

#plt.s

plt.savefig("interface.pdf",bbox_inches='tight')
