import sys
import numpy as np
import matplotlib.pyplot as plt

def load300(filename):
    data = np.loadtxt(filename,delimiter=',',comments='T')
    return data[1,:]


NR_axis = "1,0,0   0,1,0".split()
Ls      = "4.00000 10.0000 40.0000 100.000 400.000 1000.00 4000.00".split()

tf = 0.5/0.533 

fig, axes = plt.subplots()

plt.plot([4.,4000],2*[4 * 5.53348 * tf ],label=r"$\kappa_\mathrm{bulk,AC}^\mathrm{RTA}$",ls=':',c="red")
plt.plot([4.,4000],2*[4 * 7.32893 * tf ],label=r"$\kappa_\mathrm{bulk,AC}^\mathrm{beyond~RTA}$",ls='-.',c="red")
plt.plot([4.,4000],2*[4 * 15.3934 * tf ],label=r"$\kappa_\mathrm{bulk,ZZ}^\mathrm{RTA}$",ls=':',c="black")
plt.plot([4.,4000],2*[4 * 22.0869 * tf ],label=r"$\kappa_\mathrm{bulk,ZZ}^\mathrm{beyond~RTA}$",ls='-.',c="black")


for axis in NR_axis:
    
    widths = np.zeros(len(Ls))
    keff_rta  = np.zeros(len(Ls))
    keff_berta = np.zeros(len(Ls))
    
    axname = "AC"
    
    if "0,1,0" in axis:
        axname = "ZZ"
    
    ii = 0
    for L in Ls:
        filename = "phosphorene_50_50_1_nanoribbon_L_"+L+"_"+axis+"_300_301.Tsweep"
        parent_dir = "./kappas/"
        data = load300(parent_dir+filename)
        widths[ii] = float(L) 
        keff_rta[ii] = 4 * data[1]
        keff_berta[ii] = 4 * data[2]
        ii += 1
    
    ms_ = 3.5 
    
    
    datamatrix = np.zeros(shape=(len(Ls),5),dtype="float64")
    datamatrix[:,0] = widths
    datamatrix[:,1] = keff_rta   
    datamatrix[:,2] = keff_rta   * -0.2e+9
    datamatrix[:,3] = keff_berta 
    datamatrix[:,4] = keff_berta * -0.2e+9
    
    
    
    
    if axname is "AC":
        plt.plot(widths,tf*keff_rta,label=r"$\kappa_\mathrm{eff,AC}^\mathrm{RTA}$",marker='o',ms=ms_,c="red")
        plt.plot(widths,tf*keff_berta,label=r"$\mathrm{\kappa_\mathrm{eff,AC}^\mathrm{beyond~RTA}}$",marker='o',ms=ms_,c="red",ls='--')
        np.savetxt("./AC/k_eff.AC.csv",datamatrix,delimiter=',',header="#W,krta,Jrta,kberta,Jberta")
        
        
    else:
        plt.plot(widths,tf*keff_rta,label=r"$\kappa_\mathrm{eff,ZZ}^\mathrm{RTA}$",marker='o',ms=ms_,c="black")
        plt.plot(widths,tf*keff_berta,label=r"$\kappa_\mathrm{eff,AC}^\mathrm{beyond~RTA}$",marker='o',ms=ms_,c="black",ls='--')
        np.savetxt("./ZZ/k_eff.ZZ.csv",datamatrix,delimiter=',',header="#W,krta,Jrta,kberta,Jberta")


FS = 16
plt.legend(loc=(1.05, 0.15), fancybox=True, shadow=True, fontsize=FS-2)
plt.ylabel(r"$\mathrm{\kappa_\mathrm{eff} \mathrm{\left[\frac{W}{K \cdot m}\right]}}$",fontsize=FS)
plt.xlabel(r"W [nm]",fontsize=FS)
plt.xscale('log')
axes.tick_params(axis='both', which='major', labelsize=FS-2)
axes.tick_params(axis='both', which='minor', labelsize=FS-2)
fig.tight_layout()
plt.savefig("keff.pdf",bbox_inches='tight')
plt.show()
