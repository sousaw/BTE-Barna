import numpy as np
import math
import sys
import matplotlib
import copy
import scipy.linalg as la
from scipy.interpolate import CubicSpline
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from scipy.special import erfc
from scipy import signal
import csv


class Etheo:
    def __init__(self,filename):
        t = []
        e = []
        f = open(filename)
        for line in f:
            t.append(float(line.split()[0]))
            e.append(float(line.split()[1]))
        
        t = np.array(t)
        e = np.array(e)
        
        e *= 4.0e+12 
        self.cs = CubicSpline(e, t)
    
    def get(self,x):
        return self.cs(x)


def read_dist_rta(filename):
    
    data =  np.loadtxt(filename,delimiter=',')
    
    omega = data[0,1:]
    time  = data[1:,0]
    dist  = data[1:,1:]

    return time, omega/(2*math.pi), dist
    



def process_file(filename):
    f = open(filename,'r')
    
    energy = []
    time   = []
    r      = None
    for line in f:
        if "*" in line or "This" in line or "#" in line or "[" in line or "npart" in line:
            continue
        el = line.split()
        
        if len(el) is 0:
            continue
        
        if (float(el[1]) > 1200):
            break
        
        time.append(float(el[1]))
        nboxes = int((len(el) - 2)/3) - 4
        n2     = int((nboxes+1.0e-6)/4)
        
        es = []
        
        for i in range(0,nboxes):
           j = i*3 + 3
           es.append(float(el[j]))
        r = np.linspace(1.5, 399.5, nboxes)
        
        energy.append(es)
    
    f.close()
    
    return np.array(time), np.array(energy), r


def get_rta(filename):
    
    data = np.loadtxt(filename,delimiter=',')
    
    d = {}
    
    r = np.linspace(1.5, 399.5, 200)
    temp = data[:,1:201]
    time = data[:,0]
    
    dt = time[1]- time[0]
    
    for i,j in zip(time,temp):
        d[i+dt] = j
    
    return r,d
    
    


"""
It solves the Fourier equation
using laplace transform
"""
def FourierSol(x,t,alpha,L,Thot,Tcold):
    if t < 1.0e-6:
        s0 = np.zeros(shape=(len(x)))
        s0[:] = 0
        s0[0] = 0
        return s0
    
    tol = 1.0e-8
    temp  = copy.deepcopy(x)
    temp *= 0
    
    factor = 0.5/math.sqrt(alpha*t)
    
    otemp  = copy.deepcopy(temp)
    
    i = 0
    
    xp = copy.deepcopy(x)
    xp[:] -= 0.5
    xp[:] *= -1.0
    xp[:] +=  L
    
    N = None
    try:
        N = len(x)
    except:
        N = 1
    while True:
        cn = None
        try :
            cn = len(x)*[(2*i + 1)*L]
            cn = np.array(cn)
        except:
            cn = (2*i + 1)*L
        temp += erfc( factor*(cn - xp)) - erfc(factor*(cn+xp))
        
        if (la.norm(temp-otemp)/N < tol) and i > 4:
            break
        
        otemp = copy.deepcopy(temp)
        
        i += 1
    
    return temp*(Thot-Tcold)+Tcold


FS = 16


rrta, drta = get_rta("rta/temperature_300K_run_0.csv")

nums = [1,2]


boxes0 = [5,6,7,8,9]
boxes1 = [99,100,101,102,103]



omega  = None
total_dist_0 = []
total_dist_1 = []

# We print distribution
for i in nums:
    serie = i
    # Doing boxes0
    time , omega, dist1 = read_dist_rta("berta/deltaT_omega_5.csv")
    time , omega, dist2 = read_dist_rta("berta/deltaT_omega_6.csv")
    time , omega, dist3 = read_dist_rta("berta/deltaT_omega_7.csv")
    time , omega, dist4 = read_dist_rta("berta/deltaT_omega_8.csv")
    time , omega, dist5 = read_dist_rta("berta/deltaT_omega_9.csv")
    
    total_dist_0.append(dist1)
    total_dist_0.append(dist2)
    total_dist_0.append(dist3)
    total_dist_0.append(dist4)
    total_dist_0.append(dist5)
    
    time , omega, dist1 = read_dist_rta("berta/deltaT_omega_99.csv")
    time , omega, dist2 = read_dist_rta("berta/deltaT_omega_100.csv")
    time , omega, dist3 = read_dist_rta("berta/deltaT_omega_101.csv")
    time , omega, dist4 = read_dist_rta("berta/deltaT_omega_102.csv")
    time , omega, dist5 = read_dist_rta("berta/deltaT_omega_103.csv")
    
    total_dist_1.append(dist1)
    total_dist_1.append(dist2)
    total_dist_1.append(dist3)
    total_dist_1.append(dist4)
    total_dist_1.append(dist5)


rtime , romega, rdist1 = read_dist_rta("rta/deltaT_omega_5_300K_run_0.csv")
rtime , romega, rdist2 = read_dist_rta("rta/deltaT_omega_6_300K_run_0.csv")
rtime , romega, rdist3 = read_dist_rta("rta/deltaT_omega_7_300K_run_0.csv")
rtime , romega, rdist4 = read_dist_rta("rta/deltaT_omega_8_300K_run_0.csv")
rtime , romega, rdist5 = read_dist_rta("rta/deltaT_omega_9_300K_run_0.csv")

rdisttot = (rdist1 + rdist2 + rdist3 +  rdist4 + rdist5)/5.0
    
total_dist_0 = np.array(total_dist_0)
total_dist_1 = np.array(total_dist_1)

#Printing box1
b1 = total_dist_0[:,0,:]

dmean = np.mean(b1,axis=0)
dstd  = np.std(b1,axis=0)


rd200 = None
rd550 = None
for i,j in zip(rtime,rdisttot):
    if np.isclose(i-0.125,200.):
        rd200 = j
    if np.isclose(i-0.125,550.):
        rd550 = j

fig, ax1 = plt.subplots()
plt.gcf().subplots_adjust(left=0.15)


ax1.set_zorder(20)
ax1.set_facecolor('none')


ax1.plot(omega,dmean,c="black",label="bRTA")
ax1.plot(romega,rd200,c="red",label="RTA")
plt.title("t = 200 ps",fontsize=FS)
ax1.set_xlabel(r"f [THz]",fontsize=FS)
ax1.set_ylabel(r"$\mathrm{\Delta T}  \mathrm{\left[\frac{K}{rad \cdot THz}\right]}$ ",fontsize=FS)
plt.tick_params(axis='both', which='major', labelsize=FS-2)
plt.tick_params(axis='both', which='minor', labelsize=FS-2)



#print(sum(dmean),sum(rd200))

ax1.legend(fontsize=FS-2)

plt.savefig("d15_200.pdf",bbox_inches='tight')

plt.clf()

b1 = total_dist_0[:,1,:]

dmean = np.mean(b1,axis=0)
dstd  = np.std(b1,axis=0)


fig, ax1 = plt.subplots()
plt.gcf().subplots_adjust(left=0.15)


ax1.set_zorder(20)
ax1.set_facecolor('none')


ax1.plot(omega,dmean,c="black",label="bRTA")
ax1.plot(romega,rd550,c="red",label="RTA")
plt.title("t = 550 ps",fontsize=FS)
ax1.set_xlabel(r"f [THz]",fontsize=FS)
ax1.set_ylabel(r"$\mathrm{\Delta T}  \mathrm{\left[\frac{K}{rad \cdot THz}\right]}$ ",fontsize=FS)
ax1.legend(fontsize=FS)#handles, labels)
plt.tick_params(axis='both', which='major', labelsize=FS-2)
plt.tick_params(axis='both', which='minor', labelsize=FS-2)

plt.savefig("d15_550.pdf",bbox_inches='tight')


plt.clf()

#Doing box2

rtime , romega, rdist1 = read_dist_rta("rta/deltaT_omega_99_300K_run_0.csv")
rtime , romega, rdist2 = read_dist_rta("rta/deltaT_omega_100_300K_run_0.csv")
rtime , romega, rdist3 = read_dist_rta("rta/deltaT_omega_101_300K_run_0.csv")
rtime , romega, rdist4 = read_dist_rta("rta/deltaT_omega_102_300K_run_0.csv")
rtime , romega, rdist5 = read_dist_rta("rta/deltaT_omega_103_300K_run_0.csv")

rdisttot = (rdist1 + rdist2 + rdist3 +  rdist4 + rdist5)/5.0


b1 = total_dist_1[:,0,:]

dmean = np.mean(b1,axis=0)
dstd  = np.std(b1,axis=0)

rd200 = None
rd550 = None
for i,j in zip(rtime,rdisttot):
    if np.isclose(i-0.125,200.):
        rd200 = j
    if np.isclose(i-0.125,550.):
        rd550 = j

fig, ax1 = plt.subplots()
plt.gcf().subplots_adjust(left=0.15)


ax1.set_zorder(20)
ax1.set_facecolor('none')


ax1.plot(omega,dmean,c="black",label="bRTA")
ax1.plot(romega,rd200,c="red",label="RTA")
plt.title("t = 200 ps",fontsize=FS)
ax1.set_xlabel(r"f [THz]",fontsize=FS)
ax1.set_ylabel(r"$\mathrm{\Delta T}  \mathrm{\left[\frac{K}{rad \cdot THz}\right]}$ ",fontsize=FS)
ax1.legend(fontsize=FS-2)#handles, labels)
plt.tick_params(axis='both', which='major', labelsize=FS-2)
plt.tick_params(axis='both', which='minor', labelsize=FS-2)


plt.savefig("d203_200.pdf",bbox_inches='tight')
plt.clf()


b1 = total_dist_1[:,1,:]

dmean = np.mean(b1,axis=0)
dstd  = np.std(b1,axis=0)



fig, ax1 = plt.subplots()
plt.gcf().subplots_adjust(left=0.15)

ax1.set_zorder(20)
ax1.set_facecolor('none')


ax1.plot(omega,dmean,c="black",label="bRTA")
ax1.plot(romega,rd550,c="red",label="RTA")
plt.title("t = 550 ps",fontsize=FS)
ax1.set_xlabel(r"f [THz]",fontsize=FS)
ax1.set_ylabel(r"$\mathrm{\Delta T}  \mathrm{\left[\frac{K}{rad \cdot THz}\right]}$ ",fontsize=FS)
ax1.legend(fontsize=FS)#handles, labels)
plt.tick_params(axis='both', which='major', labelsize=FS-2)
plt.tick_params(axis='both', which='minor', labelsize=FS-2)
plt.savefig("d203_550.pdf",bbox_inches='tight')


plt.clf()



t = None
energies = {}
r  = None

for i in nums:
    serie = int((i-1)/5) + 1
    t , e_, r = process_file("berta/hb_"+str(i)+".log")
    energies[i] = copy.deepcopy(e_)
    
vbeam = 1 * 1 * 0.5 * 1.0e-27


    
L = 400.0
Tcold = 300
Thot = 302



kappa =   22.0869 # W/(m*K)
Cvolumetric = 452553 # J/(m**3 * K)
alpha = kappa/Cvolumetric * 1.0e+6 # nm**2/ps


x = np.linspace(0.5,400.5,801)

time = np.linspace(0.0,2000,4000) 



FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')

writer = FFMpegWriter(fps=10, metadata=metadata)
fig = plt.figure()

ef1 = FourierSol(x,t[1],alpha,L,Thot,Tcold)
xp1 = copy.deepcopy(x)
xp1[:] -= 0.5

plt.clf()

with writer.saving(fig, "comparison.mp4", 100):
    for j in range(len(t)):
        
        i = 20*j
        
        print(i,"/",len(t),t[i])
        if (i > 2200):
           break
        if (i !=800 and i != 2200):
           continue
        mark_ = None
        name_ = "( t = {:6.2f} ps)".format(t[i])
        if i == 800:
           mark_ = '-'
        else:
           mark_ = '--'

        efourier = FourierSol(x,t[i],alpha,L,Thot,Tcold)
        
        xp = copy.deepcopy(x)
        rp = copy.deepcopy(r)
        
        xp[:] -= 0.5
        rp[:] -= 0.5
        
        
        rrtap = copy.deepcopy(rrta)
        
        plt.gcf().subplots_adjust(bottom=0.15)
        
        EE = np.zeros(shape=(len(nums),len(r)),dtype='float64')
        
        inum = 0
        for ii in nums:
            EE[inum,:] = energies[ii][i,:]
            inum += 1
        
        plt.ylim(299.5,302.5)
        
        t_mean = np.mean(EE,axis=0)/(4.0*452553) + 300
        t_std  = np.std(EE,axis=0)/(4.0*452553)

        plt.errorbar(rp,t_mean,t_std ,markersize=5, capsize=5,label="bRTA "+name_,errorevery=20,c="black",ls=mark_) 
        plt.plot(xp,efourier,label="Fourier "+name_, c="blue",ls=mark_)#marker='|',markevery=20)
        if t[i]+0.125 in drta:
            plt.plot(rrtap,drta[t[i]+0.125],label="RTA "+name_,c="green",ls=mark_)#,marker='x',markevery=20)
        plt.ylabel(r"T [K]",fontsize=FS)
        plt.xlabel(r"r [nm]",fontsize=FS)
        plt.legend(loc=1)
        #plt.savefig("./frame_"+str(i)+".pdf",bbox_inches='tight')
        writer.grab_frame()
        if np.isclose(t[i],1000.) is True:
            break

plt.tick_params(axis='both', which='major', labelsize=FS-2)
plt.tick_params(axis='both', which='minor', labelsize=FS-2)
plt.savefig("profile.pdf",bbox_inches='tight')



