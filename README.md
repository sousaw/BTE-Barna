# BTE-Barna
**BTE-Barna** (**B**oltzmann **T**ransport **E**quation - **B**eyond the **R**ta for **NA**nosystems) is a software package that extends almaBTE iterative and Monte Carlo solvers of
the Peierls-Boltzmann transport equation for phonons (PBTE) to work with nanosystems based on 2D materials, within and beyond the RTA.
The package is composed of three PBTE solvers
* _kappa_Tsweep_nanos_: An iterative solver for highly symmetric nanosystems, namely nanoribbons and nanowires, based on suppressed lifetimes due to boundaries. 
 It provides the effective thermal conductivity, RTA and beyond the RTA, for given dimension (width or radius) as a function of the temperature.
* _RTAMC2D_: Energy-based deviation Monte Carlo RTA solver to explore steady-state and the transient to it. It provides the space and time resolved temperature and flux.
Additionally it provides the spectral decomposition of those quantities.
* _beRTAMC2D_: Energy-based deviation Monte Carlo beyond the RTA. It provides the space and time resolved temperature, flux and the deviational energy distribution.


In addition to those solvers we provide _PropagatorBuilder_, to calculate the propagator needed by _beRTAMC2D_ using Krylov subspace
methods to calculate it, and *dist_reader* to process the the deviational energy distribution generated by _beRTAMC2D_ to obtain the spectral decomposition of the
temperature and the heat flux.

![alt text](https://github.com/sousaw/BTE-Barna/blob/main/doc/images/BTE-Barna.jpg)
Fig. 1: General structure of **BTE-Barna** package.

