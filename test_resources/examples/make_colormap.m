%Declare some variables

Nspace = 299;
Nbins = 200;
spacegrid = linspace(1,299,Nspace);
spaceunit = 'nm';
omegagrid = zeros(Nbins);
flux = zeros(Nbins,Nspace);
flux_plot = zeros(Nbins,Nspace);

%Read csv files

for nsurf=1:Nspace

  file = strcat('./examples_output/steady_montecarlo1d/example3/spectralflux_surface_',num2str(nsurf),'_300K.csv');
  disp(['Processing ' file]);
  fflush(stdout);
  data = csvread(file);
  flux(:,nsurf) = flipud(data(2:end,2));

  if(nsurf==1)
    omegagrid = flipud(data(2:end,1));
  endif

endfor

%Obtain color tables

figure;
imagesc(spacegrid,omegagrid,zeros(Nbins,Nspace));
colormap(jet)
colordata_buffer = colormap;
colordata_plus = colordata_buffer(1:56,1:3);
colormap(copper)
colordata_buffer = flipud(colormap);
colordata_minus = colordata_buffer(1:56,1:3);
close;

%Determine largest positive and negative heat fluxes

qmax = max(max(flux));
qmin = min(min(flux));
disp(['qmax = ' num2str(qmax)]);
disp(['qmin = ' num2str(qmin)]);

%Make rescaled versions of positive and negative data

for nrow=1:size(flux,1)
 for ncol=1:size(flux,2)
  if(flux(nrow,ncol)>=0) flux_plot(nrow,ncol) = flux(nrow,ncol)/qmax; endif
  if(flux(nrow,ncol)<0) flux_plot(nrow,ncol) = -abs(flux(nrow,ncol))/abs(qmin); endif
 endfor
endfor

% Make plot

imagesc(spacegrid,omegagrid,flux_plot);
set(gca,'Ydir','normal');
caxis([-1 1]);
colormap([colordata_minus;[0 0 0];colordata_plus]);
colorbar;
set(gca,'Fontsize',16);
xlabel(strcat('position [',spaceunit,']'));
ylabel('phonon frequency [rad/ps]');
saveas(gca,'colorplot.png');