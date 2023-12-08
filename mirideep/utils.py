import os
import numpy as np
import matplotlib.pylab as plt
from astropy.io import ascii
from astropy.modeling import models, fitting

def fit_wavecorr(module,plot_wavefit=False):

	# Read wavelength correction
	local_path = os.path.join(os.path.dirname(__file__), 'rsrfs')
	refcor_file = open(os.path.join(local_path,'wavecal_wlcorr.csv'), 'rb')
	refcor = ascii.read(refcor_file)

	edges = {'ch1short':(4.9,5.66),
             'ch1medium':(5.66,6.53),
			 'ch1long':(6.53,7.51),
			 'ch2short':(7.51,8.67),
			 'ch2medium':(8.67,10.02),
			 'ch2long':(10.02,11.55),
			 'ch3short':(11.55,13.34),
			 'ch3medium':(13.34,15.41),
			 'ch3long':(15.41,17.70),
			 'ch4short':(17.70,20.69),
			 'ch4medium':(20.69,24.19),
			 'ch4long':(24.19,27.90)}

	gsubs = np.where((refcor['WL'].data>edges[module][0]) & (refcor['WL'].data<edges[module][1]))
	wl = refcor['WL'][gsubs]
	sh = refcor['SHIFT'][gsubs]

	model_poly = models.Polynomial1D(degree=3)
	fitter_poly = fitting.LinearLSQFitter()
	best_fit_poly = fitter_poly(model_poly, wl, sh)

	if plot_wavefit:
		plt.scatter(wl,300000*sh/wl)
		plt.plot(wl,300000*best_fit_poly(wl)/wl)
		plt.show()

	return best_fit_poly

#fit_wavecorr()
