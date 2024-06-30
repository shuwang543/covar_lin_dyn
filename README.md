fit_lindyn.m is the core code for fitting a matrix A and steady-steate x_ss to a time-series of means and covariances.

guess_fit.m, noise_fit.m, nonlin_fit.m, and test_uneven.m are used to perform multiple fits on random simulated dynamical systems, with guess_fit testing different initialization strategies and sample sizes, noise_fit additionally testing different scales of noise, nonlin_fit testing non-linear dynamical systems, and test_uneven testing whether time-series data with uneven time intervals can still be fit uniquely with three timepoints in the absence of noise.
