# mtad-tf
Implementation of MTAD-TF: Multivariate Time Series Anomaly Detection Using the Combination of Temporal Pattern and Feature Pattern

this is a draft commit, someone maybe interested

steps

- prepare data, train and test. test data adds anomaly labels
- train model. model for each file, 28 of them for SMD. I am just focusing on server anomaly, but MSL and SMAP should be used as well
- run predict with RMS_loss option to generate loss file. high loss indicates anomaly. 

I will post another repository for EVT POT extreme value theory peak over threshold parameter estimator and threshold predictor for anomaly flagging. So use RMS_loss as input to log likelihood model of GPD generalized pareto distribution to estimate sigma and gamma

