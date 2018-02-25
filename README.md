# local_tvc_rw_test
Estimate a time-varying coefficient model to test implications of the random walk hypothesis.

INTRODUCTION:

The goal of this project is to test for autocorrelations in stock returns, which persist for economically meaningful short-term timespans but are not, on average, significantly different from zero in the long term. A rejection of the null hypothesis under such tests would establish evidence against the random walk hypothesis; that is, that past returns are not independent of future returns in some short-run contexts. Such a finding would have clear implications for trading strategy.

The above hypothesis will be tested using a time-varying coefficient model, estimated using a local linear estimation method such as that of Ang and Kristensen (2011). Bootstrap inference methods presented by Chen, Gao, Li, and Silvapulle (2018); McKinnon (2007); and Lee and Ullah (2001) will be applied to build confidence intervals for coefficients on lagged returns. These procedures will also be applied to hypothesis tests for model specification. This will allow for inference on whether or not autoregressive coefficients are significantly different from zero at certain values of t. It will be of particular interest whether this relationship holds significantly during periods classified ex post as "asset pricing bubbles."

DATA:

The datasets used in this project were obtained from Ken French's website, http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html. They are, respectively, '12_Industry_Portfolios_Daily,' 'F-F_Momensum_Factor_daily,' and 'F-F_Reserach_Data_Factors_daily.' Please see the file clean_data.py for brief description of each dataset.


REFERENCES:

• Ang, Andrew, and Kristensen, Dennis, Testing Conditional Factor Models, National Bureau of Eco- nomic Research, Working Paper 17561, 2011, http:/www.nber.orgpapersw17561.pdf
• Xiangjin B. Chen, Jiti Gao, Degui Li and Param Silvapulle (2018) Nonparametric Estimation and Forecasting for Time-Varying Coefficient Realized Volatility Models, Journal of Business ans Economic Statistics, 36:1, 88-100, DOI: 10.1080/07350015.2016.1138118
• MacKinnon, James G; Bootstrap Hypothesis Testing, Department of Economics, Queen’s University, 2007, http:/qed.econ.queensu.caworking paperspapersqed wp 1127.pdf
• Lee, Tae-Hwy, and Ullah, Aman, Nonparametric Bootstrap Specification Testing in Economic Models, 2001, http:/www.faculty.ucr.edutaeleepapergiles.pdf
