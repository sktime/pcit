from copy import deepcopy

import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

from pcit.MetaEstimator import MetaEstimator


class compare_methods():
    '''
    Class that combines parametric and non-parametric two-sample tests for univariate random variables.
    Here used to compare prediction residuals of different prediction functionals
    -------------------------
    Attributes:
        - resid_1, resid_2: the residuals of the two prediction functionals
    '''
    def __init__(self, resid_1, resid_2):
        self.resid_1 = resid_1
        self.resid_2 = resid_2
        self.method_better = None
        self.prob = None
        self.loss_means = None
        self.loss_se = None

    def wilcox_onesided(self):
        '''
        One-sided wilcoxon test, adapted from SciPy's "wilcoxon", adjusted to be a one-sided test.
        Returns the p-value of the hypothesis that resid_1 are coming from a distribution that has
        a lower expected generalization loss than resid_2
        '''

        # https://github.com/scipy/scipy/blob/v0.14.0/scipy/stats/morestats.py#L1893
        # Copyright (c) 2001, 2002 Enthought, Inc. All rights reserved.
        # Copyright (c) 2003-2017 SciPy Developers. All rights reserved.
        # Redistribution and use in source and binary forms, with or without
        # modification, are permitted provided that the following conditions are met:
        # a. Redistributions of source code must retain the above copyright notice,
        #    this list of conditions and the following disclaimer.
        # b. Redistributions in binary form must reproduce the above copyright
        #    notice, this list of conditions and the following disclaimer in the
        #    documentation and/or other materials provided with the distribution.
        # c. Neither the name of Enthought nor the names of the SciPy Developers
        #    may be used to endorse or promote products derived from this software
        #    without specific prior written permission.

        # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
        # ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
        # BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
        # OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
        # SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
        # INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
        # CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
        # ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
        # THE POSSIBILITY OF SUCH DAMAGE.

        resid_1, resid_2 = map(np.asarray, (self.resid_1, self.resid_2))

        if len(resid_1) != len(resid_2):
            raise ValueError('Unequal N in wilcoxon.  Aborting.')

        # Calculate differences in losses
        d = resid_1 - resid_2

        d = np.compress(np.not_equal(d, 0), d, axis=-1)

        count = len(d)

        # Calculate sum of ranks of positive and negative differences
        r = stats.rankdata(abs(d))
        r_plus = np.sum((d > 0) * r, axis=0)
        r_minus = np.sum((d < 0) * r, axis=0)

        # Calculate test statistic
        T = min(r_plus, r_minus)
        mn = count * (count + 1.) * 0.25
        se = count * (count + 1.) * (2. * count + 1.)

        replist, repnum = stats.find_repeats(r)
        if repnum.size != 0:
            # Correction for repeated elements.
            se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()

        se = max(np.sqrt(se / 24), 1e-100)
        z = (T - mn) / se
        self.prob = stats.norm.sf(abs(z))

        self.method_better = r_plus < r_minus

        # Exception handling
        if all(resid_1 == resid_2):
            self.prob = 1
            self.method_better = False

        # Make p-value one-sided
        self.prob = (self.method_better == 0) * 1 + self.method_better * self.prob

        return self

    def loss_statistics(self):
        '''
        Calculate the estimated standard error and associated standard deviation for
        the generalization error
        '''

        n = len(self.resid_1)

        # MSE
        self.loss_means = [np.mean(self.resid_1), np.mean(self.resid_2)]

        # Standard errors of MSE
        se_loss_1 = np.sqrt(np.mean(np.power(self.resid_1 - self.loss_means[0], 2)) / (n - 1))
        se_loss_2 = np.sqrt(np.mean(np.power(self.resid_2 - self.loss_means[1], 2)) / (n - 1))

        # RMSE
        self.loss_means = np.sqrt(self.loss_means)

        # Standard errors of RMSE
        self.loss_se = np.divide([se_loss_1, se_loss_2], 2 * self.loss_means)

        return self.loss_means, self.loss_se

    def evaluate(self):
        '''
        General purpose function call to get all relevant loss statistics. Returns an object
        containing all test statistics corresponding to the parametric and non-parametric cases
        '''

        self.wilcox_onesided()
        self.loss_statistics()

        return self

def FDRcontrol(p_values, confidence = None):
    '''
    Routine to calculate the FDR adjusted p-values in a multiple testing scenario.
    ----------------------
    Attributes:
        - p_values: vector or matrix of p-values
        - confidence: desired FDR level

    Returns:
        - p_values_adj: Adjusted p-values
        - which_predictable: same shape as p_values_adj, but hard assignments if null is rejected or not
    '''

    # Adapted from R's 'p.adjust' in the package stats, reference:
    # R Core Team (2017). R: A language and environment for statistical
    # computing. R Foundation for Statistical Computing, Vienna, Austria. URL
    # https://www.R-project.org/.
    #

    # Transform into array
    shape = p_values.shape
    p_values = p_values.flatten()

    p = len(p_values)

    # FDR adjustment (method adapted from R)
    i = np.arange(start = p, stop = 0, step = -1)
    o = np.flip(np.argsort(p_values), axis = 0)
    ro = np.argsort(o)
    q = np.sum(1 / np.arange(1,p))
    p_values_adj = np.minimum(1, np.minimum.accumulate(q * p_values[o] / i * p))[ro]

    # Add "hard" assignments in null is rejected or not (0 or 1)
    which_predictable = None
    if confidence is not None:
        which_predictable = np.arange(0, p)[p_values_adj <= confidence]

    # Restore original shape
    p_values_adj = np.reshape(p_values_adj, newshape = shape)

    return p_values_adj, which_predictable

def get_loss_statistics(regr_loss, baseline_loss, parametric, confidence):
    '''
    Support function for the conditional independence test, calculating loss statistics
    while distinguishing between the parametric and non-parametric case
    ----------------------
    Returns:
        - p_values: p-value for one-sided hypothesis that losses are equal
        - conf_int: Confidence interval for difference between losses
    '''
    if parametric:
        # Paired t-test, test statistic adjusted for one-sided test
        tt_res = stats.ttest_rel(regr_loss, baseline_loss)
        p_value = (tt_res[0] > 0) + np.sign((tt_res[0] < 0) - 0.5) * tt_res[1] / 2

        # Calculate confidence intervals for difference in generalization loss
        loss_stat = compare_methods(regr_loss, baseline_loss).loss_statistics()

        # Mean difference
        diff_mean = loss_stat.loss_means[1] - loss_stat.loss_means[0]

        # SE of difference
        diff_se = np.sqrt(loss_stat.loss_se[0] ** 2 + loss_stat.loss_se[1] ** 2)

        # C.I.
        conf_int = (diff_mean + stats.norm.ppf(confidence) * diff_se, diff_mean,
                            diff_mean + stats.norm.ppf(1 - confidence) * diff_se)

    else:
        loss_stat = compare_methods(regr_loss, baseline_loss).evaluate()
        # p_value for wilcoxon signed rank test
        p_value = loss_stat.prob

        # Mean difference
        diff_mean = loss_stat.loss_means[1] - loss_stat.loss_means[0]

        # SE of difference
        diff_se = np.sqrt(loss_stat.loss_se[0] ** 2 + loss_stat.loss_se[1] ** 2)

        # C.I.
        conf_int = (diff_mean + stats.norm.ppf(confidence) * diff_se, diff_mean,
                            diff_mean + stats.norm.ppf(1 - confidence) * diff_se)

    return p_value, conf_int

def pred_indep(y, x, z = None, estimator = MetaEstimator(), parametric = False, confidence = 0.05, symmetric = True):
    '''
    Conditional independence test using predictive inference to detect if y and x are conditionally
    independent given z
    -------------------
    Attributes:
        - z: Conditioning set (can be empty, in which case the test is a marginal independence test)

        - estimator: object of the MetaEstimator class

        - parametric: determines test for the residuals, True  results in a t-test, False in a wilcoxon

        - confidence: confidence level for test, controls the family-wise error rate

        - symmetric: should the test by symmetric (x and y can be interchanged for some results), or
                        one-sided, where the result says if x adds to prediction of y

    Returns:
        - p_values_adj: adjusted p_values for each variable in y

        - which_predictable: which variable in y can be predicted better using the information in x

        - independent: tuple, first values is 1 if independent, otherwise 0, second value is p_value
                            of statement

        - conf_int_out: confidence interval for difference in prediction error for each y
    '''

    if not hasattr(estimator, 'isMetaEstimator'):
        print('estimator needs to be of type MetaEstimator')
        return

    if not parametric in [True, False]:
        print('parametric has to be either "True" or "False"')
        return

    if (not type(confidence) is float) or (confidence <= 0) or (confidence >= 1):
        print('confidence needs to be between 0 and 1')
        return

    if not symmetric in [True, False]:
        print('symmetric has to be either "True" or "False"')
        return

    # Run it twice to make result symmetric, if needed
    for twice in range(2):
        if z is not None:
            # Create input set, split into train and test
            x_all = np.append(x, z, axis=1)
            x_tn, x_ts, y_tn, y_ts, z_tn, z_ts = train_test_split(x_all, y, z, test_size = 1/3, random_state = 1)

        else:
            # Split into train and test
            x_tn, x_ts, y_tn, y_ts = train_test_split(x, y, test_size = 1/3, random_state = 1)

        # Prepare output arrays
        p_out = y.shape[1]
        p_values = np.ones(p_out)
        conf_int = np.ones(p_out).astype(list)

        # Loop through variables in y
        for i in range(p_out):
            estimator_base = deepcopy(estimator)
            estimator_cont = deepcopy(estimator)

            # Calculate loss residuals

            # Baseline, distinction between marginal and conditional case
            if z is None:
                base_loss = estimator_base.get_residuals(x_tn, x_ts, y_tn[:, i],y_ts[:, i], baseline = True)

            else:
                base_loss = estimator_base.get_residuals(z_tn, z_ts, y_tn[:,i], y_ts[:,i])

            # Loss residuals of prediction including x
            loss = estimator_cont.get_residuals(x_tn, x_ts, y_tn[:,i], y_ts[:,i])

            # p-value of one side test of baseline against prediction including info about x
            p_values[i], conf_int[i] = get_loss_statistics(loss, base_loss, parametric, confidence)


        if symmetric and ('p_values_1' not in locals()):
            # If symmetric test needed, save values, exchange x and y
            y_old = y.copy()
            y = x.copy()
            x = y_old.copy()
            p_values_1 = p_values.copy()
            conf_int_out = conf_int.copy()

        else:
            # If no symmetric result is needed, continue
            conf_int_out = conf_int.copy()
            break

    if symmetric:
        # Apply FDR control on all p_values
        p_values_adj, which_predictable = FDRcontrol(np.append(p_values_1, p_values), confidence)

        # Determine if null-hypothesis is true
        independent = (sum(which_predictable) == 0, np.min(p_values_adj))

        # Distinction between univariate and multivariate y
        if len(p_values_1) > 1:
            p_values_adj, which_predictable = FDRcontrol(p_values_1, confidence)

        else:
            p_values_adj = p_values_1

    else:
        #Distinction between univariate and multivariate
        if len(p_values) > 1:
            # FDR control, and check if null hypothesis is rejected
            p_values_adj, which_predictable = FDRcontrol(p_values, confidence)
            independent = (sum(which_predictable) == 0, np.min(p_values))

        else:
            p_values_adj = p_values
            independent = p_values[0] > confidence

    return p_values_adj, independent, conf_int_out
