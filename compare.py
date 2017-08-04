import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from combine import MetaEstimator

class compare_methods():
    def __init__(self, resid_1, resid_2):
        self.resid_1 = resid_1
        self.resid_2 = resid_2
        self.method_better = None
        self.prob = None
        self.loss_means = None
        self.loss_se = None

    def wilcox_onesided(self):
        # Adapted from SciPy's "wilcoxon", adjusted to be a one-sided test, and to
        # return if x is bigger than y

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

        # Test the residuals of a prediction method x against the baseline y

        resid_1, resid_2 = map(np.asarray, (self.resid_1, self.resid_2))

        if len(resid_1) != len(resid_2):
            raise ValueError('Unequal N in wilcoxon.  Aborting.')

        d = resid_1 - resid_2

        d = np.compress(np.not_equal(d, 0), d, axis=-1)

        count = len(d)

        r = stats.rankdata(abs(d))
        r_plus = np.sum((d > 0) * r, axis=0)
        r_minus = np.sum((d < 0) * r, axis=0)

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

        if all(resid_1 == resid_2):
            self.prob = 1

            self.method_better = False

        self.prob = (self.method_better == 0) * 1 + self.method_better * self.prob

        return self

    def loss_statistics(self):
        n = len(self.resid_1)

        self.loss_means = [np.mean(self.resid_1), np.mean(self.resid_2)]

        se_loss_1 = np.sqrt(np.mean(np.power(self.resid_1 - self.loss_means[0], 2)) / (n - 1))
        se_loss_2 = np.sqrt(np.mean(np.power(self.resid_2 - self.loss_means[1], 2)) / (n - 1))

        self.loss_means = np.sqrt(self.loss_means)

        self.loss_se = np.divide([se_loss_1, se_loss_2], 2 * self.loss_means)

        return self.loss_means, self.loss_se

    def evaluate(self):
        self.wilcox_onesided()
        self.loss_statistics()

        return self

def FDRcontrol(p_values, confidence = None):
    # Adapted from R's 'p.adjust' in the package stats, reference:
    #
    # R Core Team (2017). R: A language and environment for statistical
    # computing. R Foundation for Statistical Computing, Vienna, Austria. URL
    # https://www.R-project.org/.
    #
    shape = p_values.shape
    p_values = p_values.flatten()
    p = len(p_values)
    i = np.arange(start = p, stop = 0, step = -1)
    o = np.flip(np.argsort(p_values), axis = 0)
    ro = np.argsort(o)
    q = np.sum(1 / np.arange(1,p))
    p_values_adj = np.minimum(1, np.minimum.accumulate(q * p_values[o] / i * p))[ro]

    which_predictable = None
    if confidence is not None:
        which_predictable = np.arange(0, p)[p_values_adj <= confidence]

    p_values_adj = np.reshape(p_values_adj, newshape = shape)

    return p_values_adj, which_predictable

def get_loss_statistics(regr_loss, baseline_loss, parametric, confidence):
    if parametric:
        tt_res = stats.ttest_rel(regr_loss, baseline_loss)
        p_value = (tt_res[0] > 0) + np.sign((tt_res[0] < 0) - 0.5) * tt_res[1] / 2
    else:
        loss_stat = compare_methods(regr_loss, baseline_loss).evaluate()
        p_value = loss_stat.prob
        diff_mean = loss_stat.loss_means[1] - loss_stat.loss_means[0]
        diff_se = np.sqrt(loss_stat.loss_se[0] ** 2 + loss_stat.loss_se[1] ** 2)
        conf_int = (diff_mean + stats.norm.ppf(confidence) * diff_se, diff_mean,
                            diff_mean + stats.norm.ppf(1 - confidence) * diff_se)

    return p_value, conf_int

def pred_indep(y, x, method_type = None, z = None, method = 'multiplexing', estimators = None,
                        cutoff_categorical = 10, parametric = False, confidence = 0.05, symmetric = True):

    for twice in range(2):
        if z is not None:
            x_all = np.append(x, z, axis=1)
            x_tn, x_ts, y_tn, y_ts, z_tn, z_ts = train_test_split(x_all, y, z, test_size = 1/3, random_state = 1)
        else:
            x_tn, x_ts, y_tn, y_ts = train_test_split(x, y, test_size = 1/3, random_state = 1)

        p_out = y.shape[1]

        p_values = np.ones(p_out)
        conf_int = np.ones(p_out).astype(list)

        for i in range(p_out):
            if z is None:
                base_loss = MetaEstimator(method, estimators, method_type,cutoff_categorical,
                                            baseline = True).get_residuals(x_tn, x_ts, y_tn[:, i],y_ts[:, i])
            else:
                base_loss = MetaEstimator(method, estimators, method_type,
                                          cutoff_categorical).get_residuals(z_tn, z_ts, y_tn[:,i], y_ts[:,i])

            loss = MetaEstimator(method, estimators, method_type,
                                 cutoff_categorical).get_residuals(x_tn, x_ts, y_tn[:,i], y_ts[:,i])

            p_values[i], conf_int[i] = get_loss_statistics(loss, base_loss, parametric, confidence)

        if symmetric and ('p_values_1' not in locals()):
                y_old = y
                y = x
                x = y_old
                p_values_1 = p_values
                conf_int_out = conf_int
        else:
            break

    if symmetric:
        p_values_adj, which_predictable = FDRcontrol(np.append(p_values_1, p_values), confidence)
        independent = (sum(which_predictable) == 0, np.min(p_values_adj))
        if len(p_values_1) > 1:
            p_values_adj, which_predictable = FDRcontrol(p_values_1, confidence)
        else:
            p_values_adj = p_values_1
            which_predictable = (0, [])[(p_values_1[0] > confidence)]
    else:
        if len(p_values) > 1:
            p_values_adj, which_predictable = FDRcontrol(p_values, confidence)
            independent = sum(which_predictable) == 0
        else:
            p_values_adj = p_values
            which_predictable = (0, [])[p_values[0] > confidence]
    return p_values_adj, which_predictable, independent, conf_int_out
