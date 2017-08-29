import matplotlib.pyplot as plt

n_list = np.round(np.exp(list(np.arange(6,10,0.1)))).astype(int)
size_mat = 10
B = 10

def get_conf_ints(series, sd):
    sd[sd == 0] = 0.01

    low = series - 1.64 * sd
    high = series + 1.64 * sd

    low[low <= 0] = 0
    high[high >= 1] = 1

    return low, high

def get_statistics(result_mat):
    
    if len(result_mat.shape) == 4:

        fdr = np.nan_to_num(result_mat[:,1,0,:] / np.sum(result_mat[:,1,:,:], axis = 1)) ## FDR
        pwr = np.nan_to_num(result_mat[:,1,1,:] / np.sum(result_mat[:,:,1,:], axis = 1)) ## Power

        fdr_sd = np.sqrt(np.var(fdr, axis = 1))
        pwr_sd = np.sqrt(np.var(pwr, axis = 1))

        fdr = np.mean(fdr, axis = 1)
        pwr = np.mean(pwr, axis = 1)
        
    if len(result_mat.shape) == 3:

        fdr = np.nan_to_num(result_mat[:,1,0] / np.sum(result_mat[:,1,:], axis = 1)) ## FDR
        pwr = np.nan_to_num(result_mat[:,1,1] / np.sum(result_mat[:,:,1], axis = 1)) ## Power

        fdr_sd = None
        pwr_sd = None
        
    return(fdr, fdr_sd, pwr, pwr_sd)

def draw_performance_graph(n_list, fdr, pwr, fdr_sd, pwr_sd, title):
    plt.figure(figsize=(10, 6))
    plt.xscale('log')
    plt.xlabel('n')
    plt.ylim((0,1))
    pwr_low, pwr_high = get_conf_ints(pwr, pwr_sd)
    fdr_low, fdr_high = get_conf_ints(fdr, fdr_sd)
    plt.fill_between(n_list, pwr_low, pwr_high, color = 'skyblue', alpha = 0.5)
    plt.plot(n_list, pwr, color = 'blue')

    plt.fill_between(n_list, fdr_low, fdr_high, color = 'lightcoral', alpha = 0.5)
    plt.plot(n_list, fdr, color = 'red')

    plt.plot((np.min(n_list), np.max(n_list)), (0.05, 0.05), '--')
    plt.title(('Power curve and FDR for ' + title))
    plt.legend(['Power', 'FDR'], loc = 2)
    plt.xticks([500, 1000, 2500, 5000, 10000, 20000], [500, 1000, 2500, 5000, 10000, 20000])

none = np.load('16082017none.npy')
stack = np.load('16082017stacking.npy')
mplx = np.load('16082017multiplexing.npy')

none_old = np.load('05082017none.npy')
stack_old = np.load('05082017stacking.npy')
mplx_old = np.load('05082017multiplexing.npy')

none_fdr, none_fdr_sd, none_pwr, none_pwr_sd = get_statistics(none)
stack_fdr, stack_fdr_sd, stack_pwr, stack_pwr_sd = get_statistics(stack)
mplx_fdr, mplx_fdr_sd, mplx_pwr, mplx_pwr_sd = get_statistics(mplx)

none_old_fdr, temp, none_old_pwr, temp = get_statistics(none_old)
stack_old_fdr, temp, stack_old_pwr, temp = get_statistics(stack_old)
mplx_old_fdr, temp, mplx_old_pwr, temp = get_statistics(mplx_old)

none_fdr = (none_fdr + none_old_fdr) / 2
stack_fdr = (stack_fdr + stack_old_fdr) / 2
mplx_fdr = (mplx_fdr + mplx_old_fdr) / 2

none_pwr = (none_pwr + none_old_pwr) / 2
stack_pwr = (stack_pwr + stack_old_pwr) / 2
mplx_pwr = (mplx_pwr + mplx_old_pwr) / 2

draw_performance_graph(n_list, none_fdr, none_pwr, none_fdr_sd, none_pwr_sd, 'no emsembling')
draw_performance_graph(n_list, stack_fdr, stack_pwr, stack_fdr_sd, stack_pwr_sd, 'stacking')
draw_performance_graph(n_list, mplx_fdr, mplx_pwr, mplx_fdr_sd, mplx_pwr_sd, 'multiplexing')
np.mean(none_fdr)