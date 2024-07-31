""" scratchpad for testing and implementing pwspyqc-functions"""
import pandas as pd
import xarray as xr
import datetime
import os
import re
import tqdm
import warnings
import glob
import poligrain as plg




def calc_indicator_correlation(a_dataset, b_dataset, prob, exclude_nan=True, min_valid_overlap=None):
    """
    To calculate indicator correlation two datasets

    Parameters
    ----------
    a_dataset: first data vector
    b_dataset: second data vector
    perc: percentile threshold 
    
    Returns
    ----------
    indicator correlation value

    Raises
    ----------

    """

    if len(a_dataset.shape) != 1:
        raise ValueError('`a_dataset` has to be a 1D numpy.ndarray')
    if a_dataset.shape != b_dataset.shape:
        raise ValueError('`a_dataset` and `b_dataset` have to have the same shape')
    
    a_dataset = np.copy(a_dataset)
    b_dataset = np.copy(b_dataset)

    both_not_nan = ~np.isnan(a_dataset) & ~np.isnan(b_dataset)
    if exclude_nan:
        a_dataset = a_dataset[both_not_nan]
        b_dataset = b_dataset[both_not_nan]

    elif min_valid_overlap is not None:
        sum(both_not_nan) < min_valid_overlap:
        return np.nan
    else:
        if sum(both_not_nan) == 0:
            raise ValueError('No overlapping data. Define `min_valid_overlap` to return NaN in such cases.')

    # Get index at quantile threshold `prob`
    ix = int(a_dataset.shape[0] * prob)

    # Set values below quantile threshold `prob` to 0
    # and above to 1
    a_sort = np.sort(a_dataset)
    b_sort = np.sort(b_dataset)
    a_dataset[a_dataset < a_sort[ix]] = 0
    b_dataset[b_dataset < b_sort[ix]] = 0
    a_dataset[a_dataset > 0] = 1
    b_dataset[b_dataset > 0] = 1

    # Calculate correlation of 0 and 1 time series
    cc = np.corrcoef(a_dataset, b_dataset)[0, 1]

    return cc


def calc_indic_corr_all_stns(
    da_a,
    da_b,
    max_distance=50000, # this is in meters, assuming the projection units are also meters
    prob=0.99, 
    exclude_nan=True, 
    min_valid_overlap=None,
):

    """
    Indicator correlation between reference and test stations
    
    return: indicator correlation and distance values
    
    """
    import scipy
    xy_a = list(zip(da_a.x.data, da_a.y.data))
    xy_b = list(zip(da_b.x.data, da_b.y.data))
    dist_mtx = scipy.spatial.distance.cdist(xy_a, xy_b, metric='euclidean')
    indcorr_mtx=np.full_like(dist_mtx, np.nan)
    #list_corr = []
    #list_dist = []
    for i in tqdm.tqdm(range(len(xy_a) - 1)):
        for j in range(i + 1, len(xy_b)):
            # check if distance between stations is less than max_distance
            if dist_mtx[i, j] < max_distance:
                ts_a = da_a.isel(id=i)
                ts_b = da_b.isel(id=j)
                ts_b = ts_b.reindex({'time': ts_a.time})

                indcorr_mtx[i,j] = calc_indicator_correlation(
                    ts_a.data, 
                    ts_b.data,
                    prob=prob,
                    exclude_nan=exclude_nan, 
                    min_valid_overlap=min_valid_overlap,
                )
                #list_dist.append(dist_mtx[i, j])
                #list_corr.append(indi_corr)
    #dist_vals = np.asarray(list_dist)
    #corr_vals = np.asarray(list_corr) 
    
    # Dimensionen benennen! StastionsID Als xarray.Darray
    # dist_vals=xr.DataArray(dist_vals, coords={'id' : da_a.id , 'id_neighbor' : da_b.id })
    # corr_cals=.....
    # return dist_vals, corr_vals
    #return dist_mtx, indcorr_mtx
    return (
        xr.DataArray(data=dist_mtx, dims=('id', 'id_neighbor'), coords={'id' : ('id', da_a.id.data) , 'id_neighbor' : ('id_neighbor' , da_b.id.data) }), 
        xr.DataArray(data=indcorr_mtx, dims=('id', 'id_neighbor'), coords={'id' : ('id', da_a.id.data) , 'id_neighbor' : ('id_neighbor' , da_b.id.data) }),
    )






# def indicator_filter(xy_net, prc_net, xy_dwd, prc_dwd,
#                      prob=0.99, max_distance=50000,
#                      min_req_ppt_vals=2*24*30,
#                      show_plot=False,
#                      fn_figure='Indicator Filter',
#                      save_folder=None,
#                     tolerance=0.8):
#     """
#     Filters stations of the secondary network by comparing indicator
#     correlations between primary and secondary network and nearest
#     stations of the primary network.

#     Parameters
#     ----------
#     coords_net: 'numpy.ndarray' [N x 2]
#         Coordinates of secondary network
#     data_net: 'numpy.ndarray' [timesteps x N]
#         Dataset of secondary network
#     coords_dwd: 'numpy.ndarray' [M x 2]
#         Coordinates of primary network
#     data_dwd: 'numpy.ndarray' [timesteps x M]
#         Dataset of primary network
#     prob: 'float', optional (default: 0.99)
#         Percentile for the determination of the indicator correlation
#     max_distance: 'int', optional (default: 50000)
#         Distance limit between stations of primary network
#     perc_avail_data: 'float', optional (default: 0.7)
#         Percentage of available time steps
#     show_plot: 'bool', optional (default: False)
#         Show plots

#     Returns
#     ----------
#     stn_in_bool: 'numpy.ndarray (bool)' [N]
#         True: Station is 'good', False: Station is 'bad'

#     Raises
#     ----------

#     """










# def calc_indic_corr_all_stns_new(
#     da_a,
#     da_b,
#     max_distance=50000, # this is in meters, assuming the projection units are also meters
#     prob=0.99, 
#     exclude_nan=True, 
#     min_valid_overlap=None,
# ):

#     """
#     Indicator correlation between reference and test stations
    
#     return: indicator correlation and distance values
    
#     """
#     import scipy
#     xy_a = list(zip(da_a.x.data, da_a.y.data))
#     xy_b = list(zip(da_b.x.data, da_b.y.data))
#     dist_mtx = scipy.spatial.distance.cdist(xy_a, xy_b, metric='euclidean')
    
#     list_corr = []
#     list_dist = []
#     for i in tqdm.tqdm(range(len(xy_a) - 1)):
#         for j in range(i + 1, len(xy_b)):
#             # check if distance between stations is less than max_distance
#             if dist_mtx[i, j] < max_distance:
#                 indi_corr = calc_indicator_correlation(
#                     da_a.isel(id=i).data, 
#                     da_b.isel(id=j).data,
#                     prob=prob,
#                     exclude_nan=exclude_nan, 
#                     min_valid_overlap=min_valid_overlap,
#                 )
#                 list_dist.append(dist_mtx[i, j])
#                 list_corr.append(indi_corr)
#     dist_vals = np.asarray(list_dist)
#     corr_vals = np.asarray(list_corr) 
    
#     # Dimensionen benennen! StastionsID Als xarray.Darray
#     dist_vals=xr.DataArray(dist_vals, coords={'id' : da_a.id , 'id_neighbor' : da_b.id })
#     corr_cals=.....
#     return dist_vals, corr_vals
    

def indicator_filter(xy_net, prc_net, xy_dwd, prc_dwd,
                     prob=0.99, max_distance=50000,
                     min_req_ppt_vals=2*24*30,
                     show_plot=False,
                     fn_figure='Indicator Filter',
                     save_folder=None,
                    tolerance=0.8):
    """
    Filters stations of the secondary network by comparing indicator
    correlations between primary and secondary network and nearest
    stations of the primary network.

    Parameters
    ----------
    coords_net: 'numpy.ndarray' [N x 2]
        Coordinates of secondary network
    data_net: 'numpy.ndarray' [timesteps x N]
        Dataset of secondary network
    coords_dwd: 'numpy.ndarray' [M x 2]
        Coordinates of primary network
    data_dwd: 'numpy.ndarray' [timesteps x M]
        Dataset of primary network
    prob: 'float', optional (default: 0.99)
        Percentile for the determination of the indicator correlation
    max_distance: 'int', optional (default: 50000)
        Distance limit between stations of primary network
    perc_avail_data: 'float', optional (default: 0.7)
        Percentage of available time steps
    show_plot: 'bool', optional (default: False)
        Show plots

    Returns
    ----------
    stn_in_bool: 'numpy.ndarray (bool)' [N]
        True: Station is 'good', False: Station is 'bad'

    Raises
    ----------

    """
    

    
    # calculate indicator correlation between dwd stations
    dist_matrix_dwd_dwd = scsp.distance.cdist(xy_dwd, xy_dwd,
                                              metric='euclidean')
    
    dist_matrix_dwd_net = scsp.distance.cdist(xy_dwd, xy_net,
                                              metric='euclidean')
    
    dist_dwd, corr_dwd = calc_indic_corr_all_stns(coords_stns_xy=xy_dwd,
                                                pcp_vals=prc_dwd.values,
                                                max_distance=max_distance,
                                                min_req_ppt_vals=min_req_ppt_vals, prob=prob)

    #print(dist_dwd, corr_dwd)

    if show_plot:
        stn_in = []
        dist_stn_in = []
        
        stn_notin = []
        dist_stn_notin = []
        
    stn_in_bool = np.zeros(dist_matrix_dwd_net.shape[1], dtype=bool)
    for i in tqdm.tqdm(range(dist_matrix_dwd_net.shape[1])):
        #print(i, dist_matrix_dwd_net.shape[1])
        net_stn = prc_net.iloc[:, i]
        net_stn[net_stn == -9] = np.nan
        net_stn_nonan = net_stn.dropna(how='all')

        nearest_stn_ids = np.argsort(dist_matrix_dwd_net[:, i])
        #print('nearest_stn_ids', len(nearest_stn_ids))
        for stn_id in nearest_stn_ids:  # TODO: notwendig?
            # print()


            prim_stn = prc_dwd.iloc[:, stn_id]
            prim_stn[prim_stn== -9] = np.nan
            prim_stn_nonan = prim_stn.dropna(how='all')

            ij_bool_avail_data = net_stn_nonan.index.intersection(
                prim_stn_nonan.index)
                
            #print(ij_bool_avail_data)
            # If sufficient number of data available
            if len(ij_bool_avail_data) > min_req_ppt_vals:

                indi_corr = calc_indicator_correlation(
                    net_stn_nonan.loc[ij_bool_avail_data].dropna(),
                    prim_stn_nonan.loc[ij_bool_avail_data].dropna(),
                    prob)
                #print(indi_corr, dist_matrix_dwd_net[stn_id, i])
                delta = 1000
                va = corr_dwd[dist_dwd <
                              dist_matrix_dwd_net[stn_id, i] + delta]
                add_distance = 1000
                while va.shape[0] < 5:
                    va = corr_dwd[
                        dist_dwd < dist_matrix_dwd_net[
                            stn_id, i] + delta + add_distance]
                    add_distance += delta
                    #print(va)
                #print(indi_corr, np.min(va), np.min(va*tolerance))
                if indi_corr > np.min(va*tolerance, 0):
                    stn_in_bool[i] = True
                    #print('PWS accepted')
                    if show_plot:
                        stn_in.append(indi_corr)
                        dist_stn_in.append(dist_matrix_dwd_net[stn_id, i])
                    #break
                    
                else:
                    stn_notin.append(indi_corr)
                    dist_stn_notin.append(dist_matrix_dwd_net[stn_id, i])
                    #break
                

    print("Total Accepted", np.sum(stn_in_bool), dist_matrix_dwd_net.shape[1])

    if show_plot:
        plt.figure(dpi=200)
        plt.scatter(dist_stn_in, stn_in, alpha=0.4, s=15, c='blue',
                    label='PWS-Prim. In n=%d' % np.sum(stn_in_bool*1))
        plt.scatter(dist_dwd, corr_dwd, alpha=0.4, s=15, c='red',
                    label='Prim.-Prim.')
                            
        plt.scatter(dist_stn_notin, stn_notin, alpha=0.4, s=15, c='grey',
                    label='PWS-Prim. Out n=%d' % (
                        len(stn_in_bool) - np.sum(stn_in_bool*1)))

        plt.xlim([0, 30000])
        plt.ylim([0, 1])
        plt.grid()
        plt.title('{}'.format(fn_figure))
        plt.xlabel('Distance between stations [m]')
        plt.ylabel('Indicator Correlation p{:d} [-]'.format(int(prob * 100)))
        plt.legend(loc=1)
        plt.tight_layout()
        #if save_folder:
        #    plt.savefig(Path(save_folder, '{}.png'.format(fn_figure)), dpi=200,
        #                bbox_inches='tight')
        #else:
        #    plt.savefig('{}.png'.format(fn_figure), dpi=200,
        #                bbox_inches='tight')
        plt.show()

    return stn_in_bool
