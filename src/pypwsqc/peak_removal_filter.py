"""Skript with functions for the peak removal filter."""

# import packages
import numpy as np
import poligrain as plg
import pyproj
import xarray as xr
from tqdm import tqdm


def convert_to_utm(dataset, name_coord_lon, name_coord_lat, zone):
    """Convert lon and lat from WGS84 to UTM (zone) and add them as x and y coordinates.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing longitude and latitude coordinates.
    name_coord_lon : str
        Name of the longitude coordinate in the dataset.
    name_coord_lat : str
        Name of the latitude coordinate in the dataset.

    Returns
    -------
    xr.Dataset
        The dataset with added x and y coordinates in UTM (zone).
    """
    lon = dataset[name_coord_lon].to_numpy()
    lat = dataset[name_coord_lat].to_numpy()
    projection = pyproj.Proj(proj="utm", zone=zone, ellps="WGS84", preserve_units=True)

    x, y = projection(lon, lat)

    dataset = dataset.assign_coords({"x": (("id"), x), "y": (("id"), y)})
    dataset.coords["x"].attrs["units"] = f"meters (UTM {zone})"
    dataset.coords["y"].attrs["units"] = f"meters (UTM {zone})"
    return dataset


def get_closest_points_to_point(
    ds_points, ds_points_neighbors, max_distance, n_closest
):
    """Get the closest points for given point locations.

    Note that both datasets that are passed as input have to have the variables x and y
    which should be projected coordinates that preserve lengths as good as possible.

    Parameters
    ----------
    ds_points : xr.DataArray | xr.Dataset
        This is the dataset for which the nearest neighbors will be looked up. That is,
        for each point location in this dataset the nearest neighbors from
        ds_points_neighbors will be returned.
    ds_points_neighbors : xr.DataArray | xr.Dataset
        This is the dataset from which the nearest neighbors will be looked up.
    max_distance : float
        The allowed distance of neighbors has to be smaller than max_distance.
        The unites are the units used for the projected coordinates x and y in the
        two datasets.
    n_closest : int
        The maximum number of nearest neighbors to be returned.

    Returns
    -------
    xr.Dataset
        A dataset which has distance and neighbor_id as variables along the dimensions
        id, taken from ds_points and n_closest. The unit of the distance follows from
        the unit of the projected coordinates of the input datasets. The neighbor_id
        entries for point locations that are further away then max_distance are set to
        None. The according distances are np.inf.
    """
    return plg.spatial.get_closest_points_to_point(
        ds_points, ds_points_neighbors, max_distance, n_closest
    )


def get_nan_sequences(dataset, station, quantile, seq_len_threshold):
    """Find values higher than the threshold and check for leading nan sequences.

    If there are leading nan sequences, find the start and end of the sequence and the
    length of the sequence.

    Parameters
    ----------
    dataset : xr.DataArray
        Dataset, following the OpenSense data format standards.
    station : str
        Name/number of the station.
    quantile : float
        Quantile for the peak determination.
    seq_len_threshold : int
        Nan sequence has to be greater than seq_len_threshold to be considered for the
        peak removal process.

    Returns
    -------
    time_peak_lst : list
        List of times of the peaks.
    seq_start_lst : list
        List of start times of the nan sequences.
    seq_end_lst : list
        List of end times of the nan sequences.
    seq_len_lst : list
        List of lengths of the nan sequences.
    """
    data = dataset.sel(id=station).rainfall
    # get the threshold for the peaks and set time between measurements
    threshold = np.nanquantile(data, quantile)
    timegap = int(
        (data.time.to_numpy()[1] - data.time.to_numpy()[0]) / np.timedelta64(1, "m")
    )
    timedelta = np.timedelta64(timegap, "m")  # get the peaks
    peaks = data.where(data > threshold, drop=True)

    time_peak_lst = []
    seq_start_lst = []
    seq_end_lst = []
    seq_len_lst = []

    # iterate over the peaks and check if there are leading nan sequences
    for time_peak in tqdm(
        peaks.time.to_numpy(),
        desc="Check peaks for leading nans",
        unit=" peaks",
        total=len(peaks.time.to_numpy()),
    ):
        length = 0
        # check if there are leading nan sequences
        # start from the end of the potential nan sequence and go backwards as long as
        # value is nan.
        for value in reversed(
            data.sel(time=slice(None, time_peak - timedelta)).isnull().to_numpy()
        ):
            if value:
                length += 1
            elif length > seq_len_threshold:
                seq_start = time_peak - (timedelta * length)
                time_peak_lst.append(time_peak)
                seq_start_lst.append(seq_start)
                seq_end_lst.append(time_peak - timedelta)
                seq_len_lst.append(length)
                length = 0
                break
            else:
                break
        if length > seq_len_threshold:
            seq_start = time_peak - (timedelta * length)
            time_peak_lst.append(time_peak)
            seq_start_lst.append(seq_start)
            seq_end_lst.append(time_peak - timedelta)
            seq_len_lst.append(length)
    return time_peak_lst, seq_start_lst, seq_end_lst, seq_len_lst


def print_info(
    dataset,
    station,
    max_distance,
    n_closest,
    quantile,
    time_peak_lst,
    seq_len_lst,
    aa_closest_neighbors,
    ab_closest_neighbors,
):
    """
    Print some information about the selected station.

    Parameters
    ----------
    dataset : xr.DataArray
        Dataset, following the OpenSense data format standards.
    station : str
        Name/number of the station.
    max_distance : float
        The allowed distance of neighbors has to be smaller than max_distance. The
        unites are the units used for the projected coordinates x and y in the
        two datasets.
    n_closest : int
        The maximum number of nearest neighbors to be returned.
    quantile : float
        Quantile for the peak determination.
    time_peak_lst : list
        List of times of the peaks.
    seq_len_lst : list
        List of lengths of the nan sequences.
    aa_closest_neighbors : xr.Dataset
        Dataset containing the closest neighbors of a_dataset with a_dataset as
        neighbor, the distances of the neighbors and the ids of the neighbors.
    ab_closest_neighbors : xr.Dataset
        Dataset containing the closest neighbors of a_dataset with b_dataset as
        neighbor, the distances of the neighbors and the ids of the neighbors.

    Return
    ------
    None
    """
    info_lst = []
    _quantile = np.nanquantile(dataset.sel(id=station).rainfall.to_numpy(), quantile)
    num_peaks = len(time_peak_lst)
    avg_seq_len = round(np.mean(seq_len_lst), 2)
    perc_nans = round(
        (
            np.sum(seq_len_lst)
            / np.isnan(dataset.sel(id=station).rainfall.to_numpy()).sum()
        )
        * 100,
        2,
    )
    pws_neighbors = (
        np.count_nonzero(
            [
                aa_closest_neighbors.sel(id=station).neighbor_id.to_numpy()[i]
                is not None
                for i in range(
                    len(aa_closest_neighbors.sel(id=station).neighbor_id.to_numpy())
                )
            ]
        )
        - 1
    )

    print(f"station: {station}", "\n")
    print(f"max_distance: {max_distance} m")
    print(f"n_closest: {n_closest}")
    print(f"{quantile}-quantile: {_quantile} mm", "\n")
    print(f"number of peaks with nan sequence: {num_peaks}")
    print(f"average length of the nan sequences: {avg_seq_len}")
    print(f'percentage "peak nans" of total nan values: {perc_nans} %', "\n")
    print(f"pws neighbors found: {pws_neighbors}")

    info_lst.append(_quantile)
    info_lst.append(num_peaks)
    info_lst.append(avg_seq_len)
    info_lst.append(perc_nans)
    info_lst.append(pws_neighbors)

    if ab_closest_neighbors is None:
        info_lst.append(0)
        print("reference neighbors found: 0")
    else:
        ref_neighbors = np.count_nonzero(
            [
                ab_closest_neighbors.sel(id=station).neighbor_id.to_numpy()[i]
                is not None
                for i in range(
                    len(aa_closest_neighbors.sel(id=station).neighbor_id.to_numpy())
                )
            ]
        )
        info_lst.append(ref_neighbors)
        print(f"reference neighbors found: {ref_neighbors}")
    return info_lst


def inverse_distance_weighting(closest_neighbors):
    """Calculate weights for the closest neighbors, applying inverse distance weighting.

    Parameters
    ----------
    closest_neighbors : xr.Dataset
        Dataset containing the closest neighbors, the distances of the neighbors and the
        ids of the neighbors.

    Returns
    -------
    weights_da : xr.DataArray
        DataArray containing the weights for the closest neighbors.
    """
    weights_lst = []
    # iterate over all stations
    for station in tqdm(
        closest_neighbors.id.to_numpy(),
        desc="Calculate weights for stations",
        unit=" stations",
        total=len(closest_neighbors.id.to_numpy()),
    ):
        # get the distances of the neighbors
        distances = closest_neighbors.sel(id=station).distance.to_numpy()
        distances[
            distances == 0
        ] = (
            np.inf
        )  # in case of zero distance (selected station with itselfe), set to inf
        # check if neighbors are available, if not, all weights are NaN
        if np.all(distances == np.inf):
            weights_lst.append(np.full(len(distances), np.nan))
            continue
        # calculate the weights
        x = 1 / distances**2
        y = (1 / distances**2).sum()
        weight = x / y
        weights_lst.append(weight)
    # convert the list of weights to a DataArray
    weights = np.array(weights_lst)
    return xr.DataArray(
        weights,
        dims=["id", "weights"],
        coords={"id": closest_neighbors.id.to_numpy()},
        name="weights",
    )


def interpolate_precipitation(
    dataset,
    station,
    closest_neighbors,
    weights_da,
    seq_start_lst,
    time_peak_lst,
    seq_len_lst,
    seq_nan_threshold=float,
    min_station_threshold=int,
):
    """Interpolate the precipitation values to obtain the "precipitation shape/profile".

    Parameters
    ----------
    dataset : xr.DataArray
        Dataset, following the OpenSense data format standards.
    station : str
        Name/number of the station.
    closest_neighbors : xr.Dataset
        Dataset containing the closest neighbors, the distances of the neighbors and the
        ids of the neighbors.
    weights_da : xr.DataArray
        DataArray containing the weights for the closest neighbors.
    seq_start_lst : list
        List of start times of the nan sequences.
    time_peak_lst : list
        List of times of the peaks.
    seq_len_lst : list
        List of lengths of the nan sequences.
    seq_nan_threshold : float
        Threshold for the maximum allowed percentage of nan values in the time series of
        the closest neighbors referring to the time series of the nan sequence leading
        the peak.
    min_station_threshold : int
        Threshold for the minimum number of useful stations. The time series of the
        closest neighbors, therefore the closest neighbor (with a_dataset as neighbors)
        is useful, if the percentage of nan values is lower than the threshold of
        nan values per sequence.

    Returns
    -------
    seqs_lst : list
       List of interpolated precipitation values for the observed station.
    """
    # get the time of the observed station
    time = dataset.time.to_numpy()

    # get neighbors and weights
    # in case of the selected station is in the list of neighbors and weights, do not
    # include the station itselfe
    if station in closest_neighbors.sel(id=station).neighbor_id.to_numpy():
        neighbors = closest_neighbors.sel(id=station).neighbor_id.to_numpy()[1:]
        weights = weights_da.sel(id=station).to_numpy()[1:]
    else:
        neighbors = closest_neighbors.sel(id=station).neighbor_id.to_numpy()
        weights = weights_da.sel(id=station).to_numpy()

    all_neighbors_seqs = []
    # list of lists. Each list corresponds to one neighbor containing his time series
    # with starts and ends of nan sequences of the selected station
    # iterate over all neighbors
    for neighbor in tqdm(
        neighbors,
        desc="Get precipitation values from neighbors",
        unit=" neighbors",
        total=np.count_nonzero(
            [neighbors[i] is not None for i in range(len(neighbors))]
        ),
    ):
        neighbor_seqs = []
        # list of time series of the neighbor containing his time series with starts
        # and ends of nan sequences of the selected station
        if neighbor is None:  # stop if there are no (no more) neighbors
            break
        # iterate over all nan sequences of the selected station
        for seq_start, peak, seq_len in zip(
            seq_start_lst, time_peak_lst, seq_len_lst, strict=False
        ):
            # check if the start and peak of the nan sequence are in the time series of
            # the neighbor. If not, set this time series from this neighbor to NaN
            if seq_start not in time or peak not in time:
                values = np.full(seq_len + 1, np.nan)
            else:
                values = (
                    dataset.sel(id=neighbor)
                    .rainfall.sel(time=slice(seq_start, peak))
                    .to_numpy()
                )

                # check, if the nan_threshold is exceeded. If so, set this time series
                # from this neighbor to NaN
                nan_count = np.isnan(values).sum()
                if nan_count == 0:
                    pass
                elif nan_count / len(values) > seq_nan_threshold:
                    values = np.full(seq_len + 1, np.nan)
                else:
                    values = np.nan_to_num(values, nan=0)
                    # If nan threshold is not exceeded, set the nan values to zero
            neighbor_seqs.append(values)
        all_neighbors_seqs.append(neighbor_seqs)

    seqs_lst = []
    # list of time series with the interpolated precipitation values for the selected
    # station
    # iterate over all nan sequences of the selected station
    for i, length in tqdm(
        zip(range(len(seq_len_lst)), seq_len_lst, strict=False),
        desc="Interpolate precipitation values for sequences",
        unit=" sequences",
        total=len(seq_len_lst),
    ):
        useful_station_count = 0
        seq = np.zeros(
            length + 1
        )  # sequence length + 1 to later also assign a new value to the peak
        # iterate over all neighbors and their time series
        for neighbor_seqs, weight in zip(all_neighbors_seqs, weights, strict=False):
            if np.isnan(neighbor_seqs[i]).any():
                # If the neighbor time series was set to NaN, skip this neighbor for
                # this sequence
                continue
            seq += neighbor_seqs[i] * weight
            useful_station_count += 1

        # check if there are enough useful stations. If so, there is no interpolated
        # time series for this nan sequence
        if useful_station_count < min_station_threshold:
            seq = np.full(length + 1, np.nan)

        seqs_lst.append(seq)
    return seqs_lst


def distribute_peak(dataset, station, time_peak_lst, seqs_lst):
    """Distribute the peak  following the "precipitation shape/profile".

    Parameters
    ----------
    dataset : xr.DataArray
        Dataset, following the OpenSense data format standards.
    station : str
        Name/number of the station.
    time_peak_lst : list
        List of times of the peaks.
    seqs_lst : list
        List containing the interpolate precipitation values of the observed station for
        the time series of nan sequences leading a peak.

    Returns
    -------
    seqs_corr_lst : list
       List of arrays containing the values of the distributet peak for the
       nan sequences.
    """
    seqs_corr_lst = []
    # iterate over all peaks (nan sequences) of the selected station
    for time_peak, seq_num in tqdm(
        zip(time_peak_lst, range(len(seqs_lst)), strict=False),
        desc="Distribute peaks",
        unit=" peaks",
        total=len(time_peak_lst),
    ):
        # check if the time series of the selected station is NaN. If so, skip this
        # sequence. This peak will not be distributed/corrected.
        if np.isnan(seqs_lst[seq_num]).any():
            seqs_corr_lst.append(seqs_lst[seq_num])
            continue
        seq_sum = seqs_lst[seq_num].sum()
        # check if the sum of the interpolated time series is zero. If so, skip this
        # sequence. This peak and his leading nan sequence will be set to zeros.
        if seq_sum == 0:
            seqs_corr_lst.append(seqs_lst[seq_num])
        else:
            # distribute the peak to the leading nan sequence with portions of the
            # precipitation shape/profile
            value_peak = dataset.sel(id=station).rainfall.sel(time=time_peak).to_numpy()

            seq_corr = (seqs_lst[seq_num] / seq_sum) * value_peak
            seqs_corr_lst.append(seq_corr)

    return seqs_corr_lst


def overwrite_seq(dataset, station, seqs_corr_lst, seq_start_lst, time_peak_lst):
    """Overwrite the sequence of nan values leading a peak with the corrected sequences.

    Parameters
    ----------
    dataset : xr.DataArray
        Dataset, following the OpenSense data format standards.
    station : str
        Name/number of the station.
    seqs_corr_lst : list
       List of arrays containing the values of the distributet peak.
    seq_start_lst : list
        List of start times of the nan sequences.
    time_peak_lst : list
        List of times of the peaks.

    Returns
    -------
    data_corr : xr.Dataset
        Dataset, following the OpenSense data format standards with the corrected peaks
        and nan sequences.
    """
    # create a copy of the dataset
    data_corr = dataset.copy(deep=True)
    data_corr.load()
    # iterate over all sequences and overwrite the values of the leading nan sequences
    # and peaks with the corrected values
    for seq_corr, seq_start, peak in tqdm(
        zip(seqs_corr_lst, seq_start_lst, time_peak_lst, strict=False),
        desc="Overwrite sequences",
        unit=" sequences",
        total=len(seqs_corr_lst),
    ):
        # check if the time series of the selected station is NaN. If so, do nothing and
        # skip this sequence
        if np.isnan(seq_corr).any():
            continue
        time_slice = slice(seq_start, peak)
        data_corr["rainfall"].loc[{"id": station, "time": time_slice}] = seq_corr
    return data_corr
