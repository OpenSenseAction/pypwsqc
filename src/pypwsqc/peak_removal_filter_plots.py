"""Skript with functions for plots for the peak removal filter."""

# import packages
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


def plot_station_neighbors(
    a_dataset,
    b_dataset,
    station,
    aa_closest_neighbors,
    ab_closest_neighbors,
    max_distance,
    zoom=True,
):
    """Plot the neighbor stations of a selected station.

    Parameters
    ----------
    a_dataset : xr.DataArray
        Dataset, following the OpenSense data format standards.
    b_dataset : xr.DataArray
        Dataset, following the OpenSense data format standards.
    station : str
        Name/number of the station.
    aa_closest_neighbors : xr.Dataset
        Dataset containing the closest neighbors of a_dataset with a_dataset as
        neighbor, the distances of the neighbors and the ids of the neighbors.
    ab_closest_neighbors : xr.Dataset
        Dataset containing the closest neighbors of a_dataset with b_dataset as
        neighbor, the distances of the neighbors and the ids of the neighbors.
    max_distance : float
        The allowed distance of neighbors has to be smaller than max_distance.
        The unites are the units used for the projected coordinates x and y in the
        two datasets.

        Has to be the same as in closest_neighbors.
    zoom : bool
        If True, the plot is zoomed in on the station. If False,
        the plot is not zoomed in.

    Returns
    -------
    None
        Plot of the station and its neighbors.
    """
    # get the x and y coordinates of the station
    x_station = a_dataset.sel(id=station).x.to_numpy()
    y_station = a_dataset.sel(id=station).y.to_numpy()

    # get the x and y coordinates of all stations and get the neighbors of the station
    x = a_dataset.x.to_numpy()
    y = a_dataset.y.to_numpy()
    aa_neighbor = aa_closest_neighbors.sel(id=station).neighbor_id.to_numpy()
    if b_dataset is not None:
        x_ref = b_dataset.x.to_numpy()
        y_ref = b_dataset.y.to_numpy()
        ab_neigbors = ab_closest_neighbors.sel(id=station).neighbor_id.to_numpy()

    s = 10 if zoom else 2

    # plot all pws stations and reference stations
    fig, ax = plt.subplots(dpi=150)
    plt.scatter(x=x, y=y, s=s, color="blue", alpha=0.5)
    if b_dataset is not None:
        plt.scatter(x=x_ref, y=y_ref, s=s, color="black", alpha=0.5)

    for neighbor in aa_neighbor:
        if neighbor is None:
            continue
        plt.scatter(
            a_dataset.sel(id=neighbor).x.to_numpy(),
            a_dataset.sel(id=neighbor).y.to_numpy(),
            s=s,
            color="lime",
            alpha=0.5,
        )
    if b_dataset is not None:
        for neighbor_ref in ab_neigbors:
            if neighbor_ref is None:
                continue
            plt.scatter(
                b_dataset.sel(id=neighbor_ref).x.to_numpy(),
                b_dataset.sel(id=neighbor_ref).y.to_numpy(),
                s=s,
                color="magenta",
                alpha=0.5,
            )

    # highlight the station
    plt.scatter(x_station, y_station, s=s, color="red", label=f"pws station {station}")

    # plot search radius
    kreis = Circle(
        (x_station, y_station),
        radius=max_distance,
        color="black",
        linewidth=0.5,
        fill=False,
    )
    plt.gca().add_patch(kreis)
    plt.axis("equal")

    # create custom legend
    if b_dataset is not None:
        custom_legend = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                markersize=4,
                label="pws network",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="black",
                markersize=4,
                label="reference network",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="lime",
                markersize=4,
                label="pws neighbors",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="magenta",
                markersize=4,
                label="reference neighbors",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=4,
                label=f"pws station {station}",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=f"search radius: {max_distance/1000} km",
            ),
        ]
    else:
        custom_legend = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                markersize=4,
                label="pws network",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="lime",
                markersize=4,
                label="pws neighbors",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=4,
                label=f"pws station {station}",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=f"search radius: {max_distance/1000} km",
            ),
        ]

    # set zoom scope of the plot
    if zoom:
        plt.xlim(x_station - max_distance * 3, x_station + max_distance * 3)
        plt.ylim(y_station - max_distance * 3, y_station + max_distance * 3)

    name_plot = f"Neighbors of station {station}"
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.title(name_plot)
    plt.legend(handles=custom_legend)

    plt.show()
    plt.close()
    return fig, ax


def plot_peak(
    dataset,
    data_corr,
    station,
    quantile,
    peak_num,
    seq_start_lst,
    time_peak_lst,
    seq_end_lst,
    seq_len_lst,
    data_is_corrected=False,
    zoom_out=15,
):
    """Plot the rainfall of the station with the (corrected) peaks.

    Parameters
    ----------
    dataset : xr.DataArray
        Dataset, following the OpenSense data format standards.
    data_corr : xr.Dataset
        Dataset, following the OpenSense data format standards with the corrected peaks
        and nan sequences.

        If not available, set data_corr to None and data_is_corrected to False.
    station : str
        Name/number of the station.
    quantile : float
        Quantile for the peak determination.
    peak_num : int
        Number of the peak to plot.
    seq_start_lst : list
        List of start times of the nan sequences.
    time_peak_lst : list
        List of times of the peaks.
    seq_end_lst : list
        List of end times of the nan sequences.
    seq_len_lst : list
        List of lengths of the nan sequences.
    data_is_corrected : bool
        Set True, if the dataset is corrected and given as input for data_corr.
    zoom_out : int
        Factor to additionally zoom out the plot. 1 = 5 min.

    Returns
    -------
    None
        Plot of the rainfall of the station with the (corrected) peaks.
    """
    # set timegap and timegap_plt
    timegap = int(
        (dataset.time.to_numpy()[1] - dataset.time.to_numpy()[0])
        / np.timedelta64(1, "m")
    )
    timedelta = np.timedelta64(timegap, "m")
    timegap_plt_before = timedelta * seq_len_lst[peak_num] + zoom_out * timedelta
    timegap_plt_after = timedelta * seq_len_lst[peak_num] + zoom_out * timedelta

    # create time series needed for the plot
    if (
        seq_start_lst[peak_num] - timegap_plt_before
        not in dataset.sel(id=station).time.to_numpy()
    ):
        start = dataset.sel(id=station).time.to_numpy()[0]
    else:
        start = seq_start_lst[peak_num] - timegap_plt_before
    if (
        seq_end_lst[peak_num] + timegap_plt_after
        not in dataset.sel(id=station).time.to_numpy()
    ):
        end = dataset.sel(id=station).time.to_numpy()[-1]
    else:
        end = time_peak_lst[peak_num] + timegap_plt_after

    x = np.arange(start, end + timedelta, timedelta)
    x_nan_seq = np.arange(
        seq_start_lst[peak_num], seq_end_lst[peak_num] + timedelta * 2, timedelta
    )  # includes the peak
    x_nan_seq_others = [
        np.arange(seq_start, seq_end + timedelta * 2, timedelta)
        for seq_start, seq_end in zip(seq_start_lst, seq_end_lst, strict=False)
    ]  # include the peak
    x_peak = np.datetime64(time_peak_lst[peak_num])

    # get the values of the time series
    y_peak_orig = dataset.sel(id=station, time=x_peak).rainfall.to_numpy()
    if data_is_corrected:
        y = data_corr.sel(id=station, time=x).rainfall.to_numpy()
        y_nan_seq = data_corr.sel(id=station, time=x_nan_seq).rainfall.to_numpy()
        y_nan_seq_others = [
            data_corr.sel(id=station, time=x_nan_seq_other).rainfall.to_numpy()
            for x_nan_seq_other in x_nan_seq_others
        ]
        y_peak = data_corr.sel(id=station, time=x_peak).rainfall.to_numpy()
    else:
        y = dataset.sel(id=station, time=x).rainfall.to_numpy()
        y_nan_seq = dataset.sel(id=station, time=x_nan_seq).rainfall.to_numpy()
        y_nan_seq_others = [
            dataset.sel(id=station, time=x_nan_seq_other).rainfall.to_numpy()
            for x_nan_seq_other in x_nan_seq_others
        ]
        y_peak = dataset.sel(id=station, time=x_peak).rainfall.to_numpy()

    fig, ax = plt.subplots()
    if data_is_corrected:
        name_plot_1 = f"Corrected rainfall of station {station} with peak no."
        name_plot_2 = f"{peak_num + 1} out of {len(time_peak_lst)}"
        name_plot = f"{name_plot_1} {name_plot_2}"

    else:
        name_plot_1 = f"Rainfall of station {station} with peak no."
        name_plot_2 = f"{peak_num + 1} out of {len(time_peak_lst)}"
        name_plot = f"{name_plot_1} {name_plot_2}"

    # plot the time series
    ax.stem(x, y, markerfmt="o", linefmt="blue", basefmt=" ").markerline.set_markersize(
        2.5
    )

    # plot the leading nan sequence/the distributed peak values for the nan sequence
    if y_peak == y_peak_orig:
        ax.stem(
            x_nan_seq, y_nan_seq, markerfmt="o", linefmt="red", basefmt=" "
        ).markerline.set_markersize(2.5)
    else:
        ax.stem(
            x_nan_seq, y_nan_seq, markerfmt="o", linefmt="lime", basefmt=" "
        ).markerline.set_markersize(2.5)

    # plot the peak/the new value
    if y_peak == y_peak_orig:
        ax.stem(
            x_peak, y_peak, markerfmt="o", linefmt="red", basefmt=" "
        ).markerline.set_markersize(2.5)
    else:
        ax.stem(
            x_peak, y_peak, markerfmt="o", linefmt="lime", basefmt=" "
        ).markerline.set_markersize(2.5)

    # mark other peaks/remaining or corrected peaks
    for time_peak_other, x_nan_seq_other, y_nan_seq_other in zip(
        time_peak_lst, x_nan_seq_others, y_nan_seq_others, strict=False
    ):
        if time_peak_other in x:
            if time_peak_other == x_peak:  # skip the above selected peak
                continue
            y_peak_other = dataset.sel(
                id=station, time=time_peak_other
            ).rainfall.to_numpy()
            if data_is_corrected:
                y_peak_other_corr = data_corr.sel(
                    id=station, time=time_peak_other
                ).rainfall.to_numpy()
                if (
                    y_peak_other == y_peak_other_corr
                ):  # check if the peak is corrected --> green, if not --> orange
                    ax.stem(
                        x_nan_seq_other,
                        y_nan_seq_other,
                        markerfmt="o",
                        linefmt="orange",
                        basefmt=" ",
                    ).markerline.set_markersize(2.5)
                else:
                    ax.stem(
                        x_nan_seq_other,
                        y_nan_seq_other,
                        markerfmt="o",
                        linefmt="green",
                        basefmt=" ",
                    ).markerline.set_markersize(2.5)
            else:
                # plot other peaks
                ax.stem(
                    time_peak_other,
                    y_peak_other,
                    markerfmt="o",
                    linefmt="orange",
                    basefmt=" ",
                ).markerline.set_markersize(2.5)
        else:
            continue

    # create custom legend
    if data_is_corrected:
        if y_peak == y_peak_orig:
            custom_legend = [
                Line2D(
                    [0],
                    [0],
                    marker="|",
                    color="red",
                    markersize=8,
                    markeredgewidth=1.5,
                    linestyle="",
                    label="not corrected peak",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="|",
                    color="blue",
                    markersize=8,
                    markeredgewidth=1.5,
                    linestyle="",
                    label="legit values",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="|",
                    color="orange",
                    markersize=8,
                    markeredgewidth=1.5,
                    linestyle="",
                    label="remaining peaks",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="|",
                    color="green",
                    markersize=8,
                    markeredgewidth=1.5,
                    linestyle="",
                    label="other corrected peaks",
                ),
                Line2D([0], [0], linestyle="--", color="black", label="threshold"),
            ]
        else:
            custom_legend = [
                Line2D(
                    [0],
                    [0],
                    marker="|",
                    color="lime",
                    markersize=8,
                    markeredgewidth=1.5,
                    linestyle="",
                    label="corrected peak",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="|",
                    color="blue",
                    markersize=8,
                    markeredgewidth=1.5,
                    linestyle="",
                    label="legit values",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="|",
                    color="orange",
                    markersize=8,
                    markeredgewidth=1.5,
                    linestyle="",
                    label="remaining peaks",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="|",
                    color="green",
                    markersize=8,
                    markeredgewidth=1.5,
                    linestyle="",
                    label="other corrected peaks",
                ),
                Line2D([0], [0], linestyle="--", color="black", label="threshold"),
            ]
    else:
        custom_legend = [
            Line2D(
                [0],
                [0],
                marker="|",
                color="red",
                markersize=8,
                markeredgewidth=1.5,
                linestyle="",
                label="peak",
            ),
            Line2D(
                [0],
                [0],
                marker="|",
                color="blue",
                markersize=8,
                markeredgewidth=1.5,
                linestyle="",
                label="legit values",
            ),
            Line2D(
                [0],
                [0],
                marker="|",
                color="orange",
                markersize=8,
                markeredgewidth=1.5,
                linestyle="",
                label="other peaks",
            ),
            Line2D([0], [0], linestyle="--", color="black", label="threshold"),
        ]

    ax.grid(True)

    # plot horizontal line at the threshold
    threshold = np.nanquantile(dataset.sel(id=station).rainfall.to_numpy(), quantile)
    plt.axhline(y=threshold, color="black", linestyle="--")

    # set xlim and x ticks
    plt.xlim(x[0], x[-1])
    plt.xticks(rotation=45)

    plt.title(name_plot)
    plt.xlabel("time")
    plt.ylabel("rainfall [mm]")
    plt.legend(handles=custom_legend, loc="upper left")
    plt.show()
    plt.close()
    return fig, ax
