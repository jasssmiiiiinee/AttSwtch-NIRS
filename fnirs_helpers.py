"""
Helper and quality-control functions for
AttSwitch fNIRS preprocessing pipeline
"""

import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt

from mne_nirs.preprocessing import (
    scalp_coupling_index_windowed,
    peak_power,
)

from fnirs_parameters import (
    SHORT_SEPARATION_MAX_M,
    SCI_THRESHOLD,
    SCI_L_FREQ,
    SCI_H_FREQ,
    SCI_L_TRANS_BW,
    SCI_H_TRANS_BW,
    WINDOWED_QC_WINDOW_S,
    FILTER_L_FREQ,
    FILTER_H_FREQ,
    FIGURE_DPI,
)



# HELPER AND QC FUNCTIONS


def strip_measurement_suffix(channel_name):
    """
    Return the source-detector pair name.

    Examples:
    'S1_D2 760' -> 'S1_D2'
    'S1_D2 850' -> 'S1_D2'
    'S1_D2 hbo' -> 'S1_D2'
    'S1_D2 hbr' -> 'S1_D2'
    """

    return channel_name.rsplit(" ", 1)[0]


def annotations_to_dataframe(raw):
    """
    Convert LSL annotations stored in the SNIRF file
    into a table for saving as CSV.
    """

    return pd.DataFrame({
        "onset_s": raw.annotations.onset,
        "duration_s": raw.annotations.duration,
        "description": raw.annotations.description,
    })


def build_channel_qc_table(
    raw_od,
    sci,
    raw_distances,
    short_max_m,
    sci_threshold,
):
    """
    Create one QC row for every wavelength channel.
    """

    qc = pd.DataFrame({
        "channel": raw_od.ch_names,

        "pair": [
            strip_measurement_suffix(channel)
            for channel in raw_od.ch_names
        ],

        "distance_mm":
            raw_distances * 1000,

        "SCI":
            sci,
    })

    # identify short-separation wavelength channels
    qc["is_short_separation"] = (
        raw_distances < short_max_m
    )

    # identify wavelength channels below the SCI threshold
    # non-finite SCI values are also treated as poor quality
    qc["below_SCI_threshold"] = (
        ~np.isfinite(qc["SCI"])
        | (qc["SCI"] < sci_threshold)
    )

    # if either wavelength belonging to a source-detector pair fails SCI, mark the entire pair as bad
    qc["pair_marked_bad"] = (
        qc.groupby("pair")[
            "below_SCI_threshold"
        ]
        .transform("any")
    )

    return qc


def build_pair_qc_table(channel_qc):
    """
    Collapse the wavelength-level QC table into one row
    per source-detector pair.
    """

    pair_qc = (
        channel_qc
        .groupby(
            "pair",
            as_index=False,
        )
        .agg(
            distance_mm=(
                "distance_mm",
                "mean",
            ),

            SCI_mean=(
                "SCI",
                "mean",
            ),

            SCI_min=(
                "SCI",
                "min",
            ),

            is_short_separation=(
                "is_short_separation",
                "max",
            ),

            marked_bad=(
                "pair_marked_bad",
                "max",
            ),
        )
    )

    return pair_qc

def build_spatial_qc_tables(
    raw,
    raw_distances,
    pair_qc,
):
    """
    Extract the source, detector, and channel coordinates
    stored in the participant's SNIRF file.

    Creates:
    - one row per unique optode
    - one row per unique source-detector pair

    These coordinates represent the locations stored in
    the SNIRF file. They should not be interpreted as
    participant-specific digitized locations unless the
    optodes were actually digitized for that participant.
    """

    optode_rows = {}
    pair_rows = {}


    # try to identify the coordinate frame stored in MNE
    montage = raw.get_montage()

    if montage is not None:
        montage_positions = montage.get_positions()
        coordinate_frame = montage_positions.get(
            "coord_frame",
            "unknown",
        )

    else:
        coordinate_frame = "unknown"


    for index, (
        channel_name,
        channel_info,
    ) in enumerate(
        zip(
            raw.ch_names,
            raw.info["chs"],
        )
    ):

        # example:
        # "S1_D2 760" -> "S1_D2"
        pair_name = strip_measurement_suffix(
            channel_name
        )

        if "_" not in pair_name:
            continue


        source_name, detector_name = (
            pair_name.split("_", 1)
        )


        # MNE stores coordinates in metres
        midpoint = np.array(
            channel_info["loc"][0:3],
            dtype=float,
        )

        source_position = np.array(
            channel_info["loc"][3:6],
            dtype=float,
        )

        detector_position = np.array(
            channel_info["loc"][6:9],
            dtype=float,
        )


        # save each source only once
        if source_name not in optode_rows:

            optode_rows[source_name] = {
                "optode": source_name,
                "optode_type": "source",
                "x_mm":
                    source_position[0] * 1000,
                "y_mm":
                    source_position[1] * 1000,
                "z_mm":
                    source_position[2] * 1000,
                "coordinate_frame":
                    coordinate_frame,
            }


        # save each detector only once
        if detector_name not in optode_rows:

            optode_rows[detector_name] = {
                "optode": detector_name,
                "optode_type": "detector",
                "x_mm":
                    detector_position[0] * 1000,
                "y_mm":
                    detector_position[1] * 1000,
                "z_mm":
                    detector_position[2] * 1000,
                "coordinate_frame":
                    coordinate_frame,
            }


        # each pair occurs twice in the raw data:
        # once at 760 nm and once at 850 nm
        # save each pair only once
        if pair_name not in pair_rows:

            pair_rows[pair_name] = {
                "pair": pair_name,

                "source":
                    source_name,

                "detector":
                    detector_name,

                "source_x_mm":
                    source_position[0] * 1000,
                "source_y_mm":
                    source_position[1] * 1000,
                "source_z_mm":
                    source_position[2] * 1000,

                "detector_x_mm":
                    detector_position[0] * 1000,
                "detector_y_mm":
                    detector_position[1] * 1000,
                "detector_z_mm":
                    detector_position[2] * 1000,

                "midpoint_x_mm":
                    midpoint[0] * 1000,
                "midpoint_y_mm":
                    midpoint[1] * 1000,
                "midpoint_z_mm":
                    midpoint[2] * 1000,

                "coordinate_frame":
                    coordinate_frame,
            }


    optode_positions = pd.DataFrame(
        optode_rows.values()
    )

    channel_locations = pd.DataFrame(
        pair_rows.values()
    )


    # add the already calculated QC information:
    # distance, SCI, short/long status, and bad-channel status
    channel_locations = (
        channel_locations.merge(
            pair_qc,
            on="pair",
            how="left",
        )
    )


    # sort optodes
    optode_positions["number"] = (
        optode_positions["optode"]
        .str[1:]
        .astype(int)
    )

    optode_positions = (
        optode_positions
        .sort_values(
            [
                "optode_type",
                "number",
            ]
        )
        .drop(
            columns="number"
        )
        .reset_index(
            drop=True
        )
    )


    return (
        optode_positions,
        channel_locations,
        coordinate_frame,
    )


def windowed_scores_to_pair_dataframe(
    raw_od,
    scores,
    times,
    metric_name,
):
    """
    Convert channel-level windowed QC scores into one
    value per source-detector pair per time window.

    The two wavelengths belonging to the same pair receive
    the same windowed SCI/PSP score from MNE-NIRS, so they
    are collapsed into a single pair-level value.
    """

    pair_names = [
        strip_measurement_suffix(channel)
        for channel in raw_od.ch_names
    ]

    unique_pairs = list(
        dict.fromkeys(pair_names)
    )

    rows = []

    for pair in unique_pairs:

        pair_indices = [
            index
            for index, pair_name
            in enumerate(pair_names)
            if pair_name == pair
        ]

        pair_scores = np.nanmean(
            scores[pair_indices, :],
            axis=0,
        )

        for window_index, (
            window_start,
            window_end,
        ) in enumerate(times):

            rows.append({
                "pair": pair,
                "window": window_index + 1,
                "window_start_s": float(window_start),
                "window_end_s": float(window_end),
                metric_name: float(
                    pair_scores[window_index]
                ),
            })

    return pd.DataFrame(
        rows
    )


def plot_windowed_quality_heatmap(
    quality_df,
    metric_name,
    output_file,
    participant_id,
):
    """
    Plot time-resolved SCI or PSP values as a
    source-detector-pair by time heatmap.
    """

    pair_order = list(
        dict.fromkeys(
            quality_df["pair"]
        )
    )

    heatmap = (
        quality_df
        .pivot(
            index="pair",
            columns="window_start_s",
            values=metric_name,
        )
        .reindex(
            pair_order
        )
    )

    matrix = heatmap.to_numpy(
        dtype=float
    )

    window_starts = (
        heatmap.columns
        .to_numpy(
            dtype=float
        )
    )

    window_end = float(
        quality_df[
            "window_end_s"
        ].max()
    )


    # SCI has a defined correlation-like range
    if metric_name == "SCI":

        vmin = -1
        vmax = 1

        colorbar_label = (
            "Windowed SCI\n"
            "higher = stronger wavelength coupling"
        )

        title_metric = "Windowed SCI"

    else:

        # PSP does not have the same fixed scale as SCI
        # use the 99th percentile so isolated extreme values do not dominate the heatmap
        finite_values = matrix[
            np.isfinite(
                matrix
            )
        ]

        vmin = 0

        if finite_values.size > 0:
            vmax = np.nanpercentile(
                finite_values,
                99,
            )
        else:
            vmax = 1

        if vmax <= 0:
            vmax = 1

        colorbar_label = (
            "Windowed PSP\n"
            "higher = stronger cardiac-frequency peak"
        )

        title_metric = "Windowed PSP"


    fig, ax = plt.subplots(
        figsize=(16, 10)
    )


    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        extent=[
            window_starts[0],
            window_end,
            len(pair_order),
            0,
        ],
        vmin=vmin,
        vmax=vmax,
    )


    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Source-detector pair"
    )

    ax.set_title(
        f"Participant {participant_id}: "
        f"{title_metric} "
        f"({WINDOWED_QC_WINDOW_S:.0f}-s windows)"
    )


    # label every fifth pair so the y-axis remains readable
    tick_indices = np.arange(
        0,
        len(pair_order),
        5,
    )

    ax.set_yticks(
        tick_indices + 0.5
    )

    ax.set_yticklabels(
        [
            pair_order[index]
            for index in tick_indices
        ],
        fontsize=7,
    )


    colorbar = fig.colorbar(
        image,
        ax=ax,
        pad=0.02,
    )

    colorbar.set_label(
        colorbar_label
    )


    save_figure(
        fig,
        output_file,
    )


def run_windowed_quality_qc(
    raw_od,
    output_dir,
    participant_id,
):
    """
    Calculate 10-second windowed SCI and PSP as
    time-resolved quality-control metrics.

    These calculations are descriptive QC only.

    A threshold of -np.inf is used so the metrics are
    calculated without rejecting or annotating time windows.
    """

    print("\nWINDOWED SCI + PSP QUALITY CONTROL")


    # windowed SCI

    _, windowed_sci, sci_times = (
        scalp_coupling_index_windowed(
            raw_od,
            time_window=WINDOWED_QC_WINDOW_S,
            threshold=-np.inf,
            l_freq=SCI_L_FREQ,
            h_freq=SCI_H_FREQ,
            l_trans_bandwidth=SCI_L_TRANS_BW,
            h_trans_bandwidth=SCI_H_TRANS_BW,
            verbose=False,
        )
    )


    windowed_sci_df = (
        windowed_scores_to_pair_dataframe(
            raw_od=raw_od,
            scores=windowed_sci,
            times=sci_times,
            metric_name="SCI",
        )
    )


    windowed_sci_df.to_csv(
        output_dir
        / (
            f"{participant_id}_"
            f"windowed_sci_"
            f"{WINDOWED_QC_WINDOW_S:.0f}s.csv"
        ),
        index=False,
    )


    plot_windowed_quality_heatmap(
        quality_df=windowed_sci_df,
        metric_name="SCI",
        output_file=(
            output_dir
            / (
                f"{participant_id}_"
                f"03a_windowed_sci_"
                f"{WINDOWED_QC_WINDOW_S:.0f}s.png"
            )
        ),
        participant_id=participant_id,
    )


    # windowed PSP

    _, windowed_psp, psp_times = (
        peak_power(
            raw_od,
            time_window=WINDOWED_QC_WINDOW_S,
            threshold=-np.inf,
            l_freq=SCI_L_FREQ,
            h_freq=SCI_H_FREQ,
            l_trans_bandwidth=SCI_L_TRANS_BW,
            h_trans_bandwidth=SCI_H_TRANS_BW,
            verbose=False,
        )
    )


    windowed_psp_df = (
        windowed_scores_to_pair_dataframe(
            raw_od=raw_od,
            scores=windowed_psp,
            times=psp_times,
            metric_name="PSP",
        )
    )


    windowed_psp_df.to_csv(
        output_dir
        / (
            f"{participant_id}_"
            f"windowed_psp_"
            f"{WINDOWED_QC_WINDOW_S:.0f}s.csv"
        ),
        index=False,
    )


    plot_windowed_quality_heatmap(
        quality_df=windowed_psp_df,
        metric_name="PSP",
        output_file=(
            output_dir
            / (
                f"{participant_id}_"
                f"03b_windowed_psp_"
                f"{WINDOWED_QC_WINDOW_S:.0f}s.png"
            )
        ),
        participant_id=participant_id,
    )


    print(
        "Window length:",
        WINDOWED_QC_WINDOW_S,
        "seconds"
    )

    print(
        "Number of SCI windows:",
        len(sci_times)
    )

    print(
        "Number of PSP windows:",
        len(psp_times)
    )

    print(
        "No time windows were rejected "
        "or marked bad."
    )


    return (
        windowed_sci_df,
        windowed_psp_df,
    )

def get_good_hb_channels(raw_hb):
    """
    Return lists of HbO and HbR channels that are not
    marked bad.
    """

    bad_channels = set(
        raw_hb.info["bads"]
    )

    channel_types = (
        raw_hb.get_channel_types()
    )

    good_hbo = [
        channel
        for channel, channel_type
        in zip(
            raw_hb.ch_names,
            channel_types,
        )
        if (
            channel_type == "hbo"
            and channel not in bad_channels
        )
    ]

    good_hbr = [
        channel
        for channel, channel_type
        in zip(
            raw_hb.ch_names,
            channel_types,
        )
        if (
            channel_type == "hbr"
            and channel not in bad_channels
        )
    ]

    return good_hbo, good_hbr


def save_figure(
    fig,
    output_file,
):
    """
    Save a matplotlib figure and close it.
    """

    fig.savefig(
        output_file,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_raw_intensity_heatmap(
    raw,
    output_file,
    participant_id,
):
    """
    Plot all raw fNIRS wavelength channels across the
    entire recording as a channel-by-time heatmap.

    Each channel is normalized to its own median intensity
    so temporal changes can be compared across channels.
    """

    # get wavelength stored for every raw channel
    channel_wavelengths = np.array(
        [
            channel_info["loc"][9]
            for channel_info
            in raw.info["chs"]
        ],
        dtype=float,
    )


    # identify the wavelengths present in the recording
    unique_wavelengths = np.sort(
        np.unique(
            channel_wavelengths[
                np.isfinite(
                    channel_wavelengths
                )
            ]
        )
    )


    if len(unique_wavelengths) != 2:
        raise RuntimeError(
            "Expected exactly two fNIRS wavelengths "
            f"but found {unique_wavelengths}."
        )


    # sort channels by wavelength so the heatmap does not depend on the original SNIRF channel order
    wavelength_order = np.argsort(
        channel_wavelengths
    )


    raw_data = (
        raw.get_data()[
            wavelength_order,
            :
        ]
        .astype(float)
    )


    sorted_wavelengths = (
        channel_wavelengths[
            wavelength_order
        ]
    )


    # determine where the first wavelength ends
    wavelength_split = int(
        np.sum(
            sorted_wavelengths
            == unique_wavelengths[0]
        )
    )

    # raw intensity should be positive.
    raw_data[
        raw_data <= 0
    ] = np.nan

    # median intensity for each individual channel
    channel_medians = np.nanmedian(
        raw_data,
        axis=1,
        keepdims=True,
    )

    channel_medians[
        channel_medians == 0
    ] = np.nan

    # express each channel relative to its own median
    relative_intensity = (
        raw_data
        / channel_medians
    )

    # log-transform so the channel median is centered at 0
    heatmap_data = np.log10(
        relative_intensity
    )

    finite_values = heatmap_data[
        np.isfinite(
            heatmap_data
        )
    ]

    if finite_values.size == 0:
        warnings.warn(
            "Raw intensity heatmap could not be created "
            "because no finite intensity values were found."
        )
        return

    # limit influence of isolated extreme values
    absolute_limit = np.nanpercentile(
        np.abs(finite_values),
        99,
    )

    if absolute_limit == 0:
        absolute_limit = 1

    # create figure with extra room on the right for the colour bar and interpretation text
    fig, ax = plt.subplots(
        figsize=(18, 10)
    )

    fig.subplots_adjust(
        right=0.82
    )

    image = ax.imshow(
        heatmap_data,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        extent=[
            raw.times[0],
            raw.times[-1],
            len(raw.ch_names),
            0,
        ],
        vmin=-absolute_limit,
        vmax=absolute_limit,
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Raw fNIRS channel "
        "(grouped by wavelength)"
    )
        
    ax.set_title(
        f"Participant {participant_id}: "
        "Raw intensity across all channels"
        "(uncropped acquisition)"
    )

    # separate 760 nm and 850 nm wavelength channels
    ax.axhline(
        wavelength_split,
        color="white",
        linestyle="--",
        linewidth=1.5,
    )

    # label the two wavelength sections outside the heatmap
    n_channels = len(
        sorted_wavelengths
    )


    # vertical centers of each wavelength group in axis-relative coordinates
    first_wavelength_center = (
        1
        - (
            wavelength_split / 2
        )
        / n_channels
    )

    second_wavelength_center = (
        1
        - (
            wavelength_split
            + (
                n_channels
                - wavelength_split
            ) / 2
        )
        / n_channels
    )


    ax.text(
        1.015,
        first_wavelength_center,
        f"{unique_wavelengths[0]:.0f} nm",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="center",
        clip_on=False,
    )


    ax.text(
        1.015,
        second_wavelength_center,
        f"{unique_wavelengths[1]:.0f} nm",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="center",
        clip_on=False,
    )

    # colour bar
    colorbar = fig.colorbar(
        image,
        ax=ax,
        pad=0.10,
    )

    colorbar.set_label(
        "Log10 intensity relative to channel median"
    )

    # interpretation guide
    fig.text(
        0.91,
        0.62,
        (
            "HOW TO READ\n\n"
            "Yellow = above channel median\n\n"
            "Teal = near channel median\n\n"
            "Purple = below channel median"
        ),
        fontsize=10,
        ha="left",
        va="top",
    )

    # save only. do not display figure on screen
    fig.savefig(
        output_file,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Raw intensity heatmap saved:",
        output_file
    )


def plot_distance_distribution(
    pair_qc,
    output_file,
    participant_id,
):
    """
    Plot the distribution of source-detector distances.
    """

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.hist(
        pair_qc["distance_mm"],
        bins=20,
    )

    ax.axvline(
        SHORT_SEPARATION_MAX_M * 1000,
        linestyle="--",
        label=(
            f"Short-separation cutoff = "
            f"{SHORT_SEPARATION_MAX_M * 1000:.0f} mm"
        ),
    )

    n_short = int(
        pair_qc["is_short_separation"].sum()
    )

    n_long = int(
        (
            ~pair_qc["is_short_separation"]
        ).sum()
    )

    long_distances = pair_qc.loc[
        ~pair_qc["is_short_separation"],
        "distance_mm",
    ]

    ax.text(
        0.98,
        0.95,
        (
            f"Short pairs: {n_short}\n"
            f"Long pairs: {n_long}\n"
            f"Long range: "
            f"{long_distances.min():.1f}–"
            f"{long_distances.max():.1f} mm"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    ax.set_xlabel(
        "Source-detector distance (mm)"
    )

    ax.set_ylabel(
        "Number of source-detector pairs"
    )

    ax.set_title(
        f"Participant {participant_id}: "
        "Source-detector distances"
    )

    ax.legend()

    save_figure(
        fig,
        output_file,
    )



def plot_sci_by_pair(
    pair_qc,
    output_file,
    participant_id,
):
    """
    Plot the minimum SCI value for every source-detector
    pair, ordered from lowest to highest.
    """

    ordered = (
        pair_qc
        .sort_values(
            "SCI_min"
        )
        .reset_index(
            drop=True
        )
    )

    x = np.arange(
        len(ordered)
    )

    fig, ax = plt.subplots(
        figsize=(20, 6)
    )

    short_mask = (
        ordered["is_short_separation"]
        .astype(bool)
        .to_numpy()
    )

    long_mask = ~short_mask

    ax.plot(
        x[long_mask],
        ordered.loc[long_mask, "SCI_min"],
        marker="o",
        linestyle="none",
        markersize=4,
        color="blue",
        label="Long-separation pair",
    )

    ax.plot(
        x[short_mask],
        ordered.loc[short_mask, "SCI_min"],
        marker="o",
        linestyle="none",
        markersize=4,
        color="red",
        label="Short-separation pair",
    )

    ax.axhline(
        SCI_THRESHOLD,
        linestyle="--",
        label=f"SCI cutoff = {SCI_THRESHOLD:.2f}",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        ordered["pair"],
        rotation=90,
        fontsize=6,
    )

    ax.set_xlabel(
        "Source-detector pair"
    )

    ax.set_ylabel(
        "Minimum SCI"
    )

    ax.set_title(
        f"Participant {participant_id}: "
        "SCI by source-detector pair"
    )

    ax.legend()

    save_figure(
        fig,
        output_file,
    )

def plot_tddr_heatmaps(
    raw_od,
    raw_od_tddr,
    channel_qc,
    output_file,
    participant_id,
):
    """
    Compare all usable optical-density channels before
    and after TDDR across the full recording.

    Panel 1:
    Before TDDR.

    Panel 2:
    After TDDR.

    Panel 3:
    Difference (After - Before), showing exactly where
    TDDR changed the signal.

    Bad SCI pairs are excluded.

    Channels are grouped by wavelength.

    For visualization only, all panels use the PRE-TDDR
    channel median and standard deviation for scaling.
    """

    # use all wavelength channels belonging to pairs that passed the global SCI criterion
    good_channels = (
        channel_qc.loc[
            ~channel_qc["pair_marked_bad"],
            "channel",
        ]
        .tolist()
    )

    good_channels = [
        channel
        for channel in good_channels
        if channel in raw_od.ch_names
    ]

    if len(good_channels) == 0:
        warnings.warn(
            "No usable channels were available "
            "for the TDDR heatmap."
        )
        return


    # read wavelength metadata and sort channels by wavelength
    channel_info = []

    for channel in good_channels:

        channel_index = (
            raw_od.ch_names.index(
                channel
            )
        )

        wavelength = float(
            raw_od.info["chs"][
                channel_index
            ]["loc"][9]
        )

        channel_info.append(
            (
                wavelength,
                channel,
            )
        )


    channel_info = sorted(
        channel_info,
        key=lambda item: (
            item[0],
            item[1],
        ),
    )


    ordered_channels = [
        channel
        for wavelength, channel
        in channel_info
    ]

    ordered_wavelengths = np.array(
        [
            wavelength
            for wavelength, channel
            in channel_info
        ],
        dtype=float,
    )


    unique_wavelengths = np.sort(
        np.unique(
            ordered_wavelengths
        )
    )


    # get identical channels before and after TDDR
    before = raw_od.get_data(
        picks=ordered_channels
    )

    after = raw_od_tddr.get_data(
        picks=ordered_channels
    )


    # use PRE-TDDR center for both datasets
    pre_tddr_center = np.median(
        before,
        axis=1,
        keepdims=True,
    )

    before_centered = (
        before
        - pre_tddr_center
    )

    after_centered = (
        after
        - pre_tddr_center
    )


    # use PRE-TDDR SD for both datasets
    pre_tddr_scale = np.std(
        before_centered,
        axis=1,
        keepdims=True,
    )

    pre_tddr_scale[
        pre_tddr_scale == 0
    ] = 1


    before_plot = (
        before_centered
        / pre_tddr_scale
    )

    after_plot = (
        after_centered
        / pre_tddr_scale
    )


    # calculate what TDDR changed
    difference_plot = (
        after_plot
        - before_plot
    )


    # common scale for BEFORE and AFTER
    combined_values = np.concatenate(
        [
            before_plot.ravel(),
            after_plot.ravel(),
        ]
    )

    finite_values = combined_values[
        np.isfinite(
            combined_values
        )
    ]

    if finite_values.size == 0:
        warnings.warn(
            "TDDR heatmap could not be created "
            "because no finite values were found."
        )
        return


    colour_limit = np.nanpercentile(
        np.abs(finite_values),
        99,
    )

    if colour_limit == 0:
        colour_limit = 1


    # separate scale for the DIFFERENCE panel because TDDR-related changes may be much smaller
    difference_values = difference_plot[
        np.isfinite(
            difference_plot
        )
    ]

    difference_limit = np.nanpercentile(
        np.abs(difference_values),
        99,
    )

    if difference_limit == 0:
        difference_limit = 1


    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 14),
        sharex=True,
        sharey=True,
    )


    # BEFORE TDDR
    image_before = axes[0].imshow(
        before_plot,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        extent=[
            raw_od.times[0],
            raw_od.times[-1],
            len(ordered_channels),
            0,
        ],
        vmin=-colour_limit,
        vmax=colour_limit,
        cmap="RdBu_r",
    )

    axes[0].set_title(
        "Before TDDR"
    )


    # AFTER TDDR
    axes[1].imshow(
        after_plot,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        extent=[
            raw_od.times[0],
            raw_od.times[-1],
            len(ordered_channels),
            0,
        ],
        vmin=-colour_limit,
        vmax=colour_limit,
        cmap="RdBu_r",
    )

    axes[1].set_title(
        "After TDDR"
    )


    # DIFFERENCE
    image_difference = axes[2].imshow(
        difference_plot,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        extent=[
            raw_od.times[0],
            raw_od.times[-1],
            len(ordered_channels),
            0,
        ],
        vmin=-difference_limit,
        vmax=difference_limit,
        cmap="RdBu_r",
    )

    axes[2].set_title(
        "Change produced by TDDR "
        "(After - Before)"
    )


    for ax in axes:
        ax.set_ylabel(
            "Usable OD channel"
        )


    # separate wavelength groups
    if len(unique_wavelengths) == 2:

        wavelength_split = int(
            np.sum(
                ordered_wavelengths
                == unique_wavelengths[0]
            )
        )

        for ax in axes:

            ax.axhline(
                wavelength_split,
                color="black",
                linestyle="--",
                linewidth=1.0,
            )


    axes[2].set_xlabel(
        "Time (s)"
    )


    # colour bar for raw before/after comparison
    colorbar_signal = fig.colorbar(
        image_before,
        ax=axes[:2],
        pad=0.02,
    )

    colorbar_signal.set_label(
        "OD relative to pre-TDDR channel SD"
    )


    # separate colour bar for TDDR-induced change
    colorbar_difference = fig.colorbar(
        image_difference,
        ax=axes[2],
        pad=0.02,
    )

    colorbar_difference.set_label(
        "Change produced by TDDR "
        "(pre-TDDR SD units)"
    )


    fig.suptitle(
        f"Participant {participant_id}: "
        "TDDR whole-recording comparison"
    )


    save_figure(
        fig,
        output_file,
    )

def plot_tddr_motion_summary(
    raw_od,
    raw_od_tddr,
    channel_qc,
    output_file,
    participant_id,
):
    """
    Quantify motion-sensitive temporal derivatives before
    and after TDDR across the entire recording.

    TDDR operates on the <0.5-Hz component of the signal.
    Therefore these QC metrics are calculated on that same
    low-frequency component rather than on the full signal.

    Panel 1:
    Low-frequency GVTD before and after TDDR.

    Panel 2:
    For every usable OD channel, compare the 99.9th
    percentile of the absolute low-frequency temporal
    derivative before versus after TDDR.
    """

    good_channels = (
        channel_qc.loc[
            ~channel_qc["pair_marked_bad"],
            "channel",
        ]
        .tolist()
    )

    good_channels = [
        channel
        for channel in good_channels
        if channel in raw_od.ch_names
    ]


    if len(good_channels) == 0:
        warnings.warn(
            "No usable channels were available "
            "for the TDDR motion summary."
        )
        return


    before = raw_od.get_data(
        picks=good_channels
    )

    after = raw_od_tddr.get_data(
        picks=good_channels
    )


    sfreq = float(
        raw_od.info["sfreq"]
    )

    # EXTRACT THE <0.5-Hz COMPONENT
    # matches the filtering step used internally by MNE: 3rd-order Butterworth low-pass at 0.5 Hz

    tddr_cutoff_hz = 0.5
    tddr_filter_order = 3

    normalized_cutoff = (
        tddr_cutoff_hz
        * 2
        / sfreq
    )


    if normalized_cutoff < 1:

        b, a = butter(
            tddr_filter_order,
            normalized_cutoff,
        )


        before_low = np.empty_like(
            before
        )

        after_low = np.empty_like(
            after
        )


        for channel_index in range(
            before.shape[0]
        ):

            before_signal = (
                before[channel_index]
                - np.mean(
                    before[channel_index]
                )
            )

            after_signal = (
                after[channel_index]
                - np.mean(
                    after[channel_index]
                )
            )


            before_low[channel_index] = (
                filtfilt(
                    b,
                    a,
                    before_signal,
                    padlen=0,
                )
            )

            after_low[channel_index] = (
                filtfilt(
                    b,
                    a,
                    after_signal,
                    padlen=0,
                )
            )

    else:

        before_low = before.copy()
        after_low = after.copy()

    # TEMPORAL DERIVATIVES
    derivative_before = (
        np.diff(
            before_low,
            axis=1,
        )
        * sfreq
    )

    derivative_after = (
        np.diff(
            after_low,
            axis=1,
        )
        * sfreq
    )


    derivative_times = (
        raw_od.times[1:]
    )

    # GLOBAL VARIANCE OF TEMPORAL DERIVATIVES

    gvtd_before = np.sqrt(
        np.mean(
            derivative_before ** 2,
            axis=0,
        )
    )

    gvtd_after = np.sqrt(
        np.mean(
            derivative_after ** 2,
            axis=0,
        )
    )

    # EXTREME DERIVATIVE PER CHANNEL
    # 99.9th percentile targets rare extreme changes rather than ordinary temporal variability

    derivative_percentile = 99.9


    channel_extreme_before = (
        np.nanpercentile(
            np.abs(
                derivative_before
            ),
            derivative_percentile,
            axis=1,
        )
    )

    channel_extreme_after = (
        np.nanpercentile(
            np.abs(
                derivative_after
            ),
            derivative_percentile,
            axis=1,
        )
    )


    gvtd_extreme_before = float(
        np.nanpercentile(
            gvtd_before,
            derivative_percentile,
        )
    )

    gvtd_extreme_after = float(
        np.nanpercentile(
            gvtd_after,
            derivative_percentile,
        )
    )


    # percentage change
    if gvtd_extreme_before > 0:

        gvtd_percent_change = (
            (
                gvtd_extreme_after
                - gvtd_extreme_before
            )
            / gvtd_extreme_before
            * 100
        )

    else:

        gvtd_percent_change = np.nan


    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 10),
    )

    # PANEL 1: LOW-FREQUENCY GVTD

    axes[0].plot(
        derivative_times,
        gvtd_before,
        label="Before TDDR",
    )

    axes[0].plot(
        derivative_times,
        gvtd_after,
        label="After TDDR",
    )

    axes[0].set_xlabel(
        "Time (s)"
    )

    axes[0].set_ylabel(
        "Low-frequency GVTD (OD/s)"
    )

    axes[0].set_title(
        "Global low-frequency "
        "temporal-derivative signal (<0.5 Hz)"
    )

    axes[0].legend()


    axes[0].text(
        0.99,
        0.95,
        (
            f"{derivative_percentile:.1f}th "
            "percentile GVTD\n"
            f"Before: "
            f"{gvtd_extreme_before:.4f} OD/s\n"
            f"After: "
            f"{gvtd_extreme_after:.4f} OD/s\n"
            f"Change: "
            f"{gvtd_percent_change:+.1f}%"
        ),
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    # PANEL 2: CHANNEL-BY-CHANNEL COMPARISON

    axes[1].scatter(
        channel_extreme_before,
        channel_extreme_after,
        s=18,
        alpha=0.7,
    )


    scatter_limit = float(
        np.nanmax(
            [
                np.nanmax(
                    channel_extreme_before
                ),
                np.nanmax(
                    channel_extreme_after
                ),
            ]
        )
    )


    if scatter_limit <= 0:
        scatter_limit = 1


    scatter_limit *= 1.05


    axes[1].plot(
        [
            0,
            scatter_limit,
        ],
        [
            0,
            scatter_limit,
        ],
        linestyle="--",
        label="No change",
    )


    axes[1].set_xlim(
        0,
        scatter_limit,
    )

    axes[1].set_ylim(
        0,
        scatter_limit,
    )


    axes[1].set_xlabel(
        "Before TDDR: "
        "99.9th percentile "
        "|low-frequency derivative| (OD/s)"
    )

    axes[1].set_ylabel(
        "After TDDR: "
        "99.9th percentile "
        "|low-frequency derivative| (OD/s)"
    )

    axes[1].set_title(
        "Extreme low-frequency "
        "temporal derivatives by channel"
    )

    axes[1].legend()


    fig.suptitle(
        f"Participant {participant_id}: "
        "TDDR motion-correction summary"
    )


    save_figure(
        fig,
        output_file,
    )

def plot_filtering_comparison(
    raw_before,
    raw_after,
    output_file,
    participant_id,
):
    """
    Compare mean HbO and HbR signals before and after
    temporal filtering.

    The same good long-separation channels are used
    before and after filtering.

    Both panels use identical y-axis limits so the
    change in signal magnitude can be compared fairly.
    """

    before_hbo, before_hbr = (
        get_good_hb_channels(
            raw_before
        )
    )

    after_hbo, after_hbr = (
        get_good_hb_channels(
            raw_after
        )
    )


    # use only channels available at BOTH stages
    after_hbo_set = set(
        after_hbo
    )

    after_hbr_set = set(
        after_hbr
    )

    good_hbo = [
        channel
        for channel in before_hbo
        if channel in after_hbo_set
    ]

    good_hbr = [
        channel
        for channel in before_hbr
        if channel in after_hbr_set
    ]


    if (
        len(good_hbo) == 0
        or len(good_hbr) == 0
    ):
        warnings.warn(
            "No common good HbO/HbR channels "
            "were available for the filtering "
            "comparison figure."
        )
        return


    # mean signals before filtering
    mean_hbo_before = (
        raw_before
        .get_data(
            picks=good_hbo
        )
        .mean(axis=0)
        * 1e6
    )

    mean_hbr_before = (
        raw_before
        .get_data(
            picks=good_hbr
        )
        .mean(axis=0)
        * 1e6
    )


    # mean signals after filtering
    mean_hbo_after = (
        raw_after
        .get_data(
            picks=good_hbo
        )
        .mean(axis=0)
        * 1e6
    )

    mean_hbr_after = (
        raw_after
        .get_data(
            picks=good_hbr
        )
        .mean(axis=0)
        * 1e6
    )


    # determine ONE y-axis range from all four signals
    all_values = np.concatenate(
        [
            mean_hbo_before,
            mean_hbr_before,
            mean_hbo_after,
            mean_hbr_after,
        ]
    )

    finite_values = all_values[
        np.isfinite(
            all_values
        )
    ]

    if finite_values.size == 0:
        warnings.warn(
            "Filtering comparison could not be "
            "created because no finite values "
            "were found."
        )
        return


    y_min = float(
        np.min(
            finite_values
        )
    )

    y_max = float(
        np.max(
            finite_values
        )
    )

    y_range = (
        y_max
        - y_min
    )

    if y_range == 0:
        y_range = 1


    # small padding around the shared y-axis
    y_padding = (
        0.05
        * y_range
    )

    y_limits = (
        y_min - y_padding,
        y_max + y_padding,
    )


    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        sharey=True,
    )


    # BEFORE FILTER
    axes[0].plot(
        raw_before.times,
        mean_hbo_before,
        label="HbO",
    )

    axes[0].plot(
        raw_before.times,
        mean_hbr_before,
        label="HbR",
    )

    axes[0].set_title(
        "Before temporal filtering"
    )

    axes[0].set_ylabel(
        "Mean concentration change (µM)"
    )

    axes[0].set_ylim(
        y_limits
    )

    axes[0].legend()


    # AFTER FILTER
    axes[1].plot(
        raw_after.times,
        mean_hbo_after,
        label="HbO",
    )

    axes[1].plot(
        raw_after.times,
        mean_hbr_after,
        label="HbR",
    )

    axes[1].set_title(
        f"After "
        f"{FILTER_L_FREQ:.2f}–"
        f"{FILTER_H_FREQ:.2f} Hz filtering"
    )

    axes[1].set_xlabel(
        "Time (s)"
    )

    axes[1].set_ylabel(
        "Mean concentration change (µM)"
    )

    axes[1].set_ylim(
        y_limits
    )


    fig.suptitle(
        f"Participant {participant_id}: "
        "HbO/HbR filtering comparison"
    )


    save_figure(
        fig,
        output_file,
    )

def plot_psd_filtering_comparison(
    raw_before,
    raw_after,
    output_file,
    participant_id,
):
    """
    Compare mean HbO and HbR power spectral density
    before and after temporal filtering.

    HbO and HbR are shown in separate panels because
    their PSD magnitudes may differ.

    The retained filter passband is shaded.
    """

    before_hbo, before_hbr = (
        get_good_hb_channels(
            raw_before
        )
    )

    after_hbo, after_hbr = (
        get_good_hb_channels(
            raw_after
        )
    )


    # use only channels available at BOTH stages
    after_hbo_set = set(
        after_hbo
    )

    after_hbr_set = set(
        after_hbr
    )

    good_hbo = [
        channel
        for channel in before_hbo
        if channel in after_hbo_set
    ]

    good_hbr = [
        channel
        for channel in before_hbr
        if channel in after_hbr_set
    ]


    if (
        len(good_hbo) == 0
        or len(good_hbr) == 0
    ):
        warnings.warn(
            "No common good HbO/HbR channels "
            "were available for the PSD comparison."
        )
        return


    nyquist = (
        raw_before.info["sfreq"]
        / 2
    )

    fmax = min(
        1.25,
        nyquist * 0.99,
    )


    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        sharex=True,
    )


    for (
        ax,
        channel_type,
        channels,
        title,
    ) in [
        (
            axes[0],
            "hbo",
            good_hbo,
            "HbO",
        ),
        (
            axes[1],
            "hbr",
            good_hbr,
            "HbR",
        ),
    ]:

        psd_before = (
            raw_before.compute_psd(
                method="welch",
                fmin=0,
                fmax=fmax,
                picks=channels,
                verbose=False,
            )
        )

        psd_after = (
            raw_after.compute_psd(
                method="welch",
                fmin=0,
                fmax=fmax,
                picks=channels,
                verbose=False,
            )
        )


        before_data, before_freqs = (
            psd_before.get_data(
                return_freqs=True
            )
        )

        after_data, after_freqs = (
            psd_after.get_data(
                return_freqs=True
            )
        )


        mean_before = np.mean(
            before_data,
            axis=0,
        )

        mean_after = np.mean(
            after_data,
            axis=0,
        )


        mean_before_db = (
            10
            * np.log10(
                np.maximum(
                    mean_before,
                    np.finfo(float).tiny,
                )
            )
        )

        mean_after_db = (
            10
            * np.log10(
                np.maximum(
                    mean_after,
                    np.finfo(float).tiny,
                )
            )
        )


        ax.plot(
            before_freqs,
            mean_before_db,
            label="Before filter",
        )

        ax.plot(
            after_freqs,
            mean_after_db,
            label="After filter",
        )


        # show the frequency range intended to be retained
        ax.axvspan(
            FILTER_L_FREQ,
            FILTER_H_FREQ,
            alpha=0.12,
            label=(
                f"Retained passband "
                f"({FILTER_L_FREQ:.2f}–"
                f"{FILTER_H_FREQ:.2f} Hz)"
            ),
        )


        # mark exact filter cutoffs
        ax.axvline(
            FILTER_L_FREQ,
            linestyle="--",
            linewidth=1,
        )

        ax.axvline(
            FILTER_H_FREQ,
            linestyle="--",
            linewidth=1,
        )


        ax.set_ylabel(
            "Power spectral density (dB)"
        )

        ax.set_title(
            f"{title} PSD"
        )

        ax.legend()


    axes[1].set_xlabel(
        "Frequency (Hz)"
    )


    fig.suptitle(
        f"Participant {participant_id}: "
        "PSD before vs after temporal filtering"
    )


    save_figure(
        fig,
        output_file,
    )


def plot_event_timeline(
    raw,
    output_file,
    participant_id,
):
    """
    Plot the timing of each LSL event code.
    """

    if len(raw.annotations) == 0:
        warnings.warn(
            "No annotations were available "
            "for the event timeline."
        )
        return

    descriptions = np.unique(
        raw.annotations.description
    )

    description_to_y = {
        description: index
        for index, description
        in enumerate(descriptions)
    }

    y_values = [
        description_to_y[description]
        for description
        in raw.annotations.description
    ]

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    ax.scatter(
        raw.annotations.onset,
        y_values,
    )

    ax.set_yticks(
        range(
            len(descriptions)
        )
    )

    ax.set_yticklabels(
        descriptions
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Event code"
    )

    ax.set_title(
        f"Participant {participant_id}: "
        "LSL event timeline"
    )

    save_figure(
        fig,
        output_file,
    )


def plot_filtered_with_event_lines(
    raw_hb,
    output_file,
    participant_id,
):
    """
    Plot mean cleaned HbO and HbR signals with
    LSL event markers shown as vertical lines.
    """

    good_hbo, good_hbr = (
        get_good_hb_channels(
            raw_hb
        )
    )

    if (
        len(good_hbo) == 0
        or len(good_hbr) == 0
    ):
        warnings.warn(
            "No good HbO/HbR channels were available "
            "for the event-overlay figure."
        )
        return

    mean_hbo = (
        raw_hb
        .get_data(
            picks=good_hbo
        )
        .mean(axis=0)
        * 1e6
    )

    mean_hbr = (
        raw_hb
        .get_data(
            picks=good_hbr
        )
        .mean(axis=0)
        * 1e6
    )

    fig, ax = plt.subplots(
        figsize=(16, 5)
    )

    ax.plot(
        raw_hb.times,
        mean_hbo,
        label="HbO",
    )

    ax.plot(
        raw_hb.times,
        mean_hbr,
        label="HbR",
    )

    # show different experimental transitions separately
    # codes:
    # 1 = baseline rest start
    # 2 = rest after face
    # 3 = rest after house
    # 4 = face task start
    # 5 = house task start
    # 6 = final task end

    task_label_added = False
    rest_label_added = False
    baseline_label_added = False
    end_label_added = False

    for onset, description in zip(
        raw_hb.annotations.onset,
        raw_hb.annotations.description,
    ):

        code = str(description)


        # rest -> task transitions
        if code in {"4", "5"}:

            ax.axvline(
                onset,
                color="green",
                linestyle="--",
                linewidth=1.2,
                alpha=0.45,
                label=(
                    "Task onset"
                    if not task_label_added
                    else None
                ),
            )

            task_label_added = True


        # task -> rest transitions
        elif code in {"2", "3"}:

            ax.axvline(
                onset,
                color="purple",
                linestyle=":",
                linewidth=1.2,
                alpha=0.45,
                label=(
                    "Rest onset"
                    if not rest_label_added
                    else None
                ),
            )

            rest_label_added = True


        # initial baseline
        elif code == "1":

            ax.axvline(
                onset,
                color="black",
                linestyle="-.",
                linewidth=1.4,
                alpha=0.70,
                label=(
                    "Baseline start"
                    if not baseline_label_added
                    else None
                ),
            )

            baseline_label_added = True


        # final task end
        elif code == "6":

            ax.axvline(
                onset,
                color="black",
                linestyle="-",
                linewidth=1.4,
                alpha=0.70,
                label=(
                    "Final task end"
                    if not end_label_added
                    else None
                ),
            )

            end_label_added = True

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Mean concentration change (µM)"
    )

    ax.set_title(
        f"Participant {participant_id}: "
        "Filtered HbO/HbR with LSL events"
    )

    ax.legend()

    save_figure(
        fig,
        output_file,
    )


