"""
AttSwitch fNIRS preprocessing pipeline

preprocesses one participant at a time from raw SNIRF data to cleaned continuous HbO/HbR data for analysis
(participant-specific file paths are stored in local_config.py)

PIPELINE
1. load SNIRF data and LSL event annotations
2. save raw-intensity quality-control info
3. crop recording at the final task-end marker
4. convert raw intensity to optical density
5. signal quality - scalp coupling index (SCI)
6. mark SCI<0.70 source-detector pairs as bad
7. TDDR motion correction
8. short-separation regression
9. convert optical density to HbO/HbR - Beer-Lambert law
10. remove short-separation channels from cortical analysis copy
11. apply 0.01-0.20 Hz band-pass filter
12. save cleaned data, QC tables, figures, and preprocessing summaries

"""


from pathlib import Path
import json
import warnings

import mne
import mne_nirs
import numpy as np
import pandas as pd

from mne.preprocessing.nirs import (
    source_detector_distances,
    optical_density,
    scalp_coupling_index,
    temporal_derivative_distribution_repair,
    beer_lambert_law,
)

from mne_nirs.signal_enhancement import (
    short_channel_regression,
)


# PARTICIPANT CONFIGURATION

from local_config import (
    PARTICIPANT_ID,
    SNIRF_FILE,
    OUTPUT_ROOT,
)


# PREPROCESSING PARAMETERS

from fnirs_parameters import (
    SHORT_SEPARATION_MAX_M,
    EXPECTED_N_SOURCES,
    EXPECTED_N_DETECTORS,
    EXPECTED_N_PAIRS,
    EXPECTED_N_SHORT_PAIRS,
    SCI_THRESHOLD,
    SCI_L_FREQ,
    SCI_H_FREQ,
    SCI_L_TRANS_BW,
    SCI_H_TRANS_BW,
    WINDOWED_QC_WINDOW_S,
    CROP_AT_FINAL_TASK_END,
    FINAL_TASK_END_CODE,
    PPF,
    FILTER_L_FREQ,
    FILTER_H_FREQ,
    FILTER_ORDER,
    FIGURE_DPI,
    SAVE_INTERMEDIATES,
)


# HELPER AND QC FUNCTIONS

from fnirs_helpers import (
    strip_measurement_suffix,
    annotations_to_dataframe,
    build_channel_qc_table,
    build_pair_qc_table,
    build_spatial_qc_tables,
    run_windowed_quality_qc,
    get_good_hb_channels,
    plot_raw_intensity_heatmap,
    plot_distance_distribution,
    plot_sci_by_pair,
    plot_tddr_heatmaps,
    plot_tddr_motion_summary,
    plot_filtering_comparison,
    plot_psd_filtering_comparison,
    plot_event_timeline,
    plot_filtered_with_event_lines,
)

 
# PREPROCESSING

def preprocess_participant(
    participant_id,
    snirf_file,
    output_root,
):
    """
    Preprocess one participant's continuous fNIRS recording.

    Loads raw SNIRF data, performs signal-quality checks and
    preprocessing, saves cleaned HbO/HbR data and QC outputs,
    and returns the cleaned MNE Raw object and summary.
    """

    print(f"PREPROCESSING PARTICIPANT {participant_id}")

    # convert file paths to path objects
    snirf_file = Path(snirf_file)
    output_dir = Path(output_root)

    # 1. CHECK INPUT AND OUTPUT PATHS

    if not snirf_file.exists():
        raise FileNotFoundError(
            f"SNIRF file not found: {snirf_file}"
        )

    if not output_dir.exists():
        raise FileNotFoundError(
            f"PREPROCESSED folder not found: {output_dir}"
        )

    # 2. LOAD RAW SNIRF DATA

    raw = mne.io.read_raw_snirf(
        snirf_file,
        preload=True,
    )

    # save original recording duration 

    original_recording_duration_s = float(
        raw.times[-1]
    )

    # check that sampling rate can support the cardiac-frequency SCI / PSP filter

    nyquist_hz = (
        raw.info["sfreq"]
        / 2
    )

    required_nyquist_hz = (
        SCI_H_FREQ
        + SCI_H_TRANS_BW
    )

    if nyquist_hz <= required_nyquist_hz:
        raise RuntimeError(
            f"Sampling rate is {raw.info['sfreq']:.3f} Hz, "
            f"giving a Nyquist frequency of {nyquist_hz:.3f} Hz. "
            f"The current SCI/PSP settings require a Nyquist "
            f"frequency above {required_nyquist_hz:.3f} Hz. "
            "Use sampling-rate-specific SCI/PSP settings "
            "for this recording."
        )

    print("\nRAW DATA")
    print(raw)

    print(
        "Sampling rate:",
        raw.info["sfreq"],
        "Hz"
    )

    print(
        "Recording duration:",
        raw.times[-1],
        "seconds"
    )

    print(
        "Number of raw wavelength channels:",
        len(raw.ch_names)
    )

    wavelengths = []

    for ch in raw.info["chs"]:
        wavelength = ch["loc"][9]

        if (
            np.isfinite(wavelength)
            and wavelength not in wavelengths
        ):
            wavelengths.append(wavelength)

    print(
        "Wavelength order in file:",
        wavelengths
    )

    print(
        "PPF values:",
        PPF
    )

    # 3. RAW INTENSITY QC

    print("\nRAW INTENSITY QC")

    plot_raw_intensity_heatmap(
        raw,
        output_dir
        / f"{participant_id}_00_raw_intensity_heatmap.png",
        participant_id,
    )

    # 4. CHECK LSL EVENT MARKERS

    print("\nEVENT MARKERS")

    print(
        "Number of annotations:",
        len(raw.annotations)
    )

    print(
        "Event descriptions:",
        np.unique(raw.annotations.description)
    )

    # stop preprocessing if no LSL markers are found

    if len(raw.annotations) == 0:
        raise RuntimeError(
            "No event markers were found in the SNIRF file. "
            "Check LSL recording before continuing."
        )

    # save event-marker information

    events_df = annotations_to_dataframe(
        raw
    )

    events_df.to_csv(
        output_dir
        / f"{participant_id}_events.csv",
        index=False,
    )

    # CROP NON-TASK DATA BEFORE PREPROCESSING

    crop_end_s = original_recording_duration_s


    if CROP_AT_FINAL_TASK_END:

        final_task_end_onsets = (
            raw.annotations.onset[
                raw.annotations.description
                == FINAL_TASK_END_CODE
            ]
        )

        if len(final_task_end_onsets) == 0:

            raise RuntimeError(
                f"No LSL marker with code "
                f"{FINAL_TASK_END_CODE} was found. "
                "Cannot determine the final task end "
                "for cropping."
            )


        if len(final_task_end_onsets) > 1:

            warnings.warn(
                f"More than one marker with code "
                f"{FINAL_TASK_END_CODE} was found. "
                "The last occurrence will be used "
                "as the task end."
            )


        crop_end_s = float(
            final_task_end_onsets[-1]
        )


        print("\nCROPPING")

        print(
            "Original recording duration:",
            original_recording_duration_s,
            "seconds"
        )

        print(
            "Final task-end marker:",
            crop_end_s,
            "seconds"
        )


        raw.crop(
            tmin=0,
            tmax=crop_end_s,
        )


        print(
            "Cropped recording duration:",
            raw.times[-1],
            "seconds"
        )

    # 5. CHECK SOURCE-DETECTOR DISTANCES

    raw_distances = source_detector_distances(
        raw.info
    )

    short_raw_mask = (
        raw_distances
        < SHORT_SEPARATION_MAX_M
    )

    print("\nSOURCE-DETECTOR GEOMETRY")

    print(
        "Minimum distance:",
        np.min(raw_distances) * 1000,
        "mm"
    )

    print(
        "Maximum distance:",
        np.max(raw_distances) * 1000,
        "mm"
    )

    print(
        "Short-separation wavelength channels:",
        int(short_raw_mask.sum())
    )

    # 6. CONVERT RAW INTENSITY TO OPTICAL DENSITY

    raw_od = optical_density(
        raw.copy()
    )

    print("\nOPTICAL DENSITY")

    print(
        "Channel types:",
        set(raw_od.get_channel_types())
    )

    # 7. SCALP COUPLING INDEX (SCI)

    sci = scalp_coupling_index(
        raw_od,
        l_freq=SCI_L_FREQ,
        h_freq=SCI_H_FREQ,
        l_trans_bandwidth=SCI_L_TRANS_BW,
        h_trans_bandwidth=SCI_H_TRANS_BW,
    )

    # time-resolved SCI and PSP quality control
    # QC only: no segments are rejected or annotated

    windowed_sci_df, windowed_psp_df = (
        run_windowed_quality_qc(
            raw_od=raw_od,
            output_dir=output_dir,
            participant_id=participant_id,
        )
    )

    # create channel-level QC table

    channel_qc = build_channel_qc_table(
    raw_od=raw_od,
    sci=sci,
    raw_distances=raw_distances,
    short_max_m=SHORT_SEPARATION_MAX_M,
    sci_threshold=SCI_THRESHOLD,
    )


    # create one-row-per-source-detector-pair QC table

    pair_qc = build_pair_qc_table(
        channel_qc
    )

    # short-separation pair quality counts

    short_pair_qc = pair_qc.loc[
        pair_qc["is_short_separation"]
    ]

    n_short_pairs = len(
        short_pair_qc
    )

    n_bad_short_pairs = int(
        short_pair_qc["marked_bad"].sum()
    )

    n_good_short_pairs = (
        n_short_pairs
        - n_bad_short_pairs
    )


    # MONTAGE / SPATIAL QUALITY CONTROL

    print("\nMONTAGE / SPATIAL QC")

    (
        optode_positions,
        channel_locations,
        coordinate_frame,
    ) = build_spatial_qc_tables(
        raw=raw,
        raw_distances=raw_distances,
        pair_qc=pair_qc,
    )


    # save optode coordinates

    optode_positions.to_csv(
        output_dir
        / f"{participant_id}_optode_positions.csv",
        index=False,
    )


    # save channel coordinates plus SCI/QC information

    channel_locations.to_csv(
        output_dir
        / f"{participant_id}_channel_locations.csv",
        index=False,
    )


    n_sources = int(
        (
            optode_positions["optode_type"]
            == "source"
        ).sum()
    )

    n_detectors = int(
        (
            optode_positions["optode_type"]
            == "detector"
        ).sum()
    )

    n_pairs = len(
        channel_locations
    )

    n_short_pairs = int(
        channel_locations[
            "is_short_separation"
        ].sum()
    )


    print(
        "Coordinate frame:",
        coordinate_frame
    )

    print(
        "Sources:",
        n_sources
    )

    print(
        "Detectors:",
        n_detectors
    )

    print(
        "Source-detector pairs:",
        n_pairs
    )

    print(
        "Short-separation pairs:",
        n_short_pairs
    )


    # warn if recording does not match the finalized study montage

    if n_sources != EXPECTED_N_SOURCES:
        warnings.warn(
            f"Expected {EXPECTED_N_SOURCES} sources "
            f"but found {n_sources}."
        )

    if n_detectors != EXPECTED_N_DETECTORS:
        warnings.warn(
            f"Expected {EXPECTED_N_DETECTORS} detectors "
            f"but found {n_detectors}."
        )

    if n_pairs != EXPECTED_N_PAIRS:
        warnings.warn(
            f"Expected {EXPECTED_N_PAIRS} "
            f"source-detector pairs but found {n_pairs}."
        )

    if n_short_pairs != EXPECTED_N_SHORT_PAIRS:
        warnings.warn(
            f"Expected {EXPECTED_N_SHORT_PAIRS} "
            f"short-separation pairs but found "
            f"{n_short_pairs}."
        )


    # identify source-detector pairs below SCI threshold

    bad_pairs = pair_qc.loc[
        pair_qc["marked_bad"],
        "pair",
    ].tolist()


    # mark BOTH wavelengths from bad pairs as bad

    bad_channels = channel_qc.loc[
        channel_qc["pair"].isin(
            bad_pairs
        ),
        "channel",
    ].tolist()


    raw_od.info["bads"] = bad_channels


    print("\nSCI QUALITY CONTROL")

    print(
        "SCI threshold:",
        SCI_THRESHOLD
    )

    print(
        "Mean SCI:",
        float(np.mean(sci))
    )

    print(
        "Median SCI:",
        float(np.median(sci))
    )

    print(
        "Minimum SCI:",
        float(np.min(sci))
    )

    print(
        "Bad source-detector pairs:",
        len(bad_pairs)
    )

    print(
        "Bad wavelength channels:",
        len(bad_channels)
    )

    # save QC tables

    channel_qc.to_csv(
        output_dir
        / f"{participant_id}_channel_qc.csv",
        index=False,
    )

    pair_qc.to_csv(
        output_dir
        / f"{participant_id}_pair_qc.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "bad_channel": bad_channels
        }
    ).to_csv(
        output_dir
        / f"{participant_id}_bad_channels.csv",
        index=False,
    )

    # check whether any short-separation channels were marked bad

    bad_short = channel_qc.loc[
        channel_qc["pair_marked_bad"]
        & channel_qc["is_short_separation"],
        "channel",
    ].tolist()

    if bad_short:
        warnings.warn(
            "One or more short-separation channels "
            "were marked bad by SCI. Review these channels "
            "before interpreting short-channel regression."
        )

    # 8. TDDR MOTION CORRECTION

    raw_od_tddr = (
        temporal_derivative_distribution_repair(
            raw_od.copy()
        )
    )

    print("\nTDDR COMPLETE")

    # 9. SHORT-SEPARATION REGRESSION

    raw_od_ssr = short_channel_regression(
        raw_od_tddr.copy(),
        max_dist=SHORT_SEPARATION_MAX_M,
    )

    print(
        "\nSHORT-SEPARATION REGRESSION COMPLETE"
    )

    # 10. CONVERT OPTICAL DENSITY TO HbO / HbR

    raw_hb = beer_lambert_law(
        raw_od_ssr.copy(),
        ppf=PPF,
    )

    print("\nBEER-LAMBERT CONVERSION COMPLETE")

    print(
        "Channel types:",
        set(raw_hb.get_channel_types())
    )

    # 11. REMOVE SHORT-SEPARATION CHANNELS FROM FINAL CORTICAL ANALYSIS DATA
    
    hb_distances = source_detector_distances(
        raw_hb.info
    )

    short_hb_mask = (
        hb_distances
        < SHORT_SEPARATION_MAX_M
    )

    short_hb_channels = np.array(
        raw_hb.ch_names
    )[short_hb_mask]


    raw_hb_long = raw_hb.copy()

    raw_hb_long.drop_channels(
        list(short_hb_channels)
    )


    print("\nLONG-SEPARATION DATA")

    print(
        "Short HbO/HbR channels removed:",
        len(short_hb_channels)
    )

    print(
        "Long-separation HbO/HbR channels remaining:",
        len(raw_hb_long.ch_names)
    )

    # 12. BAND-PASS FILTER

    raw_hb_filtered = (
        raw_hb_long.copy()
    )

    raw_hb_filtered.filter(
        l_freq=FILTER_L_FREQ,
        h_freq=FILTER_H_FREQ,
        method="iir",
        iir_params=dict(
            order=FILTER_ORDER,
            ftype="butter",
        ),
        phase="zero",
    )


    print("\nFILTERING COMPLETE")

    print(
        f"Band-pass filter: "
        f"{FILTER_L_FREQ}-"
        f"{FILTER_H_FREQ} Hz"
    )

    # 13. SAVE FINAL CLEANED DATA

    cleaned_file = (
        output_dir
        / f"{participant_id}_desc-cleaned_hb_raw.fif"
    )

    raw_hb_filtered.save(
        cleaned_file,
        overwrite=True,
    )


    # optional: save intermediate datasets

    if SAVE_INTERMEDIATES:

        raw_od_tddr.save(
            output_dir
            / f"{participant_id}_desc-tddr_od_raw.fif",
            overwrite=True,
        )

        raw_hb_long.save(
            output_dir
            / f"{participant_id}_desc-unfiltered_hb_raw.fif",
            overwrite=True,
        )

    # 14. CREATE QC FIGURES

    print("\nCreating QC figures...")


    # source-detector distance distribution

    plot_distance_distribution(
        pair_qc,
        output_dir
        / f"{participant_id}_01_source_detector_distances.png",
        participant_id,
    )


    # SCI for every source-detector pair

    plot_sci_by_pair(
        pair_qc,
        output_dir
        / f"{participant_id}_03_sci_by_pair.png",
        participant_id,
    )


    # TDDR whole-recording comparison

    plot_tddr_heatmaps(
        raw_od,
        raw_od_tddr,
        channel_qc,
        output_dir
        / f"{participant_id}_04a_tddr_all_channels_heatmap.png",
        participant_id,
    )


    # TDDR quantitative motion summary

    plot_tddr_motion_summary(
        raw_od,
        raw_od_tddr,
        channel_qc,
        output_dir
        / f"{participant_id}_04b_tddr_motion_summary.png",
        participant_id,
    )


    # HbO/HbR before vs after temporal filtering

    plot_filtering_comparison(
        raw_hb_long,
        raw_hb_filtered,
        output_dir
        / f"{participant_id}_05_hbo_hbr_filtering_comparison.png",
        participant_id,
    )


    # PSD before vs after temporal filtering

    plot_psd_filtering_comparison(
        raw_hb_long,
        raw_hb_filtered,
        output_dir
        / f"{participant_id}_07_psd_filtering_comparison.png",
        participant_id,
    )


    # LSL event-marker timeline

    plot_event_timeline(
        raw_hb_filtered,
        output_dir
        / f"{participant_id}_09_event_timeline.png",
        participant_id,
    )


    # final HbO/HbR signal with event markers

    plot_filtered_with_event_lines(
        raw_hb_filtered,
        output_dir
        / f"{participant_id}_10_filtered_hbo_hbr_events.png",
        participant_id,
    )

    # final analyzed recording duration after cropping

    cropped_recording_duration_s = float(
        raw.times[-1]
    )
    
    # 15. FINAL CHANNEL COUNTS

    good_hbo, good_hbr = (
        get_good_hb_channels(
            raw_hb_filtered
        )
    )

    # final long-separation pair coverage

    all_long_pairs = {
        strip_measurement_suffix(channel)
        for channel in raw_hb_filtered.ch_names
    }

    good_long_pairs = {
        strip_measurement_suffix(channel)
        for channel in (
            good_hbo
            + good_hbr
        )
    }

    n_long_pairs = len(
        all_long_pairs
    )

    n_good_long_pairs = len(
        good_long_pairs
    )

    n_bad_long_pairs = (
        n_long_pairs
        - n_good_long_pairs
    )

    percent_good_long_pairs = (
        100
        * n_good_long_pairs
        / n_long_pairs
        if n_long_pairs > 0
        else np.nan
    )

    percent_bad_long_pairs = (
        100
        * n_bad_long_pairs
        / n_long_pairs
        if n_long_pairs > 0
        else np.nan
    )

    percent_good_short_pairs = (
        100
        * n_good_short_pairs
        / n_short_pairs
        if n_short_pairs > 0
        else np.nan
    )

    percent_bad_short_pairs = (
        100
        * n_bad_short_pairs
        / n_short_pairs
        if n_short_pairs > 0
        else np.nan
    )

    # SCI summary statistics using finite values only.

    finite_sci = sci[
        np.isfinite(sci)
    ]

    if finite_sci.size > 0:
        mean_sci = float(
            np.mean(finite_sci)
        )
        median_sci = float(
            np.median(finite_sci)
        )
        minimum_sci = float(
            np.min(finite_sci)
        )
        maximum_sci = float(
            np.max(finite_sci)
        )
    else:
        mean_sci = np.nan
        median_sci = np.nan
        minimum_sci = np.nan
        maximum_sci = np.nan

    # 16. CREATE PREPROCESSING SUMMARY

    summary = {

        "participant_id": participant_id,

        "input": {
            "snirf_file": str(
                snirf_file
            ),
            "event_source":
                "LSL annotations stored in SNIRF",
        },

        "software": {
            "mne": mne.__version__,
            "mne_nirs": mne_nirs.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },

        "parameters": {

            "short_separation_max_m":
                SHORT_SEPARATION_MAX_M,

            "sci_threshold":
                SCI_THRESHOLD,

            "sci_l_freq_hz":
                SCI_L_FREQ,

            "sci_h_freq_hz":
                SCI_H_FREQ,

            "sci_l_trans_bandwidth_hz":
                SCI_L_TRANS_BW,

            "sci_h_trans_bandwidth_hz":
                SCI_H_TRANS_BW,
            
            "windowed_qc_window_s":
                WINDOWED_QC_WINDOW_S,

            "crop_at_final_task_end":
                CROP_AT_FINAL_TASK_END,

            "final_task_end_code":
                FINAL_TASK_END_CODE,

            "ppf":
                PPF,

            "filter_l_freq_hz":
                FILTER_L_FREQ,

            "filter_h_freq_hz":
                FILTER_H_FREQ,

            "filter_order":
                FILTER_ORDER,

            "filter_type":
                "Butterworth IIR",

            "filter_phase":
                "zero",

            "motion_correction":
                "TDDR",

            "short_channel_regression":
                True,
        },

        "recording": {

            "sampling_rate_hz":
                float(
                    raw.info["sfreq"]
                ),

            "original_duration_s":
                original_recording_duration_s,

            "cropped_duration_s":
                cropped_recording_duration_s,

            "n_raw_channels":
                len(
                    raw.ch_names
                ),

            "n_annotations":
                len(
                    raw.annotations
                ),
        },

        "qc": {

            "n_short_wavelength_channels":
                int(
                    short_raw_mask.sum()
                ),
            
            "n_short_pairs":
                n_short_pairs,

            "n_good_short_pairs":
                n_good_short_pairs,

            "n_bad_short_pairs":
                n_bad_short_pairs,

            "percent_good_short_pairs":
                float(
                    percent_good_short_pairs
                ),

            "percent_bad_short_pairs":
                float(
                    percent_bad_short_pairs
                ),

            "n_bad_wavelength_channels":
                len(
                    bad_channels
                ),

            "n_bad_pairs":
                len(
                    bad_pairs
                ),

            "mean_sci":
                mean_sci,

            "median_sci":
                median_sci,

            "minimum_sci":
                minimum_sci,

            "maximum_sci":
                maximum_sci,
        },

        "final_data": {

            "n_long_pairs":
                n_long_pairs,

            "n_good_long_pairs":
                n_good_long_pairs,

            "n_bad_long_pairs":
                n_bad_long_pairs,

            "percent_good_long_pairs":
                float(
                    percent_good_long_pairs
                ),

            "percent_bad_long_pairs":
                float(
                    percent_bad_long_pairs
                ),

            "n_long_hb_channels":
                len(
                    raw_hb_filtered.ch_names
                ),

            "n_bad_hb_channels_marked":
                len(
                    raw_hb_filtered.info["bads"]
                ),

            "n_good_hbo_channels":
                len(
                    good_hbo
                ),

            "n_good_hbr_channels":
                len(
                    good_hbr
                ),

            "cleaned_file":
                str(
                    cleaned_file
                ),
        },
    }

    # 17. SAVE JSON SUMMARY

    summary_json_file = (
        output_dir
        / f"{participant_id}_preprocessing_summary.json"
    )

    with summary_json_file.open("w") as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # 18. SAVE READABLE TXT SUMMARY

    summary_text = f"""
Participant: {participant_id}

INPUT
SNIRF: {snirf_file}
Event source: LSL annotations stored in SNIRF

RECORDING
Sampling rate: {raw.info["sfreq"]:.6f} Hz
Original recording duration: {original_recording_duration_s:.3f} seconds
Cropped/analyzed duration: {cropped_recording_duration_s:.3f} seconds
Raw wavelength channels: {len(raw.ch_names)}
Event markers: {len(raw.annotations)}

PREPROCESSING
Short-separation cutoff: {SHORT_SEPARATION_MAX_M * 1000:.1f} mm
SCI threshold: {SCI_THRESHOLD:.2f}
SCI frequency range: {SCI_L_FREQ:.2f}-{SCI_H_FREQ:.2f} Hz
Motion correction: TDDR
Short-separation regression: Yes
PPF: {PPF}
Band-pass filter: {FILTER_L_FREQ}-{FILTER_H_FREQ} Hz
Filter order: {FILTER_ORDER}

QUALITY CONTROL
Mean SCI: {mean_sci:.4f}
Median SCI: {median_sci:.4f}
Bad source-detector pairs: {len(bad_pairs)}
Bad wavelength channels: {len(bad_channels)}
Short-separation pairs: {n_short_pairs}
Good short-separation pairs: {n_good_short_pairs} ({percent_good_short_pairs:.1f}%)
Bad short-separation pairs: {n_bad_short_pairs} ({percent_bad_short_pairs:.1f}%)

FINAL CLEANED DATA
Long-separation source-detector pairs: {n_long_pairs}
Good long-separation pairs: {n_good_long_pairs}
Bad long-separation pairs: {n_bad_long_pairs}
Good long-separation pairs: {percent_good_long_pairs:.1f}%
Bad long-separation pairs: {percent_bad_long_pairs:.1f}%
Long-separation HbO/HbR channels: {len(raw_hb_filtered.ch_names)}
Good HbO channels: {len(good_hbo)}
Good HbR channels: {len(good_hbr)}
Cleaned file: {cleaned_file}
"""

    summary_txt_file = (
        output_dir
        / f"{participant_id}_preprocessing_summary.txt"
    )

    summary_txt_file.write_text(
        summary_text.strip()
    )

    # 19. PRINT FINAL RESULTS

    print("PREPROCESSING COMPLETE")

    print(
        "Participant:",
        participant_id
    )

    print(
        "Output folder:",
        output_dir
    )

    print(
        "Cleaned data:",
        cleaned_file
    )

    print(
        "Bad source-detector pairs:",
        len(bad_pairs)
    )

    print(
        "Good HbO channels:",
        len(good_hbo)
    )

    print(
        "Good HbR channels:",
        len(good_hbr)
    )

    print(
        "Event markers:",
        len(raw.annotations)
    )


    # keep cleaned data and summary available in Python after the function finishes

    return raw_hb_filtered, summary


# RUN PARTICIPANT

if __name__ == "__main__":

    cleaned_hb, preprocessing_summary = preprocess_participant(
        participant_id=PARTICIPANT_ID,
        snirf_file=SNIRF_FILE,
        output_root=OUTPUT_ROOT,
    )
