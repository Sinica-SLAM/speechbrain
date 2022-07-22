"""
The .csv preperation functions for WSJ0-Mix.

Author
 * Cem Subakan 2020

 """

from calendar import c
import os
import csv


def prepare_wsjmix(
    datapath,
    savepath,
    n_spks=2,
    skip_prep=False,
    librimix_addnoise=False,
    fs=8000,
):
    """
    Prepared wsj2mix if n_spks=2 and wsj3mix if n_spks=3.

    Arguments:
    ----------
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file.
        n_spks (int): number of speakers
        skip_prep (bool): If True, skip data preparation
        librimix_addnoise: If True, add whamnoise to librimix datasets
    """

    if skip_prep:
        return

    if "wsj" in datapath or "WSJ" in datapath:

        if n_spks == 2:
            assert (
                "2speakers" in datapath
            ), "Inconsistent number of speakers and datapath"
            create_wsj_csv(datapath, savepath)
        elif n_spks == 3:
            assert (
                "3speakers" in datapath
            ), "Inconsistent number of speakers and datapath"
            create_wsj_csv_3spks(datapath, savepath)
        else:
            raise ValueError("Unsupported Number of Speakers")
    elif "tat" in datapath or "TAT" in datapath:
        create_tat_csv(datapath, savepath)
    else:
        print("Creating a csv file for a custom dataset")
        create_custom_dataset(datapath, savepath)


def create_custom_dataset(
    datapath,
    savepath,
    dataset_name="custom",
    set_types=["train", "valid", "test"],
    folder_names={
        "source1": "source1",
        "source2": "source2",
        "mixture": "mixture",
    },
):
    """
    This function creates the csv file for a custom source separation dataset
    """

    for set_type in set_types:
        mix_path = os.path.join(datapath, set_type, folder_names["mixture"])
        s1_path = os.path.join(datapath, set_type, folder_names["source1"])
        s2_path = os.path.join(datapath, set_type, folder_names["source2"])

        files = os.listdir(mix_path)

        mix_fl_paths = [os.path.join(mix_path, fl) for fl in files]
        s1_fl_paths = [os.path.join(s1_path, fl) for fl in files]
        s2_fl_paths = [os.path.join(s2_path, fl) for fl in files]

        csv_columns = [
            "ID",
            "duration",
            "mix_wav",
            "mix_wav_format",
            "mix_wav_opts",
            "s1_wav",
            "s1_wav_format",
            "s1_wav_opts",
            "s2_wav",
            "s2_wav_format",
            "s2_wav_opts",
            "noise_wav",
            "noise_wav_format",
            "noise_wav_opts",
        ]

        with open(
            os.path.join(savepath, dataset_name + "_" + set_type + ".csv"), "w"
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_path, s1_path, s2_path) in enumerate(
                zip(mix_fl_paths, s1_fl_paths, s2_fl_paths)
            ):

                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav": mix_path,
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None,
                    "s1_wav": s1_path,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "s2_wav": s2_path,
                    "s2_wav_format": "wav",
                    "s2_wav_opts": None,
                }
                writer.writerow(row)


def create_wsj_csv(datapath, savepath):
    """
    This function creates the csv files to get the speechbrain data loaders for the wsj0-2mix dataset.

    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
    """
    for set_type in ["tr", "cv", "tt"]:
        mix_path = os.path.join(datapath, "wav8k/min/" + set_type + "/mix/")
        s1_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s1/")
        s2_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s2/")

        files = os.listdir(mix_path)

        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        s2_fl_paths = [s2_path + fl for fl in files]

        csv_columns = [
            "ID",
            "duration",
            "mix_wav",
            "mix_wav_format",
            "mix_wav_opts",
            "s1_wav",
            "s1_wav_format",
            "s1_wav_opts",
            "s2_wav",
            "s2_wav_format",
            "s2_wav_opts",
        ]

        with open(savepath + "/wsj_" + set_type + ".csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_path, s1_path, s2_path) in enumerate(
                zip(mix_fl_paths, s1_fl_paths, s2_fl_paths)
            ):

                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav": mix_path,
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None,
                    "s1_wav": s1_path,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "s2_wav": s2_path,
                    "s2_wav_format": "wav",
                    "s2_wav_opts": None,
                }
                writer.writerow(row)


def create_wsj_csv_3spks(datapath, savepath):
    """
    This function creates the csv files to get the speechbrain data loaders for the wsj0-3mix dataset.

    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
    """
    for set_type in ["tr", "cv", "tt"]:
        mix_path = os.path.join(datapath, "wav8k/min/" + set_type + "/mix/")
        s1_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s1/")
        s2_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s2/")
        s3_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s3/")

        files = os.listdir(mix_path)

        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        s2_fl_paths = [s2_path + fl for fl in files]
        s3_fl_paths = [s3_path + fl for fl in files]

        csv_columns = [
            "ID",
            "duration",
            "mix_wav",
            "mix_wav_format",
            "mix_wav_opts",
            "s1_wav",
            "s1_wav_format",
            "s1_wav_opts",
            "s2_wav",
            "s2_wav_format",
            "s2_wav_opts",
            "s3_wav",
            "s3_wav_format",
            "s3_wav_opts",
        ]

        with open(savepath + "/wsj_" + set_type + ".csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_path, s1_path, s2_path, s3_path) in enumerate(
                zip(mix_fl_paths, s1_fl_paths, s2_fl_paths, s3_fl_paths)
            ):

                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav": mix_path,
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None,
                    "s1_wav": s1_path,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "s2_wav": s2_path,
                    "s2_wav_format": "wav",
                    "s2_wav_opts": None,
                    "s3_wav": s3_path,
                    "s3_wav_format": "wav",
                    "s3_wav_opts": None,
                }
                writer.writerow(row)


def create_tat_csv(datapath, savepath):
    """
    This function creates the csv files to get the speechbrain data loaders for the wsj0-2mix dataset.

    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
    """
    for set_type in ["tr", "cv", "tt"]:
        mix_channel_fl_paths = {}
        s1_channel_fl_paths = {}
        s2_channel_fl_paths = {}
        for channel_type in ["android", 'condenser', 'ios', 'lavalier', 'XYH-6-X', 'XYH-6-Y']:
            
            mix_path = os.path.join(datapath, f"wav8k_{channel_type}/min/" + set_type + "/mix_100/")
            s1_path = os.path.join(datapath, f"wav8k_{channel_type}/min/" + set_type + "/s1/")
            s2_path = os.path.join(datapath, f"wav8k_{channel_type}/min/" + set_type + "/s2/")

            files = os.listdir(mix_path)

            mix_fl_paths = [mix_path + fl for fl in files]
            s1_fl_paths = [s1_path + fl for fl in files]
            s2_fl_paths = [s2_path + fl for fl in files]
            mix_channel_fl_paths[channel_type] = mix_fl_paths
            s1_channel_fl_paths[channel_type] = s1_fl_paths
            s2_channel_fl_paths[channel_type] = s2_fl_paths

        csv_columns = [
            "ID",
            "duration",
            "android_mix_wav",
            "android_s1_wav",
            "android_s2_wav",
            "condenser_mix_wav",
            "condenser_s1_wav",
            "condenser_s2_wav",
            "ios_mix_wav",
            "ios_s1_wav",
            "ios_s2_wav",
            "lavalier_mix_wav",
            "lavalier_s1_wav",
            "lavalier_s2_wav",
            "XYH6X_mix_wav",
            "XYH6X_s1_wav",
            "XYH6X_s2_wav",
            "XYH6Y_mix_wav",
            "XYH6Y_s1_wav",
            "XYH6Y_s2_wav",
        ]

        with open(savepath + "/tat_" + set_type + ".csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_android_path, s1_android_path, s2_android_path, 
                    mix_condenser_path, s1_condenser_path, s2_condenser_path,
                    mix_ios_path, s1_ios_path, s2_ios_path,
                    mix_lavalier_path, s1_lavalier_path, s2_lavalier_path, 
                    mix_XYH6X_path, s1_XYH6X_path, s2_XYH6X_path,
                    mix_XYH6Y_path, s1_XYH6Y_path, s2_XYH6Y_path
                    ) in enumerate(
                        zip(mix_channel_fl_paths['android'], s1_channel_fl_paths['android'], s2_channel_fl_paths['android'],
                            mix_channel_fl_paths['condenser'], s1_channel_fl_paths['condenser'], s2_channel_fl_paths['condenser'],
                            mix_channel_fl_paths['ios'], s1_channel_fl_paths['ios'], s2_channel_fl_paths['ios'],
                            mix_channel_fl_paths['lavalier'], s1_channel_fl_paths['lavalier'], s2_channel_fl_paths['lavalier'],
                            mix_channel_fl_paths['XYH-6-X'], s1_channel_fl_paths['XYH-6-X'], s2_channel_fl_paths['XYH-6-X'],
                            mix_channel_fl_paths['XYH-6-Y'], s1_channel_fl_paths['XYH-6-Y'], s2_channel_fl_paths['XYH-6-Y'],
                        )
            ):

                row = {
                    "ID": i,
                    "duration": 1.0,
                    "android_mix_wav": mix_android_path,
                    "android_s1_wav": s1_android_path,
                    "android_s2_wav":s2_android_path,
                    "condenser_mix_wav": mix_condenser_path,
                    "condenser_s1_wav": s1_condenser_path,
                    "condenser_s2_wav": s2_condenser_path,
                    "ios_mix_wav": mix_ios_path,
                    "ios_s1_wav": s1_ios_path,
                    "ios_s2_wav": s2_ios_path,
                    "lavalier_mix_wav": mix_lavalier_path,
                    "lavalier_s1_wav": s1_lavalier_path,
                    "lavalier_s2_wav": s2_lavalier_path,
                    "XYH6X_mix_wav": mix_XYH6X_path,
                    "XYH6X_s1_wav": s1_XYH6X_path,
                    "XYH6X_s2_wav": s2_XYH6X_path,
                    "XYH6Y_mix_wav": mix_XYH6Y_path,
                    "XYH6Y_s1_wav": s1_XYH6Y_path,
                    "XYH6Y_s2_wav": s2_XYH6Y_path,
                }
                writer.writerow(row)