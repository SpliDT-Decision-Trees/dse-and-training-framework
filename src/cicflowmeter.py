# *****************************************************************************
#
# Copyright 2026
#   Murayyiam Parvez (Purdue University),
#   Annus Zulfiqar (University of Michigan),
#   Roman Beltiukov (University of California, Santa Barbara),
#   Shir Landau Feibish (The Open University of Israel),
#   Walter Willinger (NIKSUN Inc.),
#   Arpit Gupta (University of California, Santa Barbara),
#   Muhammad Shahbaz (University of Michigan)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# *****************************************************************************


import argparse
import os
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pandas as pd

# to select representative samples from each class
import representative_samples as rs


def merge_generated_csvs(path_to_dataset, destination, mode, partition=None):
    # each class gets its own directory
    CLASSES = os.listdir(path_to_dataset)
    print("Folders: ", CLASSES)
    num_classes = len(CLASSES)
    df_list = []

    # process one class in this partition
    for this_class in CLASSES:
        path_to_class = os.path.join(path_to_dataset, this_class)

        # each class can have multiple CSV files
        for csv_file in os.listdir(path_to_class):
            if not csv_file.endswith(".csv"):
                continue

            # path to csv file
            path_to_csv_file = os.path.join(path_to_class, csv_file)
            print(f"Working with {path_to_csv_file}")

            # read each csv file
            csv_df = pd.read_csv(path_to_csv_file)
            df_list.append(csv_df)

    combined_df = pd.concat(df_list, ignore_index=True)

    if mode == "partitions":
        print(f"Saving Partitions_p{partition}_c{num_classes}.csv")
        combined_df.to_csv(
            os.path.join(destination, f"Partitions_p{partition}_c{num_classes}.csv"), index=False
        )
    elif mode == "phases":
        print(f"Saving Phases_c{num_classes}.csv")
        combined_df.to_csv(os.path.join(destination, f"Phases_c{num_classes}.csv"), index=False)
    elif mode == "packets":
        print(f"Saving Packets_c{num_classes}.csv")
        combined_df.to_csv(os.path.join(destination, f"Packets_c{num_classes}.csv"), index=False)
    pass


def main():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--mode", type=str)
    CLI.add_argument("--dataset_dir", type=str, default="/CICFlowMeter/dataset")
    CLI.add_argument("--partitions", nargs="+", type=int)  # pass 1024 for phases
    CLI.add_argument("--flow_size_cutoff", type=int)  # , default=10)
    CLI.add_argument(
        "--no_subsample", action="store_true", default=False
    )  # pick one PCAP per class only
    CLI.add_argument("--merge_only", action="store_true", default=False)
    CLI.add_argument("--num_jobs", type=int, default=8)
    # convert incoming args to a dictionary
    args = CLI.parse_args()

    # create "partitions" or "phases"
    MODE = args.mode
    assert MODE in ["phases", "partitions", "packets"], (
        "MODE must be one of ['phases', 'partitions', 'packets']"
    )

    DATASET_DIR = args.dataset_dir  # "/CICFlowMeter/dataset"
    # partitions = 1 2 3 4 5 6 7 8 9 10
    # just to pass something to docker; won't be used
    ALL_PARTITIONS = args.partitions if MODE == "partitions" else [0]
    FLOW_SIZE_CUT_OFF = args.flow_size_cutoff if MODE in ["partitions", "phases"] else 0
    NUM_JOBS = args.num_jobs  # 8

    if not args.merge_only:
        PCAP_PATHS_TO_LABELS = []

        # get all PCAPs for each class
        if args.no_subsample:
            for this_class in os.listdir(DATASET_DIR):
                if this_class in ["processed", "merged_csv"]:
                    continue
                this_class_dir = os.path.join(DATASET_DIR, this_class)

                # use all PCAPs in the class
                for pcap_file in os.listdir(this_class_dir):
                    PCAP_PATHS_TO_LABELS += [(os.path.join(this_class_dir, pcap_file), this_class)]

            print(PCAP_PATHS_TO_LABELS)

        # subsample one PCAP per-class
        else:
            pcap_files_by_folder = rs.get_pcap_files_in_folders(DATASET_DIR)
            average_file_size = rs.calculate_average_size(pcap_files_by_folder)
            PCAP_PATHS_TO_LABELS = rs.select_comparable_representative_files(
                pcap_files_by_folder, average_file_size
            )

            print("\nRepresentative PCAP files:")
            for this_file, this_class in PCAP_PATHS_TO_LABELS:
                print(f"{this_class}: {this_file} (size: {rs.get_file_size(this_file)}MB)")

            print()
            print(PCAP_PATHS_TO_LABELS)

        # PCAP Processing step
        jobs = []
        # collect all processing jobs
        for PCAP_FILE_PATH, LABEL in PCAP_PATHS_TO_LABELS:
            for PARTITIONS in ALL_PARTITIONS:
                jobs += [
                    [
                        "java",
                        "-Djava.library.path=/CICFlowMeter/jnetpcap/linux/jnetpcap-1.4.r1425/",
                        "-jar",
                        "build/libs/CICFlowMeter-4.0.jar",
                        f"{PCAP_FILE_PATH}",
                        "/CICFlowMeter/dataset/processed/",
                        f"{PARTITIONS}",
                        f"{LABEL}",
                        f"{FLOW_SIZE_CUT_OFF}",
                        f"{MODE}",
                    ]
                ]
                pass
            pass

        # start parallel processes for each PCAP file in the dataset
        with ProcessPoolExecutor(max_workers=NUM_JOBS) as executor:
            futures = [executor.submit(subprocess.run, job) for job in jobs]
            pass
        pass

    # Merge step
    print("Starting merge step")
    path_to_dataset = os.path.join(
        "/CICFlowMeter/dataset/processed", f"cutoff{FLOW_SIZE_CUT_OFF}", f"{MODE}"
    )
    if MODE == "partitions":
        jobs = [
            (
                os.path.join(path_to_dataset, f"partition{this_part}"),
                path_to_dataset,
                MODE,
                this_part,
            )
            for this_part in args.partitions
        ]
    else:
        jobs = [(path_to_dataset, path_to_dataset, MODE)]

    # start parallel jobs for each requested evaluation
    with ThreadPoolExecutor(max_workers=args.num_jobs) as executor:
        futures = [executor.submit(merge_generated_csvs, *job) for job in jobs]
        for future in futures:
            try:
                future.result()  # This will raise an exception if the thread failed
            except Exception as e:
                print(f"Thread raised an exception: {e}")
                pass
            pass
        pass


if __name__ == "__main__":
    main()
