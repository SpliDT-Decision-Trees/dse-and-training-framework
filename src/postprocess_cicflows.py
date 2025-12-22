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
from concurrent.futures import ThreadPoolExecutor

import pandas as pd


def processes_one_partition(args, partition):
    # each class gets its own directory
    path_to_partition = args.dataset_path.format(partition)
    CLASSES = os.listdir(path_to_partition)
    print("Folders: ", CLASSES)
    num_classes = len(CLASSES)
    df_list = []

    # process one class in this partition
    for this_class in CLASSES:
        path_to_class = os.path.join(path_to_partition, this_class)

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
    combined_df.to_csv(
        os.path.join(args.destination, f"{args.dataset_name}_Flow_v{partition}_c{num_classes}.csv"),
        index=False,
    )
    print(f"Saved {args.dataset_name}_Flow_v{partition}_c{num_classes}.csv")
    pass


def main():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--dataset_name", type=str)
    CLI.add_argument("--dataset_path", type=str)
    CLI.add_argument("--destination", type=str)
    CLI.add_argument("--partitions", nargs="+", type=int)
    CLI.add_argument("--num_jobs", type=int)
    # convert incoming args to a dictionary
    args = CLI.parse_args()
    print(args)

    # process each partition in this dataset
    jobs = [(args, partition) for partition in args.partitions]

    # start parallel jobs for each requested evaluation
    with ThreadPoolExecutor(max_workers=args.num_jobs) as executor:
        futures = [executor.submit(processes_one_partition, *job) for job in jobs]
    pass


if __name__ == "__main__":
    main()
