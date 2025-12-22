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


import os
import pickle
import random
import time

import utils

PARTITION_CODE_FOR_PACKETS = 0
PARTITION_CODE_FOR_PHASES = 1024


def main():
    parsed_args = utils.parse_yml_config()
    random.seed(time.time())

    num_partitions = parsed_args.dataset.num_partitions
    read_path = os.path.join(parsed_args.dataset.path, parsed_args.dataset.name)
    write_path = os.path.join(read_path, parsed_args.dataset.destination)

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    # get one train-test split for all partitions
    one_partition_dataset_path = os.path.join(
        read_path, parsed_args.dataset.format.format(1, parsed_args.dataset.num_classes)
    )
    # limit maximum flows to 500K
    train_flows, test_flows = utils.get_train_test_split(
        dataset_file=one_partition_dataset_path, num_partition=1, max_flows=500000, train_split=0.8
    )

    for this_partition in num_partitions:
        print(f"\n\nProcessing dataset with {this_partition} partitions")

        dataset_path = os.path.join(
            read_path,
            parsed_args.dataset.format.format(this_partition, parsed_args.dataset.num_classes),
        )
        destination_path = os.path.join(write_path, f"dataset_df_p{this_partition}.pkl")
        print(f"Reading dataset from {dataset_path}")

        # get partition specifically for packets
        if this_partition == PARTITION_CODE_FOR_PACKETS:
            train_flows, test_flows = utils.get_train_test_split(
                dataset_file=dataset_path, num_partition=0, train_split=0.8, do_packets=True
            )
            # generate packets (IIsy and per-packet models)
            processed_df_dict = utils.read_and_process_dataset(
                dataset_path, this_partition, train_flows, test_flows, do_packets=True, num_jobs=16
            )

        # generate phases (NetBeacon)
        elif this_partition == PARTITION_CODE_FOR_PHASES:
            processed_df_dict = utils.read_and_process_dataset(
                dataset_path, this_partition, train_flows, test_flows, do_phases=True
            )

        # generate partitions (CAP)
        else:
            processed_df_dict = utils.read_and_process_dataset(
                dataset_path, this_partition, train_flows, test_flows
            )

        with open(destination_path, "wb") as pkl_save_path:
            pickle.dump(processed_df_dict, pkl_save_path)
        print(f"Processed dataset saved at {destination_path}")

        # Read and show the dataset
        read_processed_df = utils.read_processed_dataset(destination_path)
        ungrouped_dataset = read_processed_df["ungrouped_train_df"]
        grouped_dataset = read_processed_df["grouped_train_df"]
        print("Grouped Dataset")
        utils.show_ungrouped_dataset(ungrouped_dataset)
        print("Ungrouped Dataset")
        utils.show_grouped_dataset(grouped_dataset)


if __name__ == "__main__":
    main()
