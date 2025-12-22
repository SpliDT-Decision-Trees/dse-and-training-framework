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

import pandas as pd


def get_flow_sizes(file_path):
    # read csv using pandas
    this_raw_dataset_df = pd.read_csv(file_path)
    # # show column names
    # add a third column 'Total Packets' which is the sum of 'Total Fwd Packets' and 'Total Backward Packets'
    this_raw_dataset_df["Total Packets"] = (
        this_raw_dataset_df["Total Fwd Packet"] + this_raw_dataset_df["Total Bwd packets"]
    )
    # show unique flow sizes and their counts
    flow_sizes = this_raw_dataset_df["Total Packets"].value_counts()

    # create a new dataframe with flow sizes and their counts
    flow_sizes_df = pd.DataFrame(flow_sizes)
    # name the columns 'flow size', 'count'
    flow_sizes_df.columns = ["count"]
    flow_sizes_df["flow size"] = flow_sizes_df.index
    # rearrange columns; 'flow size' first, 'count' second
    flow_sizes_df = flow_sizes_df[["flow size", "count"]]
    # sort the dataframe by 'flow size'
    flow_sizes_df = flow_sizes_df.sort_values(by="flow size")
    return flow_sizes_df


def main():
    # save the dataframes to csv files with same names as dataframes
    flow_sizes_path = "plots/csv/top-level-flow-sizes"
    if not os.path.exists(flow_sizes_path):
        os.makedirs(flow_sizes_path)

    top_level_raw_dataset_path = "/home/annuszulfiqar/research/NetEye/ready-to-train-datasets/"
    dataset_paths = {
        "cic_iomt_2024": "CIC-IoMT-2024-PCAPS1-f10/CIC-IoMT-2024_Flow_v1_c19.csv",
        "cic_iot_2023": "CIC-IOT-2023-PCAPS1-f10/CIC-IOT-2023_Flow_v1_c4.csv",
        "iscxvpn2016": "ISCXVPN2016-PCAPS0-f10/ISCXVPN2016_Flow_v1_c13.csv",
        "ucsbfinetuning": "UCSBFinetuning-PCAPS0-f10/UCSBFinetuning_Flow_v1_c11.csv",
        "cic_iot_2023_32": "CIC-IOT-2023-32-PCAPS1-f10/CIC-IOT-2023-32_Flow_v1_c32.csv",
        "cic_ids_2017": "CIC-IDS-2017-PCAPS1-f10/CIC-IDS-2017_Flow_v1_c10.csv",
        "cic_ids_2018": "CIC-IDS-2018-PCAPS1-f10/CIC-IDS-2018_Flow_v1_c10.csv",
    }

    for dataset, path in dataset_paths.items():
        raw_dataset_path = os.path.join(top_level_raw_dataset_path, path)
        flow_sizes_df = get_flow_sizes(raw_dataset_path)
        flow_sizes_df.to_csv(
            os.path.join(flow_sizes_path, f"{dataset}_flow_sizes.csv"), index=False
        )
        print(
            "Flow sizes for",
            dataset,
            "saved to",
            os.path.join(flow_sizes_path, f"{dataset}_flow_sizes.csv"),
        )

    pass


if __name__ == "__main__":
    main()
