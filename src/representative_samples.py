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


import glob
import os


def get_pcap_files_in_folders(base_folder):
    # Get all subfolders in the base folder
    subfolders = os.listdir(base_folder)
    pcap_files_by_class = {}

    for this_class in subfolders:
        if this_class == "processed" or this_class == "merged_csv":
            continue
        folder = os.path.join(base_folder, this_class)
        # Get all PCAP files in the current folder
        pcap_files = glob.glob(os.path.join(folder, "*.pcap"))
        pcap_files_by_class[this_class] = pcap_files

    return pcap_files_by_class


def get_file_size(file_path):
    # return size in MB
    return round(os.path.getsize(file_path) / 1024 / 1024, 1)


def calculate_average_size(pcap_files_by_folder):
    # Calculate the average file size across all folders
    all_files = []
    for files in pcap_files_by_folder.values():
        all_files.extend(files)

    # Get sizes of all files
    file_sizes = [get_file_size(file) for file in all_files]

    # Calculate average size
    return sum(file_sizes) / len(file_sizes)


def select_comparable_representative_files(pcap_files_by_folder, avg_size):
    representative_files = []

    for this_class, files in pcap_files_by_folder.items():
        # Get the sizes of all PCAP files
        file_sizes = [(file, get_file_size(file)) for file in files]

        # Skip empty folders
        if len(file_sizes) == 0:
            continue

        # Find the file with size closest to the average size
        closest_file = min(file_sizes, key=lambda x: abs(x[1] - avg_size))
        representative_files.append((closest_file[0], this_class))

    return representative_files


def main():
    # Example usage:
    base_folder = "/CICFlowMeter/dataset/"
    pcap_files_by_class = get_pcap_files_in_folders(base_folder)
    average_file_size = calculate_average_size(pcap_files_by_class)

    representative_files = select_comparable_representative_files(
        pcap_files_by_class, average_file_size
    )

    print("\nRepresentative PCAP files:")
    for this_file, this_class in representative_files:
        print(f"{this_class}: {this_file} (size: {get_file_size(this_file)}MB)")

    print()
    print(representative_files)


if __name__ == "__main__":
    main()
