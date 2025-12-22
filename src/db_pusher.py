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
from enum import Enum

import numpy as np

import db_manager
import model
import netbeacon_tcam_rules as nb_tcam_rules
import switch_resources as sw_resources
import utils
from leo import train_leo

ATTRIBUTES = {
    "name": "TEXT",
    "iterations": "INT",
    "f1_score": "FLOAT",
    "feature_limit": "INT",
    "feature_table_entries": "INT",
    "max_depth": "INT",
    "num_flows": "INT",
    "num_partitions": "INT",
    "total_features": "INT",
    "tree_table_entries": "INT",
}


class Property(Enum):
    NAME = 0
    ITERATIONS = 1
    F1_SCORE = 2
    FEATURE_LIMIT = 3
    FEATURE_TABLE_ENTRIES = 4
    MAX_DEPTH = 5
    NUM_FLOWS = 6
    NUM_PARTITIONS = 7
    TOTAL_FEATURES = 8
    TREE_TABLE_ENTRIES = 9


ONE_PARTITION = 1


def __push_leo_as_one_partition():
    parsed_args = utils.parse_yml_config()
    # read results from db where name = leo
    # and push to the hyperparameter database
    # as single partition results for cap
    parsed_args.db_table_name = ("baseline" + "-" + parsed_args.dataset.name).replace("-", "_")
    print(parsed_args.db_table_name)

    # read leo results from baseline tables
    leo_configs = db_manager.read_from_logging_database(parsed_args)

    # update db name to hyperparameter
    parsed_args.db_table_name = (
        "hypermapper"
        + "-"
        + parsed_args.dataset.name
        + "-"
        + parsed_args.hypermapper.scenario.optimization_method
    ).replace("-", "_")
    print(parsed_args.db_table_name)

    for max_depth in parsed_args.one_partition.depths:
        for num_features in parsed_args.one_partition.features:
            count += 1
        pass

    for config in leo_configs:
        # pick only leo results
        if config[Property.NAME.value] != "leo":
            continue

        # read the dataset and train a decision tree model
        dataset_path = os.path.join(
            parsed_args.dataset.path,
            parsed_args.dataset.name,
            parsed_args.dataset.destination,
            f"dataset_df_p1.pkl",  # hardcoded for Leo
        )

        read_processed_df = utils.read_processed_dataset(dataset_path)
        ungrouped_training_dataset = read_processed_df["ungrouped_train_df"]
        ungrouped_testing_dataset = read_processed_df["ungrouped_test_df"]

        # get the training and testing dataset for first window for all flows
        X_train, y_train = utils.get_filtered_samples_and_labels(
            ungrouped_training_dataset, windows=[1]
        )
        X_test, y_test = utils.get_filtered_samples_and_labels(
            ungrouped_testing_dataset, windows=[1]
        )
        # drop the 'Flow ID' and 'Window' columns
        X_train_without_id = X_train.drop(columns=["Flow ID", "Window"], axis=1)
        # Replace inf values with NaN
        X_train_without_id = X_train_without_id.replace([np.inf, -np.inf], np.nan)
        # Identify rows with NaN values
        dropped_indices = X_train_without_id[X_train_without_id.isna().any(axis=1)].index
        # Drop rows with NaN and reset index
        X_train_without_id = X_train_without_id.drop(dropped_indices).reset_index(drop=True)
        y_train = y_train.drop(dropped_indices).reset_index(drop=True)

        # repeat the above for the testing dataset
        X_test_without_id = X_test.drop(columns=["Flow ID", "Window"], axis=1)
        # Replace inf values with NaN
        X_test_without_id = X_test_without_id.replace([np.inf, -np.inf], np.nan)
        # Identify rows with NaN values
        dropped_indices = X_test_without_id[X_test_without_id.isna().any(axis=1)].index
        # Drop rows with NaN and reset index
        X_test_without_id = X_test_without_id.drop(dropped_indices).reset_index(drop=True)
        # X_test = X_test.drop(dropped_indices).reset_index(drop=True)
        y_test = y_test.drop(dropped_indices).reset_index(drop=True)

        max_depth = config[Property.MAX_DEPTH.value]
        total_features = config[Property.TOTAL_FEATURES.value]

        # Set the maximum number of leaf nodes
        max_leaf_nodes = None
        if max_depth <= 13:
            max_leaf_nodes = int(2**max_depth)
        elif max_depth == 14:
            max_leaf_nodes = 6144
        elif max_depth == 15:
            max_leaf_nodes = 3072
        elif max_depth == 16:
            max_leaf_nodes = 2048
        else:
            max_leaf_nodes = 1024

        if not max_depth or not total_features:
            continue

        # train the model
        _, _, final_dtree = train_leo(
            X_train_without_id,
            y_train,
            X_test_without_id,
            y_test,
            max_depth,
            max_leaf_nodes,
            total_features,
        )

        # compute these using NetBeacon rulegen logic
        # keep all nodes
        tree_nodes_bool = np.ones(final_dtree.tree_.node_count, dtype=bool)
        # get the number of entries for this tree
        feature_entries, tree_entries = nb_tcam_rules.get_class_flow(
            final_dtree.tree_, tree_nodes_bool
        )
        model_table_entries = feature_entries + tree_entries

        # push the results to the hyperparameter database
        # as results of single partition CAP
        model_performance = model.ModelConfig(
            max_depth=max_depth,
            features_per_partition=total_features,
            num_partitions=1,
            f1_score=config[Property.F1_SCORE.value],
            num_flows=config[Property.NUM_FLOWS.value],
            total_features=total_features,
            table_entries=model_table_entries,
        )
        db_manager.commit_to_logging_database(parsed_args, "cap", 0, model_performance)
    pass


def push_baseline_results():
    parsed_args = utils.parse_yml_config()
    parsed_args.db_table_name = ("sigcomm-baseline" + "-" + parsed_args.dataset.name).replace(
        "-", "_"
    )
    print(parsed_args.db_table_name)

    # create the logging database for grafana dashboard
    db_manager.create_logging_database(parsed_args)

    # push leo results
    for sample in parsed_args.leo.pareto_front:
        model_performance = model.ModelConfig(
            max_depth=sample.max_depth,
            f1_score=sample.f1_score,
            num_flows=sample.num_flows,
            total_features=sample.num_features,
            table_entries=sample.num_tcam_rules,
        )
        db_manager.commit_to_logging_database(parsed_args, "leo", 0, model_performance)

    # push netbeacon results
    for sample in parsed_args.netbeacon.pareto_front:
        model_performance = model.ModelConfig(
            max_depth=sample.max_depth,
            f1_score=sample.f1_score,
            num_flows=sample.num_flows,
            total_features=sample.num_features,
            table_entries=sample.num_tcam_rules,
        )
        db_manager.commit_to_logging_database(parsed_args, "netbeacon", 0, model_performance)

    # push iisy results
    for sample in parsed_args.iisy.pareto_front:
        model_performance = model.ModelConfig(
            max_depth=sample.max_depth,
            f1_score=sample.f1_score,
            num_flows=2000000,  # placeholder
            total_features=sample.num_features,
        )
        db_manager.commit_to_logging_database(parsed_args, "iisy", 0, model_performance)

    pass


def push_one_partition_results():
    parsed_args = utils.parse_yml_config()
    parsed_args.db_table_name = (
        "hypermapper"
        + "-"
        + parsed_args.dataset.name
        + "-"
        + parsed_args.hypermapper.scenario.optimization_method
    ).replace("-", "_")

    # push one-partition results
    for sample in parsed_args.one_partition.pareto_front:
        model_performance = model.ModelConfig(
            max_depth=sample.max_depth,
            features_per_partition=sample.num_features,
            num_partitions=ONE_PARTITION,
            f1_score=sample.f1_score,
            num_flows=sample.num_flows,
            total_features=sample.num_features,
            table_entries=sample.num_tcam_rules,
        )
        db_manager.commit_to_logging_database(parsed_args, "cap", 0, model_performance)


def push_all_static_results():
    parsed_args = utils.parse_yml_config()

    # push results for our baseline decision trees
    parsed_args.db_table_name = ("sigcomm-baseline" + "-" + parsed_args.dataset.name).replace(
        "-", "_"
    )
    print(parsed_args.db_table_name)

    # create the logging database for grafana dashboard
    db_manager.create_logging_database(parsed_args)

    # push leo results
    for sample in parsed_args.leo.pareto_front:
        model_performance = model.ModelConfig(
            max_depth=sample.max_depth,
            f1_score=sample.f1_score,
            num_flows=sample.num_flows,
            total_features=sample.num_features,
            table_entries=sample.num_tcam_rules,
        )
        db_manager.commit_to_logging_database(parsed_args, "leo", 0, model_performance)

    # push netbeacon results
    for sample in parsed_args.netbeacon.pareto_front:
        model_performance = model.ModelConfig(
            max_depth=sample.max_depth,
            f1_score=sample.f1_score,
            num_flows=sample.num_flows,
            total_features=sample.num_features,
            table_entries=sample.num_tcam_rules,
        )
        db_manager.commit_to_logging_database(parsed_args, "netbeacon", 0, model_performance)

    # push iisy results
    for sample in parsed_args.iisy.pareto_front:
        model_performance = model.ModelConfig(
            max_depth=sample.max_depth,
            f1_score=sample.f1_score,
            num_flows=2000000,  # placeholder
            total_features=sample.num_features,
        )
        db_manager.commit_to_logging_database(parsed_args, "iisy", 0, model_performance)

    # push results for our one-partition decision trees
    parsed_args.db_table_name = (
        "hypermapper"
        + "-"
        + parsed_args.dataset.name
        + "-"
        + parsed_args.hypermapper.scenario.optimization_method
    ).replace("-", "_")

    # push one-partition results
    for sample in parsed_args.one_partition.pareto_front:
        model_performance = model.ModelConfig(
            max_depth=sample.max_depth,
            features_per_partition=sample.num_features,
            num_partitions=ONE_PARTITION,
            f1_score=sample.f1_score,
            num_flows=sample.num_flows,
            total_features=sample.num_features,
            table_entries=sample.num_tcam_rules,
        )
        db_manager.commit_to_logging_database(parsed_args, "cap", 0, model_performance)


if __name__ == "__main__":
    # push_baseline_results()
    push_all_static_results()
