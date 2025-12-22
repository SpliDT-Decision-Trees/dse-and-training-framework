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
import shutil
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

import utils


@dataclass
class LeoNCModelConfig:
    max_depth: int = 0
    f1_score: float = 0.0
    total_features: int = 0

    def __str__(self):
        # return as a YAML list
        return (
            f"- max_depth: {self.max_depth}\n"
            + f"  f1_score: {self.f1_score}\n"
            + f"  num_features: {self.total_features}\n"
        )


# Get the top-k offloadable features
def select_top_k_features(dtree, num_features, all_features, remove_iat=False):
    importances = dtree.feature_importances_
    # Convert to DataFrame
    features_df = pd.DataFrame({"Feature": all_features, "Importance": importances})
    # this is a special case for ISCXVPN2016 that wouldn't extend to higher flows otherwise
    if remove_iat:
        # drop any feature containing 'IAT' in it
        features_df = features_df[~features_df["Feature"].str.contains("IAT")]
    top_k_features = features_df.sort_values(by="Importance", ascending=False).head(num_features)
    return top_k_features


def train_dt_no_constraints(X_train, y_train, X_test, y_test, max_depth, max_leaf_nodes):
    # Train the model
    dtree = DecisionTreeClassifier(
        random_state=42,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        criterion="entropy",
        class_weight="balanced",
    )
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)
    # Get classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return report, dtree


def train_dt_with_constraints(
    X_train, y_train, X_test, y_test, max_depth, max_leaf_nodes, num_features
):
    # train decision tree with all features
    _, dtree = train_dt_no_constraints(X_train, y_train, X_test, y_test, max_depth, max_leaf_nodes)

    # select top-k features
    # print(f"Selecting the top {num_features} features.")
    top_k_features = select_top_k_features(dtree, num_features, X_train.columns)
    # print(f"The features selected in this experiment are: {top_k_features}")

    # pick only these top-k features from train and test samples
    X_train_top_k = X_train[top_k_features["Feature"]]
    X_test_top_k = X_test[top_k_features["Feature"]]

    # train again with just these features
    final_report, final_dtree = train_dt_no_constraints(
        X_train_top_k, y_train, X_test_top_k, y_test, max_depth, max_leaf_nodes
    )

    return final_report, top_k_features, final_dtree


def main():
    parsed_args = utils.parse_yml_config()

    # record results here
    time_now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    results_path = os.path.join(
        parsed_args.HOME_DIRECTORY,
        parsed_args.PROJECT_ROOT,
        "results",
        f"leo_nc-{parsed_args.dataset.name}-{time_now}",
    )
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)
    print("Results will be saved to: ", results_path)

    dataset_path = os.path.join(
        parsed_args.dataset.path,
        parsed_args.dataset.name,
        parsed_args.dataset.destination,
        f"dataset_df_p1.pkl",  # hardcoded for Oracle
    )

    # Initialize logs and counters
    total = len(parsed_args.leo_nc.depths) * len(parsed_args.leo_nc.features)
    print(f"Total number of experiments to run: {total}")

    # print(f"Reading dataset from {dataset_path}")
    read_processed_df = utils.read_processed_dataset(dataset_path)
    ungrouped_training_dataset = read_processed_df["ungrouped_train_df"]
    ungrouped_testing_dataset = read_processed_df["ungrouped_test_df"]

    # get the training and testing dataset for first window for all flows
    X_train, y_train = utils.get_filtered_samples_and_labels(
        ungrouped_training_dataset, windows=[1]
    )

    X_test, y_test = utils.get_filtered_samples_and_labels(ungrouped_testing_dataset, windows=[1])

    print(f"Training dataset shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Testing dataset shape: {X_test.shape}")
    print(f"Testing labels shape: {y_test.shape}")

    # drop the 'Flow ID' and 'Window' columns
    X_train_without_id = X_train.drop(columns=["Flow ID", "Window"], axis=1)
    # get all unique features and labels
    all_features = X_train_without_id.columns
    all_labels = y_train.unique()
    # Replace inf values with NaN
    X_train_without_id = X_train_without_id.replace([np.inf, -np.inf], np.nan)
    # Identify rows with NaN values
    dropped_indices = X_train_without_id[X_train_without_id.isna().any(axis=1)].index
    # Drop rows with NaN and reset index
    X_train_without_id = X_train_without_id.drop(dropped_indices).reset_index(drop=True)
    # X_train = X_train.drop(dropped_indices).reset_index(drop=True)
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

    leo_nc_performance = LeoNCModelConfig()
    best_model_str = ""
    count = 0

    for max_depth in parsed_args.leo_nc.depths:
        for num_features in parsed_args.leo_nc.features:
            count += 1

            max_leaf_nodes = min(2**max_depth, 4096)
            # Run the model and evaluate the results
            classification_reports, top_k_features, trained_model = train_dt_with_constraints(
                X_train_without_id,
                y_train,
                X_test_without_id,
                y_test,
                max_depth,
                max_leaf_nodes,
                num_features,
            )

            features_used = top_k_features["Feature"].tolist()
            total_features_used = len(features_used)

            # get the macro avg f1 score from the classification report
            f1_score = round(classification_reports["macro avg"]["f1-score"], 2)
            print(f"({count}/{total}) The depth of the tree is {max_depth}", end="")
            print(f", the number of features used are {total_features_used}", end="")
            print(f" -> Macro avg f1 score: {f1_score}")

            result_str = f"Max Depth = {max_depth}, "
            result_str += f"Feature Limit = {total_features_used}, "
            result_str += f"Total Features = {total_features_used}, "
            result_str += f"Number of Partitions = 1, "
            result_str += f"F1 Score = {f1_score}, "
            result_str += f"Feature Table Entries = 0, "
            result_str += f"Tree Table Entries = nan, "
            result_str += f"Number of flows = nan, "
            result_str += f"Partition Sizes = {max_depth}, "
            result_str += f"Resubmission Traffic = 0, "
            result_str += f"Model Features = {features_used}\n"

            with open(os.path.join(results_path, f"results-d{max_depth}.txt"), "a") as results_file:
                results_file.write(result_str)

            # select the best model configuration
            if f1_score > leo_nc_performance.f1_score:
                best_model_str = result_str
                leo_nc_performance = LeoNCModelConfig(
                    max_depth=max_depth, f1_score=f1_score, total_features=total_features_used
                )

            pass
        pass

    print()
    print(f"Max score: {leo_nc_performance.f1_score}")
    print(best_model_str)
    print()
    print(leo_nc_performance)

    pass


if __name__ == "__main__":
    main()
