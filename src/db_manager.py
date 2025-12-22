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


import psycopg2

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


def create_logging_database(args):
    # Connect to the database
    conn = psycopg2.connect(**args.database)
    cur = conn.cursor()

    # Drop the table if it exists
    cur.execute(f"DROP TABLE IF EXISTS {args.db_table_name};")

    # Create a new table
    TABLE_SCHEMA = f"""
        CREATE TABLE {args.db_table_name} (
            name TEXT,
            iterations INT,
            f1_score FLOAT,
            feature_limit INT,
            feature_table_entries INT,
            max_depth INT,
            num_flows INT,
            num_partitions INT,
            total_features INT,
            tree_table_entries INT
        );
    """
    cur.execute(TABLE_SCHEMA)
    conn.commit()
    cur.close()
    conn.close()
    pass


def commit_to_logging_database(args, tag, iterations, model_metrics):
    conn = psycopg2.connect(**args.database)
    cur = conn.cursor()

    experiment_data = (
        tag,
        iterations,
        model_metrics.f1_score,
        model_metrics.features_per_partition,
        model_metrics.feature_entries,
        model_metrics.max_depth,
        model_metrics.num_flows,
        model_metrics.num_partitions,
        model_metrics.total_features,
        model_metrics.table_entries,
    )

    # INSERT_QUERY = f"""
    #     INSERT INTO {table_name} (name, iterations, f1_score, feature_limit, feature_table_entries, max_depth, num_flows, num_partitions, total_features, tree_table_entries)
    #     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    # """

    # SQL insert query
    print(f"Saving iteration {iterations} to table {args.db_table_name}")
    INSERT_QUERY = f"INSERT INTO {args.db_table_name}"
    INSERT_QUERY += (
        f" ({', '.join(ATTRIBUTES.keys())}) VALUES ({', '.join(['%s'] * len(ATTRIBUTES))})"
    )
    cur.execute(INSERT_QUERY, experiment_data)
    conn.commit()

    cur.close()
    conn.close()
    pass


def read_from_logging_database(args):
    conn = psycopg2.connect(**args.database)
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {args.db_table_name};")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows
