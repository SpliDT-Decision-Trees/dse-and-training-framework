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
import ipaddress
import json
import logging
import os
import pickle
import re
import sys
import time

import numpy as np

# Globals injected by bfshell
# - bfrt   : top-level handle
# - P4     : pipeline handle after assignment below

logger = logging.getLogger("bfrt-controller")
logging.basicConfig(level=logging.INFO)

# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
P4 = bfrt.decision_tree.pipe  # attach to your program pipeline

FEATURE_TO_ACTION = {
    "Flow IAT Max": {
        1: "add_with_stateless_operand_load_iat_f1",
        2: "add_with_stateless_operand_load_iat_f2",
        3: "add_with_stateless_operand_load_iat_f3",
    },
    "Flow IAT Min": {
        1: "add_with_stateless_operand_load_iat_f1",
        2: "add_with_stateless_operand_load_iat_f2",
        3: "add_with_stateless_operand_load_iat_f3",
    },
    "Flow IAT Total": {
        1: "add_with_stateless_operand_load_iat_f1",
        2: "add_with_stateless_operand_load_iat_f2",
        3: "add_with_stateless_operand_load_iat_f3",
    },
    "Fwd IAT Max": {
        1: "add_with_stateless_operand_load_iat_f1",
        2: "add_with_stateless_operand_load_iat_f2",
        3: "add_with_stateless_operand_load_iat_f3",
    },
    "Fwd IAT Min": {
        1: "add_with_stateless_operand_load_iat_f1",
        2: "add_with_stateless_operand_load_iat_f2",
        3: "add_with_stateless_operand_load_iat_f3",
    },
    "Fwd IAT Total": {
        1: "add_with_stateless_operand_load_iat_f1",
        2: "add_with_stateless_operand_load_iat_f2",
        3: "add_with_stateless_operand_load_iat_f3",
    },
    "Bwd IAT Max": {
        1: "add_with_stateless_operand_load_iat_f1",
        2: "add_with_stateless_operand_load_iat_f2",
        3: "add_with_stateless_operand_load_iat_f3",
    },
    "Bwd IAT Min": {
        1: "add_with_stateless_operand_load_iat_f1",
        2: "add_with_stateless_operand_load_iat_f2",
        3: "add_with_stateless_operand_load_iat_f3",
    },
    "Bwd IAT Total": {
        1: "add_with_stateless_operand_load_iat_f1",
        2: "add_with_stateless_operand_load_iat_f2",
        3: "add_with_stateless_operand_load_iat_f3",
    },
    "Flow Duration": {
        1: "add_with_stateless_operand_load_flow_duration_f1",
        2: "add_with_stateless_operand_load_flow_duration_f2",
        3: "add_with_stateless_operand_load_flow_duration_f3",
    },
    "Packet Length Min": {
        1: "add_with_stateless_operand_load_pkt_len_f1",
        2: "add_with_stateless_operand_load_pkt_len_f2",
        3: "add_with_stateless_operand_load_pkt_len_f3",
    },
    "Packet Length Max": {
        1: "add_with_stateless_operand_load_pkt_len_f1",
        2: "add_with_stateless_operand_load_pkt_len_f2",
        3: "add_with_stateless_operand_load_pkt_len_f3",
    },
    "total Length of Fwd Packet": {
        1: "add_with_stateless_operand_load_pkt_len_f1",
        2: "add_with_stateless_operand_load_pkt_len_f2",
        3: "add_with_stateless_operand_load_pkt_len_f3",
    },
    "total Length of Bwd Packet": {
        1: "add_with_stateless_operand_load_pkt_len_f1",
        2: "add_with_stateless_operand_load_pkt_len_f2",
        3: "add_with_stateless_operand_load_pkt_len_f3",
    },
    "Fwd Packet Length Min": {
        1: "add_with_stateless_operand_load_pkt_len_f1",
        2: "add_with_stateless_operand_load_pkt_len_f2",
        3: "add_with_stateless_operand_load_pkt_len_f3",
    },
    "Fwd Packet Length Max": {
        1: "add_with_stateless_operand_load_pkt_len_f1",
        2: "add_with_stateless_operand_load_pkt_len_f2",
        3: "add_with_stateless_operand_load_pkt_len_f3",
    },
    "Bwd Packet Length Min": {
        1: "add_with_stateless_operand_load_pkt_len_f1",
        2: "add_with_stateless_operand_load_pkt_len_f2",
        3: "add_with_stateless_operand_load_pkt_len_f3",
    },
    "Bwd Packet Length Max": {
        1: "add_with_stateless_operand_load_pkt_len_f1",
        2: "add_with_stateless_operand_load_pkt_len_f2",
        3: "add_with_stateless_operand_load_pkt_len_f3",
    },
    "total Fwd Packet": {
        1: "add_with_stateless_operand_load_count_f1",
        2: "add_with_stateless_operand_load_count_f2",
        3: "add_with_stateless_operand_load_count_f3",
    },
    "total Bwd packets": {
        1: "add_with_stateless_operand_load_count_f1",
        2: "add_with_stateless_operand_load_count_f2",
        3: "add_with_stateless_operand_load_count_f3",
    },
    "Fwd Header Length": {
        1: "add_with_stateless_operand_load_hdr_len_f1",
        2: "add_with_stateless_operand_load_hdr_len_f2",
        3: "add_with_stateless_operand_load_hdr_len_f3",
    },
    "Bwd Header Length": {
        1: "add_with_stateless_operand_load_hdr_len_f1",
        2: "add_with_stateless_operand_load_hdr_len_f2",
        3: "add_with_stateless_operand_load_hdr_len_f3",
    },
    "FIN Flag Count": {
        1: "add_with_stateless_operand_load_count_f1",
        2: "add_with_stateless_operand_load_count_f2",
        3: "add_with_stateless_operand_load_count_f3",
    },
    "SYN Flag Count": {
        1: "add_with_stateless_operand_load_count_f1",
        2: "add_with_stateless_operand_load_count_f2",
        3: "add_with_stateless_operand_load_count_f3",
    },
    "RST Flag Count": {
        1: "add_with_stateless_operand_load_count_f1",
        2: "add_with_stateless_operand_load_count_f2",
        3: "add_with_stateless_operand_load_count_f3",
    },
    "PSH Flag Count": {
        1: "add_with_stateless_operand_load_count_f1",
        2: "add_with_stateless_operand_load_count_f2",
        3: "add_with_stateless_operand_load_count_f3",
    },
    "ACK Flag Count": {
        1: "add_with_stateless_operand_load_count_f1",
        2: "add_with_stateless_operand_load_count_f2",
        3: "add_with_stateless_operand_load_count_f3",
    },
    "URG Flag Count": {
        1: "add_with_stateless_operand_load_count_f1",
        2: "add_with_stateless_operand_load_count_f2",
        3: "add_with_stateless_operand_load_count_f3",
    },
    "CWR Flag Count": {
        1: "add_with_stateless_operand_load_count_f1",
        2: "add_with_stateless_operand_load_count_f2",
        3: "add_with_stateless_operand_load_count_f3",
    },
    "ECE Flag Count": {
        1: "add_with_stateless_operand_load_count_f1",
        2: "add_with_stateless_operand_load_count_f2",
        3: "add_with_stateless_operand_load_count_f3",
    },
    "Fwd Act Data Pkts": {
        1: "add_with_stateless_operand_load_count_f1",
        2: "add_with_stateless_operand_load_count_f2",
        3: "add_with_stateless_operand_load_count_f3",
    },
    "Fwd Seg Size Min": {
        1: "add_with_stateless_operand_load_count_f1",
        2: "add_with_stateless_operand_load_count_f2",
        3: "add_with_stateless_operand_load_count_f3",
    },
    "Dst Port": {
        1: "add_with_stateless_operand_load_dst_port_f1",
        2: "add_with_stateless_operand_load_dst_port_f2",
        3: "add_with_stateless_operand_load_dst_port_f3",
    },
}


# Helpers
def ip_int_to_str(ip_int):
    return str(ipaddress.IPv4Address(int(ip_int)))


def clear_all():
    """Clear all entries in relevant tables/registers."""
    P4.Ingress.f1_table.clear()
    P4.Ingress.f2_table.clear()
    P4.Ingress.classifier.clear()
    logger.info("Cleared all P4 tables")


# Feature Table Installation
def install_feature_table_rules(feat_table_datas_flow, sid):
    print("Installing feature table rules")
    rules_installed = 0

    feature_map = {
        0: (P4.Ingress.f1_table, "f1_encode", "f1"),
        1: (P4.Ingress.f2_table, "f2_encode", "f2"),
        # Add more features here later if needed
    }

    for feature_name, entries in feat_table_datas_flow.items():
        if feature_name not in feature_map:
            logger.warning(f"Unknown feature table: {feature_name}, skipping")
            continue

        table, action, field = feature_map[feature_name]

        for entry in entries:
            try:
                priority = int(entry[0])
                value = int(entry[1])
                mask = int(entry[2]) if len(entry) > 2 else 0xFFFF
                encode = int(entry[3])

                kwargs = {
                    "sid": sid,
                    field: value,
                    f"{field}_mask": mask,
                    "val": encode,
                    "MATCH_PRIORITY": priority,
                }

                getattr(table, f"add_with_{action}")(**kwargs)

                rules_installed += 1
                logger.info(
                    f"Installed Feature {feature_name} rule (val={value}, mask={mask}, sid={sid})"
                )

            except Exception as e:
                logger.error(f"Failed to install {feature_name} rule: {e}")

    bfrt.complete_operations()
    return rules_installed


# Classifier Table Installation
def install_classifier_table_rules(tree_data_p2p_flow, sid):
    print("Installing classifier table rules")
    rules_installed = 0

    table = P4.Ingress.classifier
    action = "set_next_sid"

    for entry in tree_data_p2p_flow:
        try:
            # FEATURE 1 (always exists)
            f1_value = int(entry[0])
            f1_mask = int(entry[1])

            # FEATURE 2 (only if present)
            if len(entry) > 3:
                f2_value = int(entry[2])
                f2_mask = int(entry[3])
            else:
                f2_value = 0
                f2_mask = 0xFFFF  # wildcard

            # CLASS INFO
            classinfo = entry[-1]  # last element
            next_subtree_id = classinfo["next_subtree_id"]
            class_probs = classinfo["class_probs"]

            if next_subtree_id is not None:
                next_sid = int(next_subtree_id)
            else:
                next_sid = int(np.argmax(class_probs))

            kwargs = {
                "sid": sid,
                "f1_encoded": f1_value,
                "f1_encoded_mask": f1_mask,
                "f2_encoded": f2_value,
                "f2_encoded_mask": f2_mask,
                "val": next_sid,
                "MATCH_PRIORITY": 1,
            }

            getattr(table, f"add_with_{action}")(**kwargs)

            rules_installed += 1
            logger.info(f"Installed classifier rule sid={sid} → next_sid={next_sid}")

        except Exception as e:
            logger.error(f"Failed classifier rule: {e}")

    bfrt.complete_operations()
    return rules_installed


def install_sid_operand_rules(sid, base_path="."):
    # read JSON
    json_file = os.path.join(base_path, f"subtree_{sid}_feature_map.json")
    with open(json_file, "r") as f:
        data = json.load(f)

    feature_map = data["feature_map"]

    # iterate by sorted order: feature_0, feature_1, feature_2,...
    for idx, (feature_key, feature_name) in enumerate(sorted(feature_map.items())):
        operand_id = idx + 1  # because f1,f2,f3 correspond to 1,2,3

        # choose correct P4 table
        table = {
            1: P4.Ingress.f1_op_load_table,
            2: P4.Ingress.f2_op_load_table,
        }.get(operand_id, None)

        if table is None:
            print(f"[WARN] No table defined for operand #{operand_id}")
            continue

        # choose correct action
        action_name = FEATURE_TO_ACTION.get(feature_name, {}).get(operand_id, "add_with_NoAction")
        print(
            f"[DEBUG] feature_name={feature_name}, operand_id={operand_id}, action_name={action_name}"
        )

        try:
            action_fn = getattr(table, action_name)
        except AttributeError:
            print(f"[ERROR] Action '{action_name}' not found in table for operand {operand_id}")
            continue

        print(
            f"[INFO] SID={sid} → {feature_key} → {feature_name} → f{operand_id}_op_load_table → {action_name}"
        )

        # install
        action_fn(sid=sid)


def install_sid_stateful_rules(sid, base_path="."):
    json_file = os.path.join(base_path, f"subtree_{sid}_feature_map.json")
    if not os.path.exists(json_file):
        print(f"[WARN] No mapping file for sid={sid}")
        return

    with open(json_file, "r") as f:
        data = json.load(f)

    feature_map = data["feature_map"]

    # Ports
    FWD_PORT = 1
    BWD_PORT = 2
    RECIRC_PORT = 68

    # TCP flag → ctrl bit
    FLAG_VALUE = {
        "FIN Flag Count": 0x01,
        "SYN Flag Count": 0x02,
        "RST Flag Count": 0x04,
        "PSH Flag Count": 0x08,
        "ACK Flag Count": 0x10,
        "URG Flag Count": 0x20,
        "ECE Flag Count": 0x40,
        "CWR Flag Count": 0x80,
    }

    # Feature → operation
    FEATURE_TO_OP = {
        "Flow IAT Max": "max",
        "Flow IAT Min": "min",
        "Flow IAT Total": "sum",
        "Fwd IAT Max": "max",
        "Fwd IAT Min": "min",
        "Fwd IAT Total": "sum",
        "Bwd IAT Max": "max",
        "Bwd IAT Min": "min",
        "Bwd IAT Total": "sum",
        "Flow Duration": "sum",
        "Packet Length Min": "min",
        "Packet Length Max": "max",
        "total Length of Fwd Packet": "sum",
        "total Length of Bwd Packet": "sum",
        "Fwd Packet Length Min": "min",
        "Fwd Packet Length Max": "max",
        "Bwd Packet Length Min": "min",
        "Bwd Packet Length Max": "max",
        "total Fwd Packet": "sum",
        "total Bwd packets": "sum",
        "Fwd Header Length": "sum",
        "Bwd Header Length": "sum",
        "FIN Flag Count": "sum",
        "SYN Flag Count": "sum",
        "RST Flag Count": "sum",
        "PSH Flag Count": "sum",
        "ACK Flag Count": "sum",
        "URG Flag Count": "sum",
        "ECE Flag Count": "sum",
        "CWR Flag Count": "sum",
        "Fwd Act Data Pkts": "sum",
        "Fwd Seg Size Min": "min",
    }

    # Operand → state_index → table
    OPERAND_TABLES = {
        1: {
            1: P4.Ingress.f11_op_table,
            2: P4.Ingress.f12_op_table,
            3: P4.Ingress.f13_op_table,
        },
        2: {
            1: P4.Ingress.f21_op_table,
            2: P4.Ingress.f22_op_table,
            3: P4.Ingress.f23_op_table,
        },
    }

    # Operand → state_index → op → action
    ACTION_NAME = {
        1: {
            1: {
                "sum": "add_with_stateful_action_sum_f11",
                "min": "add_with_stateful_action_min_f11",
                "max": "add_with_stateful_action_max_f11",
                "init": "add_with_stateful_action_init_f11",
            },
            2: {
                "sum": "add_with_stateful_action_sum_f12",
                "min": "add_with_stateful_action_min_f12",
                "max": "add_with_stateful_action_max_f12",
                "init": "add_with_stateful_action_init_f12",
            },
            3: {
                "sum": "add_with_stateful_action_sum_f13",
                "min": "add_with_stateful_action_min_f13",
                "max": "add_with_stateful_action_max_f13",
                "init": "add_with_stateful_action_init_f13",
            },
        },
        2: {
            1: {
                "sum": "add_with_stateful_action_sum_f21",
                "min": "add_with_stateful_action_min_f21",
                "max": "add_with_stateful_action_max_f21",
                "init": "add_with_stateful_action_init_f21",
            },
            2: {
                "sum": "add_with_stateful_action_sum_f22",
                "min": "add_with_stateful_action_min_f22",
                "max": "add_with_stateful_action_max_f22",
                "init": "add_with_stateful_action_init_f22",
            },
            3: {
                "sum": "add_with_stateful_action_sum_f23",
                "min": "add_with_stateful_action_min_f23",
                "max": "add_with_stateful_action_max_f23",
                "init": "add_with_stateful_action_init_f23",
            },
        },
    }

    print(f"[INFO] Installing stateful rules for SID={sid}")

    # INIT rules on recirculation port
    for operand_id, tables in OPERAND_TABLES.items():
        for state_idx, table in tables.items():
            init_action = ACTION_NAME[operand_id][state_idx]["init"]
            getattr(table, init_action)(sid=sid, ingress_port=RECIRC_PORT, ctrl=0)
            print(f"[INIT] SID={sid} operand={operand_id} state={state_idx}")

    # Feature-specific rules
    for idx, (_, feature_name) in enumerate(sorted(feature_map.items())):
        operand_id = idx + 1

        if operand_id not in OPERAND_TABLES:
            continue

        # FLAG FEATURES
        if feature_name in FLAG_VALUE:
            flag_val = FLAG_VALUE[feature_name]
            for state_idx, table in OPERAND_TABLES[operand_id].items():
                action = ACTION_NAME[operand_id][state_idx]["sum"]
                getattr(table, action)(sid=sid, ingress_port=0, ctrl=flag_val)
            continue

        if feature_name not in FEATURE_TO_OP:
            print(f"[WARN] Unknown feature {feature_name}, skipping")
            continue

        op = FEATURE_TO_OP[feature_name]

        fname = feature_name.lower()
        if "fwd" in fname:
            port = FWD_PORT
        elif "bwd" in fname:
            port = BWD_PORT
        else:
            port = 0

        for state_idx, table in OPERAND_TABLES[operand_id].items():
            action = ACTION_NAME[operand_id][state_idx][op]
            getattr(table, action)(sid=sid, ingress_port=port, ctrl=0)

    bfrt.complete_operations()
    print(f"[DONE] Installed stateful rules for SID={sid}")


# Digest Handling
def digest_cb(dev_id, pipe_id, direction, parser_id, session, msg):
    global DT

    print(f"> Incoming Digests ({len(msg)}): ")
    installed_rules = 0
    for digest in msg:
        print(json.dumps(digest, indent=4))

    return 0


# Load Pickle Rules
def load_pkl_rules(sid):
    logger.info(f"Loading pickled rules for SID {sid}")
    filename = f"subtree_{sid}_model.pkl"
    path = os.path.join(DIR_PATH, filename)
    try:
        with open(path, "rb") as f:
            # [feat_table_datas_flow, tree_data_p2p_flow] = pickle.load(f)
            data = pickle.load(f)

        feat_table_datas_flow = data["feature_table_entries"]
        tree_data_p2p_flow = data["tree_table_entries"]

        install_feature_table_rules(feat_table_datas_flow, sid)
        install_classifier_table_rules(tree_data_p2p_flow, sid)
        install_sid_operand_rules(sid, base_path=DIR_PATH)
        install_sid_stateful_rules(sid, base_path=DIR_PATH)
        return True
    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
        return False


# Controller Entry
def controller(dir_path):
    print("> Starting controller...")
    clear_all()

    # Load all pickled rule files
    pattern = re.compile(r"subtree_(\d+)_model.pkl")
    for filename in os.listdir(dir_path):
        logger.info(f"Checking file: {filename}")
        match = pattern.match(filename)
        if match:
            sid = int(match.group(1))
            load_pkl_rules(sid)

    # Register digest callback
    try:
        P4.IngressDeparser.digest_a.callback_deregister()
    except:
        pass

    P4.IngressDeparser.digest_a.callback_register(digest_cb)

    bfrt.complete_operations()
    logger.info("Controller ready, listening for digests...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "models", "ISCXVPN2016-PCAPS0-f10", "d10_np2_fl2"),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="d10_np2_fl2",
    )
    args = parser.parse_args()
    DIR_PATH = os.path.join(PROJECT_ROOT, "models", args.dir_path, args.model_name)
    controller(DIR_PATH)


if __name__ == "__main__":
    main()
