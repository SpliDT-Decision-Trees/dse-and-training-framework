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

import json
import os
from jinja2 import Environment, FileSystemLoader

# =========================
# OUTPUT DIRECTORIES
# =========================
P4SRC_DIR  = "p4src"
COMMON_DIR = "p4src/common"

os.makedirs(COMMON_DIR, exist_ok=True)

# =========================
# LOAD SPEC
# =========================
with open("spec.json") as f:
    spec = json.load(f)

num_flows = spec["num_flows"]
spec_num_features = spec["num_active_features"]

# FLOW SIZE POLICY
if num_flows > 200_000:
    GEN_NUM_FLOWS = 143300
    GEN_NUM_FLOWS_LARGE = 429900
else:
    GEN_NUM_FLOWS = num_flows
    GEN_NUM_FLOWS_LARGE = num_flows

# =========================
# FEATURE COUNT POLICY
# =========================
if num_flows >= 500_000:
    GEN_NUM_ACTIVE_FEATURES = 2
else:
    GEN_NUM_ACTIVE_FEATURES = spec_num_features

# =========================
# INGRESS REGISTER POLICY
# =========================
if num_flows < 200_000:
    GEN_REGS_PER_FEATURE = 1
elif num_flows <= 500_000:
    GEN_REGS_PER_FEATURE = 3
else:
    # >= 1,000,000
    GEN_REGS_PER_FEATURE = 6


# =========================
# JINJA ENV
# =========================
env = Environment(
    loader=FileSystemLoader("templates"),
    trim_blocks=True,
    lstrip_blocks=True
)

def render(template_name: str, out_dir: str, out_name: str):
    tmpl = env.get_template(template_name)
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w") as f:
        f.write(
            tmpl.render(
            **spec,
            GEN_NUM_FLOWS=GEN_NUM_FLOWS,
            GEN_NUM_FLOWS_LARGE=GEN_NUM_FLOWS_LARGE,
            GEN_NUM_ACTIVE_FEATURES=GEN_NUM_ACTIVE_FEATURES,
            GEN_REGS_PER_FEATURE=GEN_REGS_PER_FEATURE
        )
    )
    print(f"✔ generated {out_path}")

# =========================
# GENERATE FILES
# =========================

# Common includes
render("headers.p4.j2", COMMON_DIR, "headers.p4")
render("ingress.p4.j2", COMMON_DIR, "ingress.p4")
render("egress.p4.j2",  COMMON_DIR, "egress.p4")

# Top-level pipeline
render("decision_tree.p4.j2", P4SRC_DIR, "decision_tree.p4")

print("P4 generation complete")


