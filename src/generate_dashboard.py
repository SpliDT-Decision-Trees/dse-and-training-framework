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
import uuid

from box import Box

import utils


def main():
    parsed_args = utils.parse_yml_config()
    assert sum(parsed_args.operational_mode.values()) == 1, (
        "Error: Select just one operational mode"
    )

    # load the template dashboard json file
    template_path = os.path.join(
        parsed_args.HOME_DIRECTORY, parsed_args.PROJECT_ROOT, parsed_args.dashboards.template
    )
    with open(template_path, "r") as this_file:
        template_dashboard = Box(json.load(this_file))

    db_table_name = None
    if parsed_args.operational_mode.bruteforce:
        # bruteforce_cic_iomt_2024_pcaps0_f50
        db_table_name = ("bruteforce" + "-" + parsed_args.dataset.name).replace("-", "_").lower()

    elif parsed_args.operational_mode.hypermapper:
        # hypermapper_cic_iomt_2024_pcaps0_f50_bayesian_optimization
        db_table_name = (
            (
                "hypermapper"
                + "-"
                + parsed_args.dataset.name
                + "-"
                + parsed_args.hypermapper.scenario.optimization_method
            )
            .replace("-", "_")
            .lower()
        )

    else:
        return Exception(f"No mode selected for training")

    # for IIsy, Leo, and Netbeacon
    baseline_table_name = ("baseline" + "-" + parsed_args.dataset.name).replace("-", "_").lower()

    # update the dashboard title and generate a uid for it
    template_dashboard.title = db_table_name
    template_dashboard.uid = str(uuid.uuid4())

    # in each panel, for each target, change the query to the new table name
    for panel in template_dashboard.panels:
        for target in panel.targets:
            # print(target.rawSql)
            query_components = target.rawSql.split()
            # print(query_components)
            if "hypermapper" in query_components[6] or "bruteforce" in query_components[6]:
                query_components[6] = db_table_name
            else:
                query_components[6] = baseline_table_name
            target.rawSql = " ".join(query_components)
            print(target.rawSql)

    # write the new dashboard to provisioning directory
    write_path = os.path.join(
        parsed_args.HOME_DIRECTORY,
        parsed_args.PROJECT_ROOT,
        parsed_args.dashboards.provisioned,
        db_table_name + ".json",
    )
    with open(write_path, "w") as that_file:
        json.dump(template_dashboard.to_dict(), that_file, indent=4)

    print(f"Dashboard {db_table_name} generated successfully.")


if __name__ == "__main__":
    main()
