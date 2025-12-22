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


export PROJECT_ROOT = $(PWD)
export CICFLOWMETER_CONTAINER_REGISTRY = ghcr.io/splidt-decision-trees/cicflowmeter:latest
export GGPLOT_CONTAINER_REGISTRY = ghcr.io/splidt-decision-trees/ggplot:latest
export GRAFANA_CONTAINER_REGISTRY = ghcr.io/splidt-decision-trees/grafana:latest
export POSTGRES_CONTAINER_REGISTRY = ghcr.io/splidt-decision-trees/postgres:latest

# switch
export SWITCH_DIR = $(PROJECT_ROOT)/dataplane-driver
export SDE_DIR = /home/murayyiam/bf-sde-9.13.1
export CUSTOM_TOOLS_DIR = /home/murayyiam/tools
export P4_BUILD_SCRIPT = $(CUSTOM_TOOLS_DIR)/p4_build.sh
export VETH_UP_SCRIPT = veth_setup.sh
export VETH_DOWN_SCRIPT = veth_teardown.sh
export TOFINO_MODEL_SCRIPT = run_tofino_model.sh
export SWITCHD_SCRIPT = run_switchd.sh
export BFSHELL_SCRIPT = run_bfshell.sh
export P4_PROGRAM ?= decision_tree

export PYTHON = python3
export SUDO = echo $(USER_PASSWORD) | sudo -S

export RAW_DATASET_PATH = /mnt/sdb1/CIC-IOT-2023/PCAP
# variables that change with Make commands
CPU_CORES ?= 84

dev-cicflowmeter:
	docker run -it --rm --network host \
		-v $(PROJECT_ROOT)/src/:/CICFlowMeter/scripts \
		--entrypoint "/bin/bash" $(CICFLOWMETER_CONTAINER_REGISTRY)

with-volume-cicflowmeter:
	docker run -it --rm --network host \
		--cpuset-cpus="1-$(CPU_CORES)" \
		-v $(PROJECT_ROOT)/src/:/CICFlowMeter/scripts \
		-v $(RAW_DATASET_PATH):/CICFlowMeter/dataset -w /CICFlowMeter/ \
		--entrypoint "/bin/bash" $(CICFLOWMETER_CONTAINER_REGISTRY)

commit-cicflowmeter:
	docker commit \
		$(shell docker ps --filter "ancestor=$(CICFLOWMETER_CONTAINER_REGISTRY)" --format "{{.ID}}") \
		$(CICFLOWMETER_CONTAINER_REGISTRY)

pull-cicflowmeter:
	docker pull $(CICFLOWMETER_CONTAINER_REGISTRY)

push-cicflowmeter:
	docker push $(CICFLOWMETER_CONTAINER_REGISTRY)

start-ggplot-docker:
	docker run --privileged --rm -it \
		--name jupyter-ggplot \
		--network host \
		-v $(PROJECT_ROOT):/home/jovyan/work \
		$(GGPLOT_CONTAINER_REGISTRY) bash

commit-ggplot:
	docker commit \
		$(shell docker ps --filter "ancestor=$(GGPLOT_CONTAINER_REGISTRY)" --format "{{.ID}}") \
		$(GGPLOT_CONTAINER_REGISTRY)

pull-ggplot:
	docker pull $(GGPLOT_CONTAINER_REGISTRY)

push-ggplot:
	docker push $(GGPLOT_CONTAINER_REGISTRY)

dashboards-docker-up:
	cd $(PROJECT_ROOT)/grafana && docker compose up -d

restore-postgres-db:
	docker exec -it \
		$(shell docker ps --filter "ancestor=$(POSTGRES_CONTAINER_REGISTRY)" --format "{{.ID}}") \
		pg_restore -U postgres -h localhost -d decision_trees \
		-F c rawdb/decision_trees.dump

# will start grafana and postgres, then restore the postgres db
start-dashboards: dashboards-docker-up
	@echo "Waiting for Postgres to start..."
	@sleep 3
	$(MAKE) restore-postgres-db

# dumps two copies to stay safe
dump-postgres-db:
	docker exec -it \
		$(shell docker ps --filter "ancestor=$(POSTGRES_CONTAINER_REGISTRY)" --format "{{.ID}}") \
		pg_dump -U postgres -h localhost -d decision_trees \
		-F c -f rawdb/decision_trees.dump
	docker exec -it \
		$(shell docker ps --filter "ancestor=$(POSTGRES_CONTAINER_REGISTRY)" --format "{{.ID}}") \
		pg_dump -U postgres -h localhost -d decision_trees \
		-F c -f redundantdb/decision_trees.dump

# will dump the postgres db, then stop the grafana and postgres containers
stop-dashboards: dump-postgres-db
	cd $(PROJECT_ROOT)/grafana && docker compose down

# --user postgres
get-into-postgres-docker:
	docker exec -it \
		$(shell docker ps --filter "ancestor=$(POSTGRES_CONTAINER_REGISTRY)" --format "{{.ID}}") \
		bash

connect-to-postgres:
	docker exec -it \
		$(shell docker ps --filter "ancestor=$(POSTGRES_CONTAINER_REGISTRY)" --format "{{.ID}}") \
		psql -h localhost -p 5432 -U postgres

# shouldn't have to run this ever
commit-dashboards:
	docker commit \
		$(shell docker ps --filter "ancestor=$(POSTGRES_CONTAINER_REGISTRY)" --format "{{.ID}}") \
		$(POSTGRES_CONTAINER_REGISTRY)

# shouldn't have to run this ever
push-dashboards: 
	docker push $(POSTGRES_CONTAINER_REGISTRY)

# switch related commands -- only use for running the switch
build-p4:
	$(P4_BUILD_SCRIPT) \
		-p $(SWITCH_DIR)/p4src/$(P4_PROGRAM).p4 \
		-I $(SWITCH_DIR)/p4src/common -DMY_VAR=1

vinterfaces-up:
	cd $(CUSTOM_TOOLS_DIR) && $(SUDO) ./$(VETH_UP_SCRIPT)

vinterfaces-down:
	cd $(CUSTOM_TOOLS_DIR) && $(SUDO) ./$(VETH_DOWN_SCRIPT)

tofino-model:
	cd $(SDE_DIR) && ./$(TOFINO_MODEL_SCRIPT) -p $(P4_PROGRAM)

switchd:
	cd $(SDE_DIR) && ./$(SWITCHD_SCRIPT) -p $(P4_PROGRAM)

bfshell:
	cd $(SDE_DIR) && ./$(BFSHELL_SCRIPT)

DIR_PATH ?= ISCXVPN2016-PCAPS0-f10
MODEL_NAME ?= d10_np2_fl2
bfshell-script:
	cd $(SDE_DIR) && ./$(BFSHELL_SCRIPT) -b $(SWITCH_DIR)/bfrt_python/control_plane/bfrt-controller.py -i --dir_path $(DIR_PATH) --model_name $(MODEL_NAME)

controller:
	$(PYTHON) $(SWITCH_DIR)/bfrt_python/control_plane/controller.py