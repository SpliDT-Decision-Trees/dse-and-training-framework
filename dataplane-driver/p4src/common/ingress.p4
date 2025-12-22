/*****************************************************************************
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
# *****************************************************************************/


/*************************************************************************
************************ INGRESS *****************************************
*************************************************************************/
parser IngressParser(
    packet_in pkt,
    out headers_t hdr,
    out metadata_t ig_md,
    out ingress_intrinsic_metadata_t ig_intr_md) {
    
    state start {
        pkt.extract(ig_intr_md);
        pkt.advance(PORT_METADATA_SIZE);
        transition select(ig_intr_md.ingress_port) {
            RECIRCULATE_PORT : parse_resubmit;
            default          : parse_ethernet;
        }
    }
    state parse_resubmit{
        pkt.extract(hdr.resubmit_hdr);
        transition accept;
        // transition parse_ethernet;  // keep parsing real headers after resubmit_hdr
    }
    
    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            TYPE_IPV4: parse_ipv4;
            default: accept;
        }
    }
    
    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol){
            6: parse_tcp;
            17: parse_udp;
            default: parse_features;
        }
    }

    state parse_tcp {
        pkt.extract(hdr.tcp);
        transition parse_features;
    }

    state parse_udp {
        pkt.extract(hdr.udp);
        transition parse_features;
    }

    state parse_features {
        pkt.extract(hdr.features);
        transition accept;
    }

}

control Ingress(
    inout headers_t hdr,
    inout metadata_t ig_md,
    in ingress_intrinsic_metadata_t ig_intr_md,
    in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {

    // mandatory hash calculation
    Hash<bit<KEY_SIZE>>(HashAlgorithm_t.CRC32) crc32;
    action hash_packet(bit<32> ipAddr1, bit<32> ipAddr2, bit<16> port1, bit<16> port2, bit<8> proto) {
        ig_md.flow_hash = crc32.get(
            {ipAddr1, ipAddr2, port1, port2, proto}
            );
    }


    // Registers
    Register<bit<FEATURE_WIDTH_8>, bit<KEY_SIZE>>(NUM_FLOWS_LARGE) sid_register;          // SID per flow window
    Register<bit<FEATURE_WIDTH_8>, bit<KEY_SIZE>>(NUM_FLOWS_LARGE) pkt_count_register;    // pkt count per flow

    // Number of features per subtree (3)
    Register<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>>(NUM_FLOWS) f11_register; 
    Register<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>>(NUM_FLOWS) f12_register;
    Register<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>>(NUM_FLOWS) f13_register;

    Register<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>>(NUM_FLOWS) f21_register;
    Register<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>>(NUM_FLOWS) f22_register;
    Register<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>>(NUM_FLOWS) f23_register;

    Register<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>>(NUM_FLOWS) f31_register;
    Register<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>>(NUM_FLOWS) f32_register;
    Register<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>>(NUM_FLOWS) f33_register; 
    

    // Register Actions

    // Action to increment the packet counter
    RegisterAction<bit<FEATURE_WIDTH_8>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_8>>(pkt_count_register) increment_counter = {
        void apply(inout bit<FEATURE_WIDTH_8> value, out bit<FEATURE_WIDTH_8> read_value){
            
            value = value + 1;
            read_value = value;
        }
    };

    // Action to take sum of the feature values in feat1 
    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f11_register) feat11_sum = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            value = value + ig_md.f1;
            read_value = value;
        }
    };

    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f12_register) feat12_sum = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            value = value + ig_md.f1;
            read_value = value;
        }
    };

    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f13_register) feat13_sum = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            value = value + ig_md.f1;
            read_value = value;
        }
    };


    // Action to take max of the feature values in feat1
    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f11_register) feat11_max = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            if (value < ig_md.f1) {
                value = ig_md.f1;
            }
            read_value = value;
        }
    };

    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f12_register) feat12_max = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            if (value < ig_md.f1) {
                value = ig_md.f1;
            }
            read_value = value;
        }
    };

    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f13_register) feat13_max = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            if (value < ig_md.f1) {
                value = ig_md.f1;
            }
            read_value = value;
        }
    };

    // Action to take min of the feature values in feat1
    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f11_register) feat11_min = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            if (value > ig_md.f1 || value==0 ) {
                value = ig_md.f1;
            }
            read_value = value;
        }
    };

    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f12_register) feat12_min = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            if (value > ig_md.f1 || value==0 ) {
                value = ig_md.f1;
            }
            read_value = value;
        }
    };

    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f13_register) feat13_min = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            if (value > ig_md.f1 || value==0 ) {
                value = ig_md.f1;
            }
            read_value = value;
        }
    };

    // Action to init of the feature values in feat1
    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f11_register) feat11_init = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            value = 0;
            read_value = value;
        }
    };

    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f12_register) feat12_init = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            value = 0;
            read_value = value;
        }
    };

    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f13_register) feat13_init = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            value = 0;
            read_value = value;
        }
    };
    

    // Action to take sum of the feature values in feat2 
    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f21_register) feat21_sum = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            value = value + ig_md.f2;
            read_value = value;
        }
    };

    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f22_register) feat22_sum = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            value = value + ig_md.f2;
            read_value = value;
        }
    };

    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f23_register) feat23_sum = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            value = value + ig_md.f2;
            read_value = value;
        }
    };

    // Action to take max of the feature values in feat2
    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f21_register) feat21_max = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            if (value < ig_md.f2) {
                value = ig_md.f2;
            }
            read_value = value;
        }
    };

    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f22_register) feat22_max = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            if (value < ig_md.f2) {
                value = ig_md.f2;
            }
            read_value = value;
        }
    };

    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f23_register) feat23_max = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            if (value < ig_md.f2) {
                value = ig_md.f2;
            }
            read_value = value;
        }
    };

    // Action to take min of the feature values in feat2
    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f21_register) feat21_min = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            if (value > ig_md.f2 || value==0 ) {
                value = ig_md.f2;
            }
            read_value = value;
        }
    };

    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f22_register) feat22_min = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            if (value > ig_md.f2 || value==0 ) {
                value = ig_md.f2;
            }
            read_value = value;
        }
    };

    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f23_register) feat23_min = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            if (value > ig_md.f2 || value==0 ) {
                value = ig_md.f2;
            }
            read_value = value;
        }
    };

    // Action to init of the feature values in feat2
    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f21_register) feat21_init = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            value = 0;
            read_value = value;
        }
    };

    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f22_register) feat22_init = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            value = 0;
            read_value = value;
        }
    };

    RegisterAction<bit<FEATURE_WIDTH_32>, bit<KEY_SIZE>, bit<FEATURE_WIDTH_32>>(f23_register) feat23_init = {
        void apply(inout bit<FEATURE_WIDTH_32> value, out bit<FEATURE_WIDTH_32> read_value){
            value = 0;
            read_value = value;
        }
    };
    
    // Actions
    
    action f1_encode(bit<32> val) { ig_md.f1_encoded = val; }
    action f2_encode(bit<32> val) { ig_md.f2_encoded = val; }
    // action f3_encode(bit<16> val) { ig_md.f3_encoded = val; }

    action set_next_sid(bit<FEATURE_WIDTH_8> val) { 
        ig_md.next_sid_value = val; 
        hdr.resubmit_hdr.sid = val;
    }

    action emit_digest() {
        hdr.digest_a.setValid();
        hdr.digest_a.sid = ig_md.next_sid_value;
        // ig_tm_md.ucast_egress_port = (bit<9>) CPU_PORT; // send to CPU
        ig_dprsr_md.digest_type = L2_LEARN_DIGEST; // mark digest
    }

    action recirculate_packet() {
        hdr.resubmit_hdr.setValid();
        hdr.resubmit_hdr.srcAddr = ig_md.src_addr;
        hdr.resubmit_hdr.dstAddr = ig_md.dst_addr;
        hdr.resubmit_hdr.src_port = ig_md.src_port;
        hdr.resubmit_hdr.dst_port = ig_md.dst_port;
        hdr.resubmit_hdr.protocol = hdr.ipv4.protocol;
        // hdr.resubmit_hdr.sid = ig_md.next_sid_value;
        ig_tm_md.ucast_egress_port = RECIRCULATE_PORT;
    }

    
    
    action stateful_action_sum_f11() {
        ig_md.f1 = feat11_sum.execute(ig_md.flow_hash);
    }

    action stateful_action_sum_f12() {
        ig_md.f1 = feat12_sum.execute(ig_md.flow_hash);
    }

    action stateful_action_sum_f13() {
        ig_md.f1 = feat13_sum.execute(ig_md.flow_hash);
    }

    action stateful_action_init_f11() {
        ig_md.f1 = feat11_init.execute(ig_md.flow_hash);
    }

    action stateful_action_init_f12() {
        ig_md.f1 = feat12_init.execute(ig_md.flow_hash);
    }

    action stateful_action_init_f13() {
        ig_md.f1 = feat13_init.execute(ig_md.flow_hash);
    }

    action stateful_action_min_f11() {
        ig_md.f1 = feat11_min.execute(ig_md.flow_hash);
    }

    action stateful_action_min_f12() {
        ig_md.f1 = feat12_min.execute(ig_md.flow_hash);
    }

    action stateful_action_min_f13() {
        ig_md.f1 = feat13_min.execute(ig_md.flow_hash);
    }

    action stateful_action_max_f11() {
        ig_md.f1 = feat11_max.execute(ig_md.flow_hash);
    }

    action stateful_action_max_f12() {
        ig_md.f1 = feat12_max.execute(ig_md.flow_hash);
    }

    action stateful_action_max_f13() {
        ig_md.f1 = feat13_max.execute(ig_md.flow_hash);
    }
    
    
    action stateful_action_sum_f21() {
        ig_md.f2 = feat21_sum.execute(ig_md.flow_hash);
    }

    action stateful_action_sum_f22() {
        ig_md.f2 = feat22_sum.execute(ig_md.flow_hash);
    }

    action stateful_action_sum_f23() {
        ig_md.f2 = feat23_sum.execute(ig_md.flow_hash);
    }

    action stateful_action_init_f21() {
        ig_md.f2 = feat21_init.execute(ig_md.flow_hash);
    }

    action stateful_action_init_f22() {
        ig_md.f2 = feat22_init.execute(ig_md.flow_hash);
    }

    action stateful_action_init_f23() {
        ig_md.f2 = feat23_init.execute(ig_md.flow_hash);
    }

    action stateful_action_min_f21() {
        ig_md.f2 = feat21_min.execute(ig_md.flow_hash);
    }

    action stateful_action_min_f22() {
        ig_md.f2 = feat22_min.execute(ig_md.flow_hash);
    }

    action stateful_action_min_f23() {
        ig_md.f2 = feat23_min.execute(ig_md.flow_hash);
    }

    action stateful_action_max_f21() {
        ig_md.f2 = feat21_max.execute(ig_md.flow_hash);
    }

    action stateful_action_max_f22() {
        ig_md.f2 = feat22_max.execute(ig_md.flow_hash);
    }

    action stateful_action_max_f23() {
        ig_md.f2 = feat23_max.execute(ig_md.flow_hash);
    }

    
    // stateless operand load actions
    action stateless_operand_load_iat_f1() {
        ig_md.f1 = hdr.features.iat;
    }

    action stateless_operand_load_pkt_len_f1() {
        ig_md.f1 = hdr.features.pkt_len;
    }

    action stateless_operand_load_hdr_len_f1() {
        ig_md.f1 = hdr.features.hdr_len;
    }

    action stateless_operand_load_count_f1() {
        ig_md.f1 = 1;
    }

    action stateless_operand_load_flow_duration_f1() {
        ig_md.f1 = hdr.features.flow_duration;
    }

    action stateless_operand_load_dst_port_f1() {
        ig_md.f1 = hdr.features.dst_port;
    }
    
    
    action stateless_operand_load_iat_f2() {
        ig_md.f2 = hdr.features.iat;
    }

    action stateless_operand_load_pkt_len_f2() {
        ig_md.f2 = hdr.features.pkt_len;
    }

    action stateless_operand_load_hdr_len_f2() {
        ig_md.f2 = hdr.features.hdr_len;
    }

    action stateless_operand_load_count_f2() {
        ig_md.f2 = 1;
    }

    action stateless_operand_load_flow_duration_f2() {
        ig_md.f2 = hdr.features.flow_duration;
    }

    action stateless_operand_load_dst_port_f2() {
        ig_md.f2 = hdr.features.dst_port;
    }

    action stateless_operand_load_iat_f3() {
        ig_md.f3 = hdr.features.iat;
    }

    action stateless_operand_load_pkt_len_f3() {
        ig_md.f3 = hdr.features.pkt_len;
    }

    action stateless_operand_load_hdr_len_f3() {
        ig_md.f3 = hdr.features.hdr_len;
    }

    action stateless_operand_load_count_f3() {
        ig_md.f3 = 1;
    }

    action stateless_operand_load_flow_duration_f3() {
        ig_md.f3 = hdr.features.flow_duration;
    }

    action stateless_operand_load_dst_port_f3() {
        ig_md.f3 = hdr.features.dst_port;
    }
    

    // Tables
    table f1_op_load_table {
        key = { 
            ig_md.sid : exact; 
        }
        actions = { 
            stateless_operand_load_iat_f1;
            stateless_operand_load_pkt_len_f1;
            stateless_operand_load_hdr_len_f1;
            stateless_operand_load_count_f1;
            stateless_operand_load_flow_duration_f1;
            NoAction; 
        }
        size = 33;
        default_action = NoAction();
    }
    
    
    table f2_op_load_table {
        key = { 
            ig_md.sid : exact; 
        }
        actions = { 
            stateless_operand_load_iat_f2;
            stateless_operand_load_pkt_len_f2;
            stateless_operand_load_hdr_len_f2;
            stateless_operand_load_count_f2;
            stateless_operand_load_flow_duration_f2;
            NoAction; 
        }
        size = 33;
        default_action = NoAction();
    }

    table f3_op_load_table {
        key = { 
            ig_md.sid : exact; 
        }
        actions = { 
            stateless_operand_load_iat_f3;
            stateless_operand_load_pkt_len_f3;
            stateless_operand_load_hdr_len_f3;
            stateless_operand_load_count_f3;
            stateless_operand_load_flow_duration_f3;
            NoAction; 
        }
        size = 33;
        default_action = NoAction();
    }
    

    table f11_op_table {
        key = { 
            ig_md.sid : exact; 
            ig_intr_md.ingress_port : exact; // fwd, bwd.
            hdr.tcp.ctrl : exact;
        }
        actions = { 
            stateful_action_sum_f11;
            stateful_action_init_f11;
            stateful_action_min_f11;
            stateful_action_max_f11;
            NoAction; 
        }
        size = 50;
        default_action = NoAction();
    }

    table f12_op_table {
        key = { 
            ig_md.sid : exact; 
            ig_intr_md.ingress_port : exact; // fwd, bwd.
            hdr.tcp.ctrl : exact;
        }
        actions = { 
            stateful_action_sum_f12;
            stateful_action_init_f12;
            stateful_action_min_f12;
            stateful_action_max_f12;
            NoAction; 
        }
        size = 50;
        default_action = NoAction();
    }
    
    table f13_op_table {
        key = { 
            ig_md.sid : exact; 
            ig_intr_md.ingress_port : exact; // fwd, bwd.
            hdr.tcp.ctrl : exact;
        }
        actions = { 
            stateful_action_sum_f13;
            stateful_action_init_f13;
            stateful_action_min_f13;
            stateful_action_max_f13;
            NoAction; 
        }
        size = 50;
        default_action = NoAction();
    }

    table f21_op_table {
        key = { 
            ig_md.sid : exact; 
            ig_intr_md.ingress_port : exact;
            hdr.tcp.ctrl : exact;
        }
        actions = { 
            stateful_action_sum_f21;
            stateful_action_init_f21;
            stateful_action_min_f21;
            stateful_action_max_f21;
            NoAction; 
        }
        size = 50;
        default_action = NoAction();
    }

    table f22_op_table {
        key = { 
            ig_md.sid : exact; 
            ig_intr_md.ingress_port : exact;
            hdr.tcp.ctrl : exact;
        }
        actions = { 
            stateful_action_sum_f22;
            stateful_action_init_f22;
            stateful_action_min_f22;
            stateful_action_max_f22;
            NoAction; 
        }
        size = 50;
        default_action = NoAction();
    }

    table f23_op_table {
        key = { 
            ig_md.sid : exact; 
            ig_intr_md.ingress_port : exact;
            hdr.tcp.ctrl : exact;
        }
        actions = { 
            stateful_action_sum_f23;
            stateful_action_init_f23;
            stateful_action_min_f23;
            stateful_action_max_f23;
            NoAction; 
        }
        size = 50;
        default_action = NoAction();
    }
    
    table f1_table {
        key = {
            ig_md.sid: exact;
            ig_md.f1: ternary;
        }
        actions = {
            f1_encode;
            NoAction;
        }
        size = 2560;
        default_action = NoAction();
    }

    table f2_table {
        key = {
            ig_md.sid: exact;
            ig_md.f2: ternary;
        }
        actions = {
            f2_encode;
            NoAction;
        }
        size = 2560;
        default_action = NoAction();
    }

    table classifier {
        key = { 
            ig_md.sid : exact; 
            ig_md.f1_encoded : ternary; 
            ig_md.f2_encoded : ternary; 
            
        }
        actions = { 
            set_next_sid; 
            NoAction; 
        }
        size = 1024;
        default_action = NoAction();
    }

    action set_reg_index_1() {
        ig_md.state_index = 1;
    }

    action set_reg_index_2() {
        ig_md.state_index = 2;
    }

    action set_reg_index_3() {
        ig_md.state_index = 3;
    }

    table reg_index_table {
        key = {
            ig_md.flow_hash : range;
        }
        actions = {
            set_reg_index_1;
            set_reg_index_2;
            set_reg_index_3;
            NoAction;
        }
        size = 3;
        default_action = NoAction();
    }

    apply {
         if (ig_intr_md.ingress_port== RECIRCULATE_PORT) {
            ig_md.src_addr = hdr.resubmit_hdr.srcAddr;
            ig_md.dst_addr = hdr.resubmit_hdr.dstAddr;
            ig_md.src_port = hdr.resubmit_hdr.src_port;
            ig_md.dst_port = hdr.resubmit_hdr.dst_port;
            ig_md.protocol = hdr.resubmit_hdr.protocol;
            // take hash of the flow-id
            hash_packet(ig_md.src_addr, 
                        ig_md.dst_addr, 
                        ig_md.src_port,
                        ig_md.dst_port, 
                        ig_md.protocol);
            ig_md.sid=hdr.resubmit_hdr.sid;

            sid_register.write(ig_md.flow_hash, ig_md.sid);
            pkt_count_register.write(ig_md.flow_hash,0);
            
            if (ig_md.state_index == 1){
                f11_op_table.apply();
                f21_op_table.apply();
                // f31_op_table.apply();
            } 
            else if (ig_md.state_index == 2){
                f12_op_table.apply();
                f22_op_table.apply();
                // f32_op_table.apply();
            } 
            else if (ig_md.state_index == 3){
                f13_op_table.apply();
                f23_op_table.apply();
                // f33_op_table.apply();
            }
                
        } 
        else {
            ig_md.src_addr = hdr.ipv4.srcAddr;
            ig_md.dst_addr = hdr.ipv4.dstAddr;
            ig_md.src_port = hdr.tcp.srcPort;
            ig_md.dst_port = hdr.tcp.dstPort;
            ig_md.protocol = hdr.ipv4.protocol;
            if (hdr.ipv4.isValid()) {
                // take hash of the flow-id
                hash_packet(ig_md.src_addr, 
                            ig_md.dst_addr, 
                            ig_md.src_port, 
                            ig_md.dst_port, 
                            ig_md.protocol);
                reg_index_table.apply();
            }
            ig_md.sid = sid_register.read(ig_md.flow_hash);
            ig_md.packet_counter = pkt_count_register.read(ig_md.flow_hash);
            
            // TODO: Fix
            f1_op_load_table.apply();
            f2_op_load_table.apply();
            // f3_op_load_table.apply();
            
            if (ig_md.state_index == 1){
                f11_op_table.apply();
                f21_op_table.apply();
                // f31_op_table.apply();
            } 
            else if (ig_md.state_index == 2){
                f12_op_table.apply();
                f22_op_table.apply();
                // f32_op_table.apply();
            } 
            else if (ig_md.state_index == 3) {
                f13_op_table.apply();
                f23_op_table.apply();
                // f33_op_table.apply();
            }

        }

        if(ig_md.packet_counter==FLOW_SIZE){
            f1_table.apply();
            f2_table.apply();
            classifier.apply();
        }
    

        if (hdr.tcp.isValid() && (hdr.tcp.ctrl & 0x01) == 0) {
            if (ig_md.packet_counter==FLOW_SIZE && ig_intr_md.ingress_port != RECIRCULATE_PORT) {
                recirculate_packet();
            }
        } 

        if (hdr.tcp.isValid() && (hdr.tcp.ctrl & 0x01) != 0) {
            emit_digest();
        }
        
    }
}

control IngressDeparser(packet_out pkt,
                        /* User */
                        inout headers_t hdr,
                        in metadata_t ig_md,
                        /* Intrinsic */
                        in    ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md) 
{
    Digest<digest_a_t>() digest_a;

    apply {
        
        if (ig_dprsr_md.digest_type == L2_LEARN_DIGEST) {
            hdr.digest_a.setValid();
            hdr.digest_a.srcAddr = ig_md.src_addr;
            hdr.digest_a.dstAddr = ig_md.dst_addr;
            hdr.digest_a.src_port = ig_md.src_port;
            hdr.digest_a.dst_port = ig_md.dst_port;
            hdr.digest_a.protocol = hdr.ipv4.protocol;
            digest_a.pack(hdr.digest_a);
        }
        pkt.emit(hdr.resubmit_hdr);

    }
}