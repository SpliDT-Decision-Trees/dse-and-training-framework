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

#ifndef __HEADERS_P4__
#define __HEADERS_P4__

/*************************************************************************
************************* CONSTANTS **************************************
*************************************************************************/

#define FLOW_SIZE 5
#define KEY_SIZE 16
#define RECIRCULATE_PORT 68
#define FEATURE_WIDTH_32 32
#define FEATURE_WIDTH_16 16
#define FEATURE_WIDTH_8 8
#define NUM_FLOWS 143300  // 2^17
#define NUM_FLOWS_LARGE 429900  // 2^20

const bit<3> L2_LEARN_DIGEST = 2;
const bit<16> TYPE_IPV4 = 0x0800;

/*************************************************************************
************************* TYPEDEFS ***************************************
*************************************************************************/

typedef bit<48> macAddr_t;
typedef bit<32> ip4Addr_t;

/*************************************************************************
************************* HEADERS ****************************************
*************************************************************************/

header ethernet_t {
    macAddr_t dstAddr;
    macAddr_t srcAddr;
    bit<16> etherType;
}

header ipv4_t {
    bit<4> version;
    bit<4> ihl;
    bit<8> diffserv;
    bit<16> totalLen;
    bit<16> identification;
    bit<3> flags;
    bit<13> fragOffset;
    bit<8> ttl;
    bit<8> protocol;
    bit<16> hdrChecksum;
    ip4Addr_t srcAddr;
    ip4Addr_t dstAddr;
}

header tcp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<32> seqNo;
    bit<32> ackNo;
    bit<4> dataOffset;
    bit<3> res;
    bit<3> ecn;
    bit<6> ctrl;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgentPtr;
}

header udp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<16> length_;
    bit<16> checksum;
}

header resubmit_hdr_t {
    ip4Addr_t srcAddr;
    ip4Addr_t dstAddr;
    bit<16> src_port;
    bit<16> dst_port;
    bit<8> protocol;
    bit<8> sid;
}


header features_t {
    bit<32> iat;
    bit<32> pkt_len;
    bit<32> hdr_len;
    bit<32> flow_duration;
    bit<32> dst_port;
    // bit<8> type;  100 for first packet, 200 for resubmit
}

header digest_a_t {
    ip4Addr_t srcAddr;
    ip4Addr_t dstAddr;
    bit<16> src_port;
    bit<16> dst_port;
    bit<8> protocol;
    bit<8> sid;
}


/*************************************************************************
************************* STRUCTS ****************************************
*************************************************************************/
// headers struct bundles all defined headers
struct headers_t {
    ethernet_t     ethernet;
    ipv4_t         ipv4;
    tcp_t          tcp;
    udp_t          udp;
    features_t     features;
    resubmit_hdr_t resubmit_hdr;
    digest_a_t     digest_a;
}

// pipeline user metadata
struct metadata_t {
    bit<2> state_index;
    
    bit<32> f1;
    bit<32> f2;
    bit<32> f3;
    
    bit<32> f1_encoded;
    bit<32> f2_encoded;
    
    bit<8>  sid;
    // bit<2> recirculate_packet;
    bit<8>  packet_counter;
    bit<8>  next_sid_value;
    bit<KEY_SIZE> flow_hash;

    // for hash and digest
    ip4Addr_t src_addr;
    ip4Addr_t dst_addr;
    bit<16>   src_port;
    bit<16>   dst_port;
    bit<8> protocol;
}

#endif // __HEADERS_P4__