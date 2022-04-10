import copy
import os
from re import I
import sys
import csv
import glob
import h5py
import time
import pyshark
import socket
import pickle
import random
import hashlib
import argparse
import ipaddress
import numpy as np
from lxml import etree
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle as sklearn_shuffle
from multiprocessing import Process, Manager, Value, Queue
from util_functions import *

# Sample commands
# split a pcap file into smaller chunks to leverage multi-core CPUs: tcpdump -r dataset.pcap -w dataset-chunk -C 1000
# dataset parsing (first step): python3 lucid_dataset_parser.py --dataset_type SYN2020 --dataset_folder ./sample-dataset/ --packets_per_flow 10 --dataset_id SYN2020 --traffic_type all --time_window 10
# dataset parsing (second step): python3 lucid_dataset_parser.py --preprocess_folder ./sample-dataset/

IDS2018_DDOS_FLOWS = {'attackers': ['18.218.115.60', '18.219.9.1','18.219.32.43','18.218.55.126','52.14.136.135','18.219.5.43','18.216.200.189','18.218.229.235','18.218.11.51','18.216.24.42'],
                      'victims': ['18.218.83.150','172.31.69.28']}

IDS2017_DDOS_FLOWS = {'attackers': ['172.16.0.1'],
                      'victims': ['192.168.10.50']}

CUSTOM_DDOS_SYN = {'attackers': ['11.0.0.' + str(x) for x in range(1,255)],
                      'victims': ['10.42.0.2']}

DDOS_ATTACK_SPECS = {
    'IDS2017' : IDS2017_DDOS_FLOWS,
    'IDS2018' : IDS2018_DDOS_FLOWS,
    'SYN2020' : CUSTOM_DDOS_SYN
}


vector_proto = CountVectorizer()
vector_proto.fit_transform(protocols).todense()

random.seed(SEED)
np.random.seed(SEED)



def get_pkt_direction(srcIP,dstIP):
    internalIP = "192.168"
    if internalIP in srcIP and internalIP in dstIP:
        return 0
    elif internalIP in srcIP:
        return 1
    elif internalIP in dstIP:
        return 2
    else:
        print ("No private address in this flow!!!!")
        return 3

class packet_features:
    def __init__(self):
        self.id_fwd = (0,0,0,0,0) # 5-tuple src_ip_addr, src_port,,dst_ip_addr,dst_port,protocol
        self.id_bwd = (0,0,0,0,0)  # 5-tuple src_ip_addr, src_port,,dst_ip_addr,dst_port,protocol
        self.features_list = []


    def __str__(self):
        return "{} -> {}".format(self.id_fwd, self.features_list)

def get_ddos_flows(attackers,victims):
    DDOS_FLOWS = {}

    if '/' in attackers: # subnet
        DDOS_FLOWS['attackers'] = [str(ip) for ip in list(ipaddress.IPv4Network(attackers).hosts())]
    else: # single address
        DDOS_FLOWS['attackers'] = [str(ipaddress.IPv4Address(attackers))]

    if '/' in victims:  # subnet
        DDOS_FLOWS['victims'] = [str(ip) for ip in list(ipaddress.IPv4Network(victims).hosts())]
    else:  # single address
        DDOS_FLOWS['victims'] = [str(ipaddress.IPv4Address(victims))]

    return DDOS_FLOWS

# function that build the labels based on the dataset type
def parse_labels(dataset_type=None, attackers=None,victims=None):
    output_dict = {}

    if attackers is not None and victims is not None:
        DDOS_FLOWS = get_ddos_flows(attackers, victims)
    elif dataset_type is not None and dataset_type in DDOS_ATTACK_SPECS:
        DDOS_FLOWS = DDOS_ATTACK_SPECS[dataset_type]
    else:
        return None

    for attacker in DDOS_FLOWS['attackers']:
        for victim in DDOS_FLOWS['victims']:
            ip_src = str(attacker)
            ip_dst = str(victim)
            key_fwd = (ip_src, ip_dst)
            key_bwd = (ip_dst, ip_src)

            if key_fwd not in output_dict:
                output_dict[key_fwd] = 1
            if key_bwd not in output_dict:
                output_dict[key_bwd] = 1

    return output_dict

def parse_packet(pkt):
    pf = packet_features()
    tmp_id = [0,0,0,0,0]

    try:
        pf.features_list.append(float(pkt.sniff_timestamp))  # timestampchild.find('Tag').text
        pf.features_list.append(int(pkt.ip.len))  # packet length
        pf.features_list.append(int(hashlib.sha256(str(pkt.highest_layer).encode('utf-8')).hexdigest(),
                                    16) % 10 ** 8)  # highest layer in the packet
        pf.features_list.append(int(int(pkt.ip.flags, 16)))  # IP flags
        tmp_id[0] = str(pkt.ip.src)  # int(ipaddress.IPv4Address(pkt.ip.src))
        tmp_id[2] = str(pkt.ip.dst)  # int(ipaddress.IPv4Address(pkt.ip.dst))

        protocols = vector_proto.transform([pkt.frame_info.protocols]).toarray().tolist()[0]
        protocols = [1 if i >= 1 else 0 for i in
                     protocols]  # we do not want the protocols counted more than once (sometimes they are listed twice in pkt.frame_info.protocols)
        protocols_value = int(np.dot(np.array(protocols), powers_of_two))
        pf.features_list.append(protocols_value)

        protocol = int(pkt.ip.proto)
        tmp_id[4] = protocol
        if pkt.transport_layer != None:
            if protocol == socket.IPPROTO_TCP:
                tmp_id[1] = int(pkt.tcp.srcport)
                tmp_id[3] = int(pkt.tcp.dstport)
                pf.features_list.append(int(pkt.tcp.len))  # TCP length
                pf.features_list.append(int(pkt.tcp.ack))  # TCP ack
                pf.features_list.append(int(pkt.tcp.flags, 16))  # TCP flags
                pf.features_list.append(int(pkt.tcp.window_size_value))  # TCP window size
                pf.features_list = pf.features_list + [0, 0]  # UDP + ICMP positions
            elif protocol == socket.IPPROTO_UDP:
                pf.features_list = pf.features_list + [0, 0, 0, 0]  # TCP positions
                tmp_id[1] = int(pkt.udp.srcport)
                pf.features_list.append(int(pkt.udp.length))  # UDP length
                tmp_id[3] = int(pkt.udp.dstport)
                pf.features_list = pf.features_list + [0]  # ICMP position
        elif protocol == socket.IPPROTO_ICMP:
            pf.features_list = pf.features_list + [0, 0, 0, 0, 0]  # TCP and UDP positions
            pf.features_list.append(int(pkt.icmp.type))  # ICMP type
        else:
            pf.features_list = pf.features_list + [0, 0, 0, 0, 0, 0]  # padding for layer3-only packets
            tmp_id[4] = 0

        pf.id_fwd = (tmp_id[0], tmp_id[1], tmp_id[2], tmp_id[3], tmp_id[4])
        pf.id_bwd = (tmp_id[2], tmp_id[3], tmp_id[0], tmp_id[1], tmp_id[4])

        return pf

    except AttributeError as e:
        # ignore packets that aren't TCP/UDP or IPv4
        return None

# Offline preprocessing of pcap files for model training, validation and testing
def process_pcap(pcap_file,dataset_type,in_labels,max_flow_len,labelled_flows,traffic_type='all',time_window=TIME_WINDOW):
    start_time = time.time()
    temp_dict = OrderedDict()
    start_time_window = -1

    pcap_name = pcap_file.split("/")[-1]
    print("Processing file: ", pcap_name)

    cap = pyshark.FileCapture(pcap_file)
    for i, pkt in enumerate(cap):
        if i % 1000 == 0:
            print(pcap_name + " packet #", i)

        # start_time_window is used to group packets/flows captured in a time-window
        if start_time_window == -1 or float(pkt.sniff_timestamp) > start_time_window + time_window:
            start_time_window = float(pkt.sniff_timestamp)

        pf = parse_packet(pkt)
        temp_dict = store_packet(pf, temp_dict, start_time_window, max_flow_len)

    apply_labels(temp_dict, labelled_flows, in_labels, traffic_type)
    print('Completed file {} in {} seconds.'.format(pcap_name, time.time() - start_time))

# Transforms live traffic into input samples for inference
def process_live_traffic(cap, dataset_type, in_labels, max_flow_len, traffic_type='all',time_window=TIME_WINDOW):
    start_time = time.time()
    temp_dict = OrderedDict()
    labelled_flows = []

    start_time_window = start_time
    time_window = start_time_window + time_window

    if isinstance(cap, pyshark.LiveCapture) == True:
        for pkt in cap.sniff_continuously():
            if time.time() >= time_window:
                break
            pf = parse_packet(pkt)
            temp_dict = store_packet(pf, temp_dict, start_time_window, max_flow_len)
    elif isinstance(cap, pyshark.FileCapture) == True:
        while time.time() < time_window:
            pkt = cap.next()
            pf = parse_packet(pkt)
            temp_dict = store_packet(pf,temp_dict,start_time_window,max_flow_len)

    apply_labels(temp_dict,labelled_flows, in_labels,traffic_type)
    return labelled_flows

def store_packet(pf,temp_dict,start_time_window, max_flow_len):
    if pf is not None:
        if pf.id_fwd in temp_dict and start_time_window in temp_dict[pf.id_fwd] and \
                temp_dict[pf.id_fwd][start_time_window].shape[0] < max_flow_len:
            temp_dict[pf.id_fwd][start_time_window] = np.vstack(
                [temp_dict[pf.id_fwd][start_time_window], pf.features_list])
        elif pf.id_bwd in temp_dict and start_time_window in temp_dict[pf.id_bwd] and \
                temp_dict[pf.id_bwd][start_time_window].shape[0] < max_flow_len:
            temp_dict[pf.id_bwd][start_time_window] = np.vstack(
                [temp_dict[pf.id_bwd][start_time_window], pf.features_list])
        else:
            if pf.id_fwd not in temp_dict and pf.id_bwd not in temp_dict:
                temp_dict[pf.id_fwd] = {start_time_window: np.array([pf.features_list]), 'label': 0}
            elif pf.id_fwd in temp_dict and start_time_window not in temp_dict[pf.id_fwd]:
                temp_dict[pf.id_fwd][start_time_window] = np.array([pf.features_list])
            elif pf.id_bwd in temp_dict and start_time_window not in temp_dict[pf.id_bwd]:
                temp_dict[pf.id_bwd][start_time_window] = np.array([pf.features_list])
    return temp_dict

def apply_labels(flows, labelled_flows, labels, traffic_type):
    for five_tuple, flow in flows.items():
        if labels is not None:
            short_key = (five_tuple[0], five_tuple[2])  # for IDS2017/IDS2018 dataset the labels have shorter keys
            flow['label'] = labels.get(short_key, 0)

        for flow_key, packet_list in flow.items():
            # relative time wrt the time of the first packet in the flow
            if flow_key != 'label':
                amin = np.amin(packet_list,axis=0)[0]
                packet_list[:, 0] = packet_list[:, 0] - amin

        if traffic_type == 'ddos' and flow['label'] == 0: # we only want malicious flows from this dataset
            continue
        elif traffic_type == 'benign' and flow['label'] > 0: # we only want benign flows from this dataset
            continue
        else:
            labelled_flows.append((five_tuple,flow))

# returns the total number of flows
def count_flows(preprocessed_flows):
    ddos_flows = 0
    total_flows = len(preprocessed_flows)
    ddos_fragments = 0
    total_fragments = 0
    for flow in preprocessed_flows:
        flow_fragments = len(flow[1]) - 1
        total_fragments += flow_fragments
        if flow[1]['label'] > 0:
            ddos_flows += 1
            ddos_fragments += flow_fragments  # the label does not count

    return (total_flows, ddos_flows, total_flows - ddos_flows), (total_fragments, ddos_fragments, total_fragments-ddos_fragments)

# balance the dataset based on the number of benign and malicious fragments of flows
def balance_dataset(flows,total_fragments=float('inf')):
    new_flow_list = []

    _,(_, ddos_fragments, benign_fragments) = count_flows(flows)

    if ddos_fragments == 0 or benign_fragments == 0:
        min_fragments = total_fragments
    else:
        min_fragments = min(total_fragments/2,ddos_fragments,benign_fragments)

    random.shuffle(flows)
    new_benign_fragments = 0
    new_ddos_fragments = 0

    for flow in flows:
        if flow[1]['label'] == 0 and (new_benign_fragments < min_fragments ):
            new_benign_fragments += len(flow[1]) - 1
            new_flow_list.append(flow)
        elif flow[1]['label'] == 1 and (new_ddos_fragments < min_fragments):
            new_ddos_fragments += len(flow[1]) - 1
            new_flow_list.append(flow)

    return new_flow_list, new_benign_fragments, new_ddos_fragments

# convert the dataset from dictionaries with 5-tuples keys into a list of flow fragments and another list of labels
def dataset_to_list_of_fragments(dataset):
    keys = []
    X = []
    y = []

    for flow in dataset:
        tuple = flow[0]
        flow_data = flow[1]
        label = flow_data['label']
        for key, fragment in flow_data.items():
            if key != 'label':
                X.append(fragment)
                y.append(label)
                keys.append(tuple)

    return X,y,keys

def train_test_split(flow_list,train_size=TRAIN_SIZE, shuffle=True):
    test_list = []
    _,(total_examples,_,_) = count_flows(flow_list)
    test_examples = total_examples - total_examples*train_size

    if shuffle == True:
        random.shuffle(flow_list)

    current_test_examples = 0
    while current_test_examples < test_examples:
        flow = flow_list.pop(0)
        test_list.append(flow)
        current_test_examples += len(flow[1])-1


    return flow_list,test_list

def main(argv):
    command_options = " ".join(str(x) for x in argv[1:])


# TODO Wywalić argumenty

    help_string = 'Usage[0]: python3 lucid_dataset_parser.py --dataset_type <dataset_name> --dataset_folder <folder path> --dataset_id <dataset identifier> --packets_per_flow <n> --time_window <t>\n' \
                  'Usage[1]: python3 lucid_dataset_parser.py --preprocess_folder <folder path>'
    manager = Manager()

    parser = argparse.ArgumentParser(
        description='Dataset parser',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', '--dataset_folder', nargs='+', type=str,
                        help='Folder with the dataset')
    parser.add_argument('-o', '--output_folder', nargs='+', type=str,
                        help='Output folder')
    parser.add_argument('-f', '--traffic_type', default='all', nargs='+', type=str,
                        help='Type of flow to process (all, benign, ddos)')
    parser.add_argument('-p', '--preprocess_folder', nargs='+', type=str,
                        help='Folder with preprocessed data')
    parser.add_argument('--preprocess_file', nargs='+', type=str,
                        help='File with preprocessed data')
    parser.add_argument('-b', '--balance_folder', nargs='+', type=str,
                        help='Folder where balancing datasets')
    parser.add_argument('-n', '--packets_per_flow', nargs='+', type=str,
                        help='Packet per flow sample')
    parser.add_argument('-s', '--samples', default=float('inf'), type=int,
                        help='Number of training samples in the reduced output')
    parser.add_argument('-i', '--dataset_id', nargs='+', type=str,
                        help='String to append to the names of output files')

    parser.add_argument('-t', '--dataset_type', nargs='+', type=str,
                        help='Type of the dataset. Available options are: IDS2017, IDS2018, SYN2020')

    parser.add_argument('-w', '--time_window', nargs='+', type=str,
                        help='Length of the time window')

    parser.add_argument('--no_split', help='Do not split the dataset', action='store_true')

    args = parser.parse_args()

    if args.packets_per_flow is not None:
        max_flow_len = int(args.packets_per_flow[0])
    else:
        max_flow_len = MAX_FLOW_LEN

    if args.time_window is not None:
        time_window = float(args.time_window[0])
    else:
        time_window = TIME_WINDOW

    if args.dataset_id is not None:
        dataset_id = str(args.dataset_id[0])
    else:
        dataset_id = ''

    if args.traffic_type is not None:
        traffic_type = str(args.traffic_type[0])
    else:
        traffic_type = 'all'

    if args.dataset_folder is not None and args.dataset_type is not None:
        process_list = []
        flows_list = []

        if args.output_folder is not None and os.path.isdir(args.output_folder[0]) is True:
            output_folder = args.output_folder[0]
        else:
            output_folder = args.dataset_folder[0]

        filelist = glob.glob(args.dataset_folder[0]+ '/*.pcap')
        in_labels = parse_labels(args.dataset_type[0],args.dataset_folder[0])

        start_time = time.time()
        for file in filelist:
            try:
                flows = manager.list()
                p = Process(target=process_pcap,args=(file,args.dataset_type[0],in_labels,max_flow_len,flows,traffic_type,time_window))
                process_list.append(p)
                flows_list.append(flows)
            except FileNotFoundError as e:
                continue

        for p in process_list:
            p.start()

        for p in process_list:
            p.join()

        np.seterr(divide='ignore', invalid='ignore')

        try:
            preprocessed_flows = list(flows_list[0])
        except:
            print ("ERROR: No traffic flows. \nPlease check that the dataset folder name (" + args.dataset_folder[0] + ") is correct and \nthe folder contains the traffic traces in pcap format (the pcap extension is mandatory)")
            exit(1)

        #concatenation of the features
        for results in flows_list[1:]:
            preprocessed_flows = preprocessed_flows + list(results)

        #TODO: TOO KNN    
        #5-tuple src_ip_addr, src_port,,dst_ip_addr,dst_port,protocol
        #print(preprocessed_flows)
        list_columns = ["src_ip_addr", "src_port", "dst_ip_addr", "dst_port", "protocol", "label", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11"]
        list_rows = []

        for five_tuple, time_window_tmp in preprocessed_flows:
            src_ip_addr,src_port,dst_ip_addr,dst_port,protocol = five_tuple
            label = time_window_tmp['label']
            t = [(k,v) for k, v in time_window_tmp.items()]
            x = [i[1] for i in t]

            try:
                for y in x:
                    ii  = y.tolist()
                    for el in ii:
                        list_rows.append((src_ip_addr,src_port,dst_ip_addr,dst_port,protocol, label) + tuple(el))
            except:
                pass

        with open('sample-dataset/knn_tuples.csv', 'w') as csvfile:
            write = csv.writer(csvfile)
            write.writerow(list_columns)
            write.writerows(list_rows)
    


        process_time = time.time()-start_time

        if dataset_id == '':
            dataset_id = str(args.dataset_type[0])

        filename = str(int(time_window)) + 't-' + str(max_flow_len) + 'n-' + dataset_id + '-preprocess'
        output_file = output_folder + '/' + filename
        output_file = output_file.replace("//", "/") # remove double slashes when needed


        (total_flows, ddos_flows, benign_flows),  (total_fragments, ddos_fragments, benign_fragments) = count_flows(preprocessed_flows)

        log_string = time.strftime("%Y-%m-%d %H:%M:%S") + " | dataset_type:" + args.dataset_type[0] + \
                     " | flows (tot,ben,ddos):(" + str(total_flows) + "," + str(benign_flows) + "," + str(ddos_flows) + \
                     ") | fragments (tot,ben,ddos):(" + str(total_fragments) + "," + str(benign_fragments) + "," + str(ddos_fragments) + \
                     ") | options:" + command_options + " | process_time:" + str(process_time) + " |\n"
        print (log_string)

        # saving log file
        with open(output_folder + '/history.log', "a") as myfile:
            myfile.write(log_string)

    if args.preprocess_folder is not None or args.preprocess_file is not None:
        if args.preprocess_folder is not None:
            output_folder = args.output_folder[0] if args.output_folder is not None else args.preprocess_folder[0]
            filelist = glob.glob(args.preprocess_folder[0] + '/*.data')
        else:
            output_folder = args.output_folder[0] if args.output_folder is not None else os.path.dirname(os.path.realpath(args.preprocess_file[0]))
            filelist = args.preprocess_file

        # obtain time_window and flow_len from filename and ensure that all files have the same values
        time_window = None
        max_flow_len = None
        dataset_id = None
        for file in filelist:
            filename = file.split('/')[-1].strip()
            current_time_window = int(filename.split('-')[0].strip().replace('t',''))
            current_max_flow_len = int(filename.split('-')[1].strip().replace('n',''))
            current_dataset_id = str(filename.split('-')[2].strip())
            if time_window != None and current_time_window != time_window:
                print ("Incosistent time windows!!")
                exit()
            else:
                time_window = current_time_window
            if max_flow_len != None and current_max_flow_len != max_flow_len:
                print ("Incosistent flow lengths!!")
                exit()
            else:
                max_flow_len = current_max_flow_len

            if dataset_id != None and current_dataset_id != dataset_id:
                dataset_id = "IDS201X"
            else:
                dataset_id = current_dataset_id
                

        # balance samples and redux the number of samples when requested
        preprocessed_flows, benign_fragments, ddos_fragments = balance_dataset(preprocessed_flows,args.samples)

        if len(preprocessed_flows) == 0:
            print("Empty dataset!")
            exit()

        preprocessed_train, preprocessed_test = train_test_split(preprocessed_flows,train_size=TRAIN_SIZE, shuffle=True)
        preprocessed_train, preprocessed_val = train_test_split(preprocessed_train, train_size=TRAIN_SIZE, shuffle=True)

        X_train, y_train, _ = dataset_to_list_of_fragments(preprocessed_train)
        X_val, y_val, _ = dataset_to_list_of_fragments(preprocessed_val)
        X_test, y_test, _ = dataset_to_list_of_fragments(preprocessed_test)

        # normalization and padding
        X_full = X_train + X_val + X_test
        y_full = y_train + y_val + y_test
        mins,maxs = static_min_max(time_window=time_window)

        total_examples = len(y_full)
        total_ddos_examples = sum(y_full)
        total_benign_examples = len(y_full) - sum(y_full)

        output_file = output_folder + '/' + str(time_window) + 't-' + str(max_flow_len) + 'n-' + dataset_id + '-dataset'
        if args.no_split == True: # don't split the dataset
            norm_X_full = normalize_and_padding(X_full, mins, maxs, max_flow_len)
            #norm_X_full = padding(X_full,max_flow_len) # only padding
            norm_X_full_np = np.array(norm_X_full)
            y_full_np = np.array(y_full)

            hf = h5py.File(output_file + '-full.hdf5', 'w')
            hf.create_dataset('set_x', data=norm_X_full_np)
            hf.create_dataset('set_y', data=y_full_np)
            hf.close()

            [full_packets] = count_packets_in_dataset([norm_X_full_np])
            log_string = time.strftime("%Y-%m-%d %H:%M:%S") + " | Total examples (tot,ben,ddos):(" + str(total_examples) + "," + str(total_benign_examples) + "," + str(total_ddos_examples) + \
                         ") | Total packets:(" + str(full_packets) + \
                         ") | options:" + command_options + " |\n"
        else:
            norm_X_train = normalize_and_padding(X_train,mins,maxs,max_flow_len)
            norm_X_val = normalize_and_padding(X_val, mins, maxs, max_flow_len)
            norm_X_test = normalize_and_padding(X_test, mins, maxs, max_flow_len)

            norm_X_train_np = np.array(norm_X_train)
            y_train_np = np.array(y_train)
            norm_X_val_np = np.array(norm_X_val)
            y_val_np = np.array(y_val)
            norm_X_test_np = np.array(norm_X_test)
            y_test_np = np.array(y_test)

            hf = h5py.File(output_file + '-train.hdf5', 'w')
            hf.create_dataset('set_x', data=norm_X_train_np)
            hf.create_dataset('set_y', data=y_train_np)
            hf.close()

            hf = h5py.File(output_file + '-val.hdf5', 'w')
            hf.create_dataset('set_x', data=norm_X_val_np)
            hf.create_dataset('set_y', data=y_val_np)
            hf.close()

            hf = h5py.File(output_file + '-test.hdf5', 'w')
            hf.create_dataset('set_x', data=norm_X_test_np)
            hf.create_dataset('set_y', data=y_test_np)
            hf.close()

            [train_packets, val_packets, test_packets] = count_packets_in_dataset([norm_X_train_np, norm_X_val_np, norm_X_test_np])
            log_string = time.strftime("%Y-%m-%d %H:%M:%S") + " | examples (tot,ben,ddos):(" + str(total_examples) + "," + str(total_benign_examples) + "," + str(total_ddos_examples) + \
                         ") | Train/Val/Test sizes: (" + str(norm_X_train_np.shape[0]) + "," + str(norm_X_val_np.shape[0]) + "," + str(norm_X_test_np.shape[0]) + \
                         ") | Packets (train,val,test):(" + str(train_packets) + "," + str(val_packets) + "," + str(test_packets) + \
                         ") | options:" + command_options + " |\n"

        print(log_string)

        # saving log file
        with open(output_folder + '/history.log', "a") as myfile:
            myfile.write(log_string)

    if args.balance_folder is not None and args.output_folder is not None:
        output_folder = args.output_folder[0] if args.output_folder is not None else args.balance_folder[0]
        datasets = []
        for folder in args.balance_folder:
            datasets += glob.glob(folder + '/*.hdf5')
        train_filelist = {}
        val_filelist = {}
        test_filelist = {}
        min_samples_train = float('inf')
        min_samples_val = float('inf')
        min_samples_test = float('inf')

        output_filename_prefix = None

        for file in datasets:
            filename = file.split('/')[-1].strip()
            dataset = h5py.File(file, "r")
            X = np.array(dataset["set_x"][:])  # features
            Y = np.array(dataset["set_y"][:])  # labels
            if 'train' in filename:
                key = filename.split('dataset')[0].strip() + 'dataset-balanced-train.hdf5'
                if output_filename_prefix ==None:
                    output_filename_prefix = filename.split('IDS')[0].strip()
                else:
                    if filename.split('IDS')[0].strip() != output_filename_prefix:
                        print ("Inconsistent datasets!")
                        exit()
                train_filelist[key] = (X,Y)
                if X.shape[0] < min_samples_train:
                    min_samples_train = X.shape[0]
            elif 'val' in filename:
                key = filename.split('dataset')[0].strip() + 'dataset-balanced-val.hdf5'
                if output_filename_prefix ==None:
                    output_filename_prefix = filename.split('IDS')[0].strip()
                else:
                    if filename.split('IDS')[0].strip() != output_filename_prefix:
                        print ("Inconsistent datasets!")
                        exit()
                val_filelist[key] = (X,Y)
                if X.shape[0] < min_samples_val:
                    min_samples_val = X.shape[0]
            elif 'test' in filename:
                key = filename.split('dataset')[0].strip() + 'dataset-balanced-test.hdf5'
                if output_filename_prefix ==None:
                    output_filename_prefix = filename.split('IDS')[0].strip()
                else:
                    if filename.split('IDS')[0].strip() != output_filename_prefix:
                        print ("Inconsistent datasets!")
                        exit()
                test_filelist[key] = (X, Y)
                if X.shape[0] < min_samples_test:
                    min_samples_test = X.shape[0]

        final_X = {'train':None,'val':None,'test':None}
        final_y = {'train':None,'val':None,'test':None}

        for key,value in train_filelist.items():
            X_short = value[0][:min_samples_train,...]
            y_short = value[1][:min_samples_train,...]

            if final_X['train'] is None:
                final_X['train'] = X_short
                final_y['train'] = y_short
            else:
                final_X['train'] = np.vstack((final_X['train'],X_short))
                final_y['train'] = np.hstack((final_y['train'],y_short))

        for key,value in val_filelist.items():
            X_short = value[0][:min_samples_val,...]
            y_short = value[1][:min_samples_val,...]

            if final_X['val'] is None:
                final_X['val'] = X_short
                final_y['val'] = y_short
            else:
                final_X['val'] = np.vstack((final_X['val'],X_short))
                final_y['val'] = np.hstack((final_y['val'],y_short))


        for key,value in test_filelist.items():
            X_short = value[0][:min_samples_test,...]
            y_short = value[1][:min_samples_test,...]

            if final_X['test'] is None:
                final_X['test'] = X_short
                final_y['test'] = y_short
            else:
                final_X['test'] = np.vstack((final_X['test'],X_short))
                final_y['test'] = np.hstack((final_y['test'],y_short))

        for key,value in final_X.items():
            filename = output_filename_prefix + 'IDS201X-dataset-balanced-' + key + '.hdf5'
            hf = h5py.File(output_folder + '/' + filename, 'w')
            hf.create_dataset('set_x', data=value)
            hf.create_dataset('set_y', data=final_y[key])
            hf.close()

        total_flows = final_y['train'].shape[0]+final_y['val'].shape[0]+final_y['test'].shape[0]
        ddos_flows = sum(final_y['train'])+sum(final_y['val'])+sum(final_y['test'])
        benign_flows = total_flows-ddos_flows
        [train_packets, val_packets, test_packets] = count_packets_in_dataset([final_X['train'], final_X['val'], final_X['test']])
        log_string = time.strftime("%Y-%m-%d %H:%M:%S") + " | total_flows (tot,ben,ddos):(" + str(total_flows) + "," + str(benign_flows) + "," + str(ddos_flows) + \
                     ") | Packets (train,val,test):(" + str(train_packets) + "," + str(val_packets) + "," + str(test_packets) + \
                     ") | Train/Val/Test sizes: (" + str(final_y['train'].shape[0]) + "," + str(final_y['val'].shape[0]) + "," + str(final_y['test'].shape[0]) + \
                     ") | options:" + command_options + " |\n"

        print(log_string)

        # saving log file
        with open(output_folder + '/history.log', "a") as myfile:
            myfile.write(log_string)


    if args.dataset_folder is None and args.preprocess_folder is None and args.preprocess_file is None and args.balance_folder is None:
        print (help_string)
    if args.dataset_type is None and args.dataset_folder is not None:
        print("Please specify the dataset type (IDS2017, IDS2018, SYN2020)!")
        print(help_string)
    if args.output_folder is None and args.balance_folder is not None:
        print("Please specify the output folder!")
        print(help_string)

if __name__ == "__main__":
    main(sys.argv)
