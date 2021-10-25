# LUCID: A Practical, Lightweight Deep Learning Solution for DDoS Attack Detection

LUCID (Lightweight, Usable CNN in DDoS Detection) is a lightweight Deep Learning-based DDoS detection framework suitable for online resource-constrained environments, which leverages Convolutional Neural Networks (CNNs) to learn the behaviour of DDoS and benign traffic flows with both low processing overhead and attack detection time. LUCID includes a dataset-agnostic pre-processing mechanism that produces traffic observations consistent with those collected in existing online systems, where the detection algorithms must cope with segments of traffic flows collected over pre-defined time windows.

More details on the architecture of LUCID and its performance in terms of detection accuracy and execution time are available in the following research paper:

R. Doriguzzi-Corin, S. Millar, S. Scott-Hayward, J. Martínez-del-Rincón and D. Siracusa, "Lucid: A Practical, Lightweight Deep Learning Solution for DDoS Attack Detection," in *IEEE Transactions on Network and Service Management*, vol. 17, no. 2, pp. 876-889, June 2020, doi: 10.1109/TNSM.2020.2971776.



## Installation

The current LUCID's CNN is implemented in Python v3.8 with Keras and Tensorflow 2, while the traffic pre-processing tool is implemented in Python v3.8, Numpy and Pyshark. The original version, implemented with Tensorflow 1.13.1 and evaluated in the aforementioned paper, is available in branch *lucid-tensorflow-1.13.1*.

LUCID requires the installation of a number of Python tools and libraries. This can be done by using the ```conda``` software environment (https://docs.conda.io/projects/conda/en/latest/).
We suggest the installation of ```miniconda```, a light version of ```conda```. ```miniconda``` is available for MS Windows, MacOSX and Linux and can be installed by following the guidelines available at https://docs.conda.io/en/latest/miniconda.html#. 

In a Linux OS, execute the following command and follows the on-screen instructions:

```
bash Miniconda3-latest-Linux-x86_64.sh
```

Then create a new ```conda``` environment (called ```python38```) based on Python 3.8 and including part the required packages:

```
conda create -n python38 python=3.8 numpy tensorflow=2.3.0 h5py lxml
```

Activate the new ```python38``` environment:

```
conda activate python38
```

And finalise the installation with a few more packages:

```
(python38)$ pip3 install pyshark sklearn
```

Pyshark is just Python wrapper for tshark, allowing python packet parsing using wireshark dissectors. This means that ```tshark``` must be also installed. On an Ubuntu-based OS, use the following command:

```
sudo apt install tshark
```

Please note that the current LUCID code works with ```tshark``` **version 3.2.13 or lower**. Issues have been reported when using newest releases such as 3.4.X.

For the sake of simplicity, we omit the command prompt ```(python38)$``` in the following example commands in this README.   ```(python38)$``` indicates that we are working inside the ```python38``` execution environment, which provides all the required libraries and tools. If the command prompt is not visible, re-activate the environment as explained above.

## Traffic pre-processing

LUCID requires a labelled dataset, including the traffic traces in the format of ```pcap``` files. The traffic pre-processing functions are implemented in the ```lucid_dataset_parser.py``` Python script. It currently supports two DDoS datasets from the University of New Brunswick (UNB) (https://www.unb.ca/cic/datasets/index.html): CIC-IDS2017 and CSE-CIC-IDS2018, plus a custom dataset containing a SYN Flood DDoS attack (SYN2020) that will be used for this guide and included in the ```sample-dataset``` folder.

With term *support*, we mean the capability of the script to correctly label the packets and the traffic flows either as benign or DDoS. In general, this is done by parsing a file with the labels provided with the traffic traces, like in the case of the UNB datasets, or by manually indicating the IP address(es) of the attacker(s) and the IP address(es) of the victim(s) in the code. Of course, also in the latter case, the script must be tuned with the correct information of the traffic (all the attacker/victim pairs of IP addresses), as this information is very specific to the dataset and to the methodology used to generate the traffic. 

Said that, ```lucid_dataset_parser.py``` includes the structures with the pairs attacker/victim of the three datasets mentioned above (CIC-IDS2017, CSE-CIC-IDS2018 and SYN2020), but it can be easily extended to support other datasets by replicating the available code.

For instance, the following Python dictionary provides the IP addresses of the 254 attackers and the victim involved in the custom SYN Flood attack:   

```
CUSTOM_DDOS_SYN = {'attackers': ['11.0.0.' + str(x) for x in range(1,255)],
                      'victims': ['10.42.0.2']}
```

### Command options

The following parameters can be specified when using ```lucid_dataset_parser.py```:

- ```-d```, ```--dataset_folder```: Folder with the dataset
- ```-o```, ```--output_folder ```: Folder where  the scripts saves the output. The dataset folder is used when this option is not used
- ```-f```, ```--traffic_type ```: Type of flow to process (all, benign, ddos)
- ```-p```, ```--preprocess_folder ```: Folder containing the intermediate files ```*.data```
- ```-t```, ```--dataset_type ```: Type of the dataset. Available options are: IDS2017, IDS2018, SYN2020
- ```-n```, ```--packets_per_flow ```: Maximum number of packets in a sample
- ```-w```, ```--time_window ```: Length of the time window (in seconds)
- ```-i```, ```--dataset_id ```: String to append to the names of output files



### First step

The traffic pre-processing operation comprises two steps. The first parses the file with the labels (if needed) all extracts the features from the packets of all the ```pcap``` files contained in the source directory. The features are grouped in flows, where a flow is a set of features from packets with the same source IP, source UDP/TCP port, destination IP and destination UDP/TCP port and protocol. Flows are bi-directional, therefore, packet (srcIP,srcPort,dstIP,dstPort,proto) belongs to the same flow of (dstIP,dstPort,srcIP,srcPort,proto). The result is a set of intermediate binary files with extension ```.data```.

This first step can be executed with command:

```
python3 lucid_dataset_parser.py --dataset_type SYN2020 --dataset_folder ./sample-dataset/ --packets_per_flow 10 --dataset_id SYN2020 --traffic_type all --time_window 10
```

This will process in parallel the two files, producing a file named ```10t-10n-SYN2020-preprocess.data```. In general, the script loads all the ```pcap``` files contained in the folder indicated with option ```--dataset_folder``` and starting with prefix ```dataset-chunk-```. The files are processed in parallel to minimise the execution time.

Prefix ```10t-10n``` means that the pre-processing has been done using a time window of 10 seconds (10t) and a flow length of 10 packets (10n). Please note that ```SYN2020``` in the filename is the result of option ```--dataset_id SYN2020``` in the command.

Time window and flow length are two hyperparameters of LUCID. For more information, please refer to the research paper mentioned above. 

### Second step

The second step loads the ```*.data``` files, merges them into a single data structure stored in RAM memory,  balances the dataset so that number of benign and DDoS samples are approximately the same, splits the data structure into training, validation and test sets, normalises the features between 0 and 1 and executes the padding of samples with zeros so that they all have the same shape (since having samples of fixed shape is a requirement for a CNN to be able to learn over a full sample set).

Finally, three files (training, validation and test sets) are saved in *hierarchical data format* ```hdf5``` . 

The second step is executed with command:

```
python3 lucid_dataset_parser.py --preprocess_folder ./sample-dataset/
```

If option ```--output_folder``` is not used, the output will be produced in the input folder specified with option ```--preprocess_folder```.

At the end of this operation, the script prints a summary of the pre-processed dataset. In our case, with this tiny traffic traces, the result should be something like:

```
2020-08-27 11:02:20 | examples (tot,ben,ddos):(3518,1759,1759) | Train/Val/Test sizes: (2849,317,352) | Packets (train,val,test):(15325,1677,1761) | options:--preprocess_folder ./sample-dataset/ |
```

Which means 3518 samples in total (1759 benign and 1759 DDoS), 2849 in the training set, 317 in the validation set and 352 in the test set. The output also shows the total number of packets in the dataset divided in training, validation and test sets and the options used with the script. 

All the output of the ```lucid_dataset_parser.py``` script is saved within the output folder in the ```history.log``` file.

## Training

The LUCID CNN is implemented in script ```lucid_cnn.py```. The script executes a grid search throughout a set of hyperparameters and saves the model that maximises the F1 score metric in ```h5``` format (hierarchical data format).

At each point in the grid (each combination of hyperparameters), the training continues indefinitely and stops when the loss does not decrease for a consecutive 25 times. This value is defined with variable ```MAX_CONSECUTIVE_LOSS_INCREASE=25``` at the beginning of the script. Part of the hyperparameters is defined in the script as follows:

- **Learning rate**:  ```LR = [0.1,0.01,0.001]```
- **Batch size**: ```BATCH_SIZE = [1024,2048]```
- **Number of convolutional filters**: ```KERNELS = [1,2,4,8,16,32,64]```
- **Height of the pooling kernel**: ```pool_height in ['min','max']```, where ```min=3``` and ```max``` is the total height of the output of one convolutional filter

Other two important hyperparameters must be specified during the first step of the data preprocessing (see above):

- **Maximum number of packets/sample (n)**: indicates the maximum number of packets of a flow recorded in chronological order in a sample.
- **Time window (t)**: Time window (in seconds) used to simulate the capturing process of online systems by splitting the flows into subflows of fixed duration.

To tune LUCID with this two hyperparameters, the data preprocessing step must be executed multiple times to produce different versions of the dataset, one for each combination of **n** and **t**. This of course will produce multiple versions of the dataset with different prefixes like: ```10t-10n```, ```10t-100n```, ```100t-10n``` and ```100t-100n``` when testing with ```n=10,100``` and ```t=10,100```.

All these files can be stored into a single folder, or in multiple  subfolders. The script takes care of loading all the versions of the dataset available in the folder (and its subfolders) specified with option ```--dataset_folder```, as described below.

### Command options

To execute the training process, the following parameters can be specified when using ```lucid_cnn.py```:

- ```-t```, ```--train```: Starts the training process and specifies the folder with the dataset
- ```-e```, ```--epochs ```: Maximum number of training epochs for each set of hyperparameters. This option overrides the *early stopping* mechanism based on the loss trend described above
- ```-r```, ```--regularization ```: Applies either *l1* or *l2* regularization to the convolutional layer's connection weights
- ```-d```, ```--dropout ```: Applies *dropout* regularization to the convolutional layer's neurons.

### The training process

To train LUCID, execute the following command:

```
python3 lucid_cnn.py --train ./sample-dataset/  --epochs 100 --regularization l2 --dropout 0.5
```

This command trains LUCID over the grid of hyperparameters, 100 epochs for each point in the grid. During the training, *l2* and *dropout* regularization methods are used, the latter with *dropout rate* equal to 50%. The output is saved in a text file in the same folder containing the dataset. In that folder, the model that maximises the F1 score on the validation set is also saved in ```h5``` format, along with a ```csv``` file with the performance of the model.  The name of the two files is the same (except for the extension) and is in the following format:

```
10t-10n-SYN2020-LUCID.h5
10t-10n-SYN2020-LUCID.csv
```

Where the prefix 10t-10n indicates the values of hyperparameters ```time window``` and ```packets/sample``` that produced the best results in terms of F1 score on the validation set. The values of the other hyperparameters are reported in the ```csv``` file:

```
Model         TIME(sec) ACC    ERR    F1     PPV    TPR    FPR    TNR    FNR    Parameters
SYN2020-LUCID     0.078 1.0000 0.2634 1.0000 1.0000 1.0000 0.0000 1.0000 0.0000 lr=0.100,b=1024,n=010,t=010,k=001,h=(03,11),m=min
```

along with prediction results obtained on the validation set, such as execution time (TIME), accuracy (ACC), loss (ERR), F1 score (F1), positive predictive value or precision (PPV), true positive rate (TPR), false positive rate (FPR), true negative rate (TNR) and false negative rate (FNR).

The hyperparameters are learning rate (lr), batch size (b), packet/sample (n), time window (t), number of convolutional filters (k), filter size (h), max pooling size (m).

## Testing

Testing means evaluating a trained model of LUCID with unseen data (data not used during the training and validation steps), such as the test set in the ```sample-dataset``` folder. For this process,  the ```lucid_cnn.py``` provides a different set of options:

- ```-p```, ```--predict```: Perform prediction on the test sets contained in a given folder specified with this option. The folder must contain files in ```hdf5``` format with the ```test``` suffix
- ```-m```, ```--model```: Model to be used for the prediction. The model in ```h5``` format produced with the training
- ```-i```, ```--iterations```: Repetitions of the prediction process (useful to estimate the average prediction time)

To test LUCID, run the following command:

```
python3 lucid_cnn.py --predict ./sample-dataset/ --model ./sample-dataset/10t-10n-SYN2020-LUCID.h5
```

The output printed on the terminal and saved in a text file in the folder with the dataset. The output has the following format:

```
Model         TIME(sec) PACKETS SAMPLES DDOS% ACC    ERR    F1     PPV    TPR    FPR    TNR    FNR    Data Source
SYN2020-LUCID     0.036 0001761 0000352 0.503 0.9915 0.2944 0.9915 0.9831 1.0000 0.0169 0.9831 0.0000 10t-10n-SYN2020-dataset-test.hdf5
```

Where ```TIME``` is the execution time on a test set.  The values of ```PACKETS``` and ```SAMPLES``` are the the total number of packets and samples in the test set respectively. More precisely, ```PACKETS``` is the total amount of packets represented in the samples (traffic flows) of the test set. ```ERR``` is the total loss on the test set (binary cross entropy),  ```ACC```, ```F1```, ```PPV```  are classification accuracy, F1 and precision scores respectively, ```TPR```, ```FPR```, ```TNR```, ```FNR``` are the true positive, false positive, true negative and false negative rates respectively. 

The last column indicates the name of the test set used for the prediction test. Note that the script loads and process all the test sets in the folder specified with option ``` --predict``` (identified with the suffix ```test.hdf5```). This means that the output is formed by multiple lines, on for each test set. 

## Online Inference

Once trained, LUCID can perform inference on live network traffic or on pre-recorded traffic traces saved in ```pcap``` format. This operational mode is implemented in the ```lucid_cnn.py``` script and leverages on ```pyshark``` and ```tshark``` tools to capture the network packets from one of the network cards of the machine where the script is executed, or to extract the packets from a ```pcap``` file. In both cases, the script simulates an online deployment, where the traffic is collected for a predefined amount of time (```time_window```) and then sent to the neural network for classification.

Online inference can be started by executing ```lucid_cnn.py``` followed by one or more of these options: 

- ```-pl```, ```--predict_live```: Perform prediction on the network traffic sniffed from a network card or from a ```pcap``` file available on the file system. Therefore, this option must be followed by either the name of a network interface (e.g., ```eth0```) or the path to a ```pcap``` file (e.g., ```/home/user/traffic_capture.pcap```)
- ```-m```, ```--model```: Model to be used for the prediction. The model in ```h5``` format produced with the training
- ```-y```, ```--dataset_type```: One between ```IDS2017``` and ```IDS2018``` in the case of ```pcap``` files from the UNB's datasets, or ```SYN2020``` for the custom dataset provided with this code. This option is not used by LUCID for the classification task, but only to produce the classification statistics (e.g., accuracy, F1 score, etc,) by comparing the ground truth labels with the LUCID's output
- ```-a```, ```--attack_net```: Specifies the subnet of the attack network (e.g., ```192.168.0.0/24```). Like option ```dataset_type```, this is used to generate the ground truth labels. This option is used, along with option ```victim_net```, in the case of custom traffic or pcap file with IP address schemes different from those in the three datasets ```IDS2017```,  ```IDS2018```  or ```SYN2020``` 
- ```-y```, ```--victim_net```: The subnet of the victim network (e.g., ```10.42.0.0/24```), specified along with option ```attack_net``` (see description above).

### Inference on live traffic

If the argument of ```predict_live``` option is a network interface, LUCID will sniff the network traffic from that interface and will return the classification results every time the time window expires. The duration of the time window is automatically detected from the prefix of the model's name (e.g., ```10t``` indicates a 10-second time window). To start the inference on live traffic, use the following command:

```
python3 lucid_cnn.py --predict_live eth0 --model ./sample-dataset/10t-10n-SYN2020-LUCID.h5 --dataset_type SYN2020
```

Where ```eth0``` is the name of the network interface, while ```dataset_type``` indicates the address scheme of the traffic. This is optional and, as written above, it is only used to obtain the ground truth labels needed to compute the classification accuracy.

In the example, ```SYN2020``` refers to a SYN flood attack built using the following addressing scheme, defined in ```lucid_dataset_parser.py```, and used in the sample dataset:

```
CUSTOM_DDOS_SYN = {'attackers': ['11.0.0.' + str(x) for x in range(1,255)],
                      'victims': ['10.42.0.2']}
```

Of course, the above dictionary can be changed to meet the address scheme of the network where the experiments are executed. Alternatively, one can use the ```attack_net``` and ```victim_net``` options as follows:

```
python3 lucid_cnn.py --predict_live ens3 --model ./sample-dataset/10t-10n-SYN2020-LUCID.h5 --attack_net 11.0.0.0/24 --victim_net 10.42.0.0/24
```

Once LUCID has been started on the victim machine using one of the two examples above, we can start the attack from another host machine using one of the following scripts based on the ```mausezahn``` tool (https://github.com/uweber/mausezahn):

```
sudo mz eth0 -A  11.0.0.0/24 -B 10.42.0.2 -t tcp " dp=80,sp=1024-60000,flags=syn"
```

In this script, ```eth0``` refers to the egress network interface on the attacker's machine and ```10.42.0.2``` is the IP address of the victim machine.

The output of LUCID on the victim machine will be similar to that reported in Section **Testing** above. 

### Inference on pcap files

Similar to the previous case on live traffic, inference on a pre-recorded traffic trace can be started with command:

```
python3 lucid_cnn.py --predict_live ./sample-dataset/dataset-chunk-syn.pcap --model ./sample-dataset/10t-10n-SYN2020-LUCID.h5 --dataset_type SYN2020
```

In this case, the argument of option ```predict_live``` must be the path to a pcap file. The script parses the file from the beginning to the end, printing the classification results every time the time window expires. The duration of the time window is automatically detected from the prefix of the model's name (e.g., ```10t``` indicates a 10-second time window). 

The output of LUCID on the victim machine will be similar to that reported in Section **Testing** above. 


## Acknowledgements

If you are using LUCID's code for a scientific research, please cite the related paper in your manuscript as follows:

*R. Doriguzzi-Corin, S. Millar, S. Scott-Hayward, J. Martínez-del-Rincón and D. Siracusa, "Lucid: A Practical, Lightweight Deep Learning Solution for DDoS Attack Detection," in IEEE Transactions on Network and Service Management, vol. 17, no. 2, pp. 876-889, June 2020, doi: 10.1109/TNSM.2020.2971776.*



This work has been partially funded by the European Union’s Horizon 2020 Research and Innovation Programme under grant agreements no. 815141 (DECENTER project) and n. 833685 (SPIDER project).

## License

The code is released under the Apache License, Version 2.0.

