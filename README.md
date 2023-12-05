# LUCID: A Practical, Lightweight Deep Learning Solution for DDoS Attack Detection

LUCID (Lightweight, Usable CNN in DDoS Detection) is a lightweight Deep Learning-based DDoS detection framework suitable for online resource-constrained environments, which leverages Convolutional Neural Networks (CNNs) to learn the behaviour of DDoS and benign traffic flows with both low processing overhead and attack detection time. LUCID includes a dataset-agnostic pre-processing mechanism that produces traffic observations consistent with those collected in existing online systems, where the detection algorithms must cope with segments of traffic flows collected over pre-defined time windows.

More details on the architecture of LUCID and its performance in terms of detection accuracy and execution time are available in the following research paper:

R. Doriguzzi-Corin, S. Millar, S. Scott-Hayward, J. Martínez-del-Rincón and D. Siracusa, "Lucid: A Practical, Lightweight Deep Learning Solution for DDoS Attack Detection," in *IEEE Transactions on Network and Service Management*, vol. 17, no. 2, pp. 876-889, June 2020, doi: 10.1109/TNSM.2020.2971776.



## Installation

The current LUCID's CNN is implemented in Python v3.9 with Keras and Tensorflow 2, while the traffic pre-processing tool is implemented in Python v3.9, Numpy and Pyshark. The original version, implemented with Tensorflow 1.13.1 and evaluated in the aforementioned paper, is available in branch *lucid-tensorflow-1.13.1*.

LUCID requires the installation of a number of Python tools and libraries. This can be done by using the ```conda``` software environment (https://docs.conda.io/projects/conda/en/latest/).
We suggest the installation of ```miniconda```, a light version of ```conda```. ```miniconda``` is available for MS Windows, MacOSX and Linux and can be installed by following the guidelines available at https://docs.conda.io/en/latest/miniconda.html#. 

Execute one of the following commands (based on your operating system) and follow the on-screen instructions:

```
bash Miniconda3-latest-Linux-x86_64.sh (on Linux operating systems)
bash Miniconda3-latest-MacOSX-x86_64.sh (on macOS)
```

Then create a new ```conda``` environment (called ```python39```) based on Python 3.9:

```
conda create -n python39 python=3.9
```

Activate the new ```python39``` environment:

```
conda activate python39
```

And configure the environment with ```tensorflow``` and a few more packages:

On Linux operating systems:
```
(python39)$ pip install tensorflow==2.7.1
(python39)$ pip install scikit-learn h5py pyshark protobuf==3.19.6
```

On macOS (tested on Apple M1 CPU)
```
(python39)$ conda install -c conda-forge tensorflow=2.7.1
(python39)$ conda install -c conda-forge scikit-learn h5py pyshark
```
Pyshark is used in the ```lucid_dataset_parser.py``` script for data pre-processing.
Pyshark is just Python wrapper for tshark, meaning that ```tshark``` must be also installed. On an Ubuntu-based OS, use the following command:

```
sudo apt install tshark
```

Please note that the current parser code works with ```tshark``` **version 3.2.3 or lower** or **version 3.6 or higher**. Issues have been reported when using intermediate releases such as 3.4.X.

For the sake of simplicity, we omit the command prompt ```(python39)$``` in the following example commands in this README.   ```(python39)$``` indicates that we are working inside the ```python39``` execution environment, which provides all the required libraries and tools. If the command prompt is not visible, re-activate the environment as explained above.

## Traffic pre-processing

LUCID requires a labelled dataset, including the traffic traces in the format of ```pcap``` files. The traffic pre-processing functions are implemented in the ```lucid_dataset_parser.py``` Python script. It currently supports three DDoS datasets from the University of New Brunswick (UNB) (https://www.unb.ca/cic/datasets/index.html): CIC-IDS2017, CSE-CIC-IDS2018 and CIC-DDoS2019. A portion of the latter will be used for this guide and included in the ```sample-dataset``` folder.

With term *support*, we mean the capability of the script to correctly label the packets and the traffic flows either as benign or DDoS. In general, this is done by parsing a file with the labels provided with the traffic traces, like in the case of the UNB datasets, or by manually indicating the IP address(es) of the attacker(s) and the IP address(es) of the victim(s) in the code. Of course, also in the latter case, the script must be tuned with the correct information of the traffic (all the attacker/victim pairs of IP addresses), as this information is very specific to the dataset and to the methodology used to generate the traffic. 

Said that, ```lucid_dataset_parser.py``` includes the structures with the pairs attacker/victim of the three datasets mentioned above (CIC-IDS2017, CSE-CIC-IDS2018 and CIC-DDoS2019), but it can be easily extended to support other datasets by replicating the available code.

For instance, the following Python dictionary provides the IP addresses of the attackers and the victims involved in the CIC-DDoS2019 dataset:   

```
DOS2019_FLOWS = {'attackers': ['172.16.0.5'], 'victims': ['192.168.50.1', '192.168.50.4']}
```
Please note that in the CIC-DDoS2019 dataset, the source IP addresses of DDoS packets have been replaced with the IP address of the router that connects the attack network and the victim network within the testbed ([CIC-DDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html)).

### Command options

The following parameters can be specified when using ```lucid_dataset_parser.py```:

- ```-d```, ```--dataset_folder```: Folder with the dataset
- ```-o```, ```--output_folder ```: Folder where the scripts save the output. The dataset folder is used when this option is not used
- ```-f```, ```--traffic_type ```: Type of flow to process (all, benign, ddos)
- ```-p```, ```--preprocess_folder ```: Folder containing the intermediate files ```*.data```
- ```-t```, ```--dataset_type ```: Type of the dataset. Available options are: DOS2017, DOS2018, DOS2019
- ```-n```, ```--packets_per_flow ```: Maximum number of packets in a sample
- ```-w```, ```--time_window ```: Length of the time window (in seconds)
- ```-i```, ```--dataset_id ```: String to append to the names of output files



### First step

In the first step, the parser extracts the features from the packets of all the ```pcap``` files contained in the source directory. The features are grouped in flows and labelled according to the value of parameter ```dataset_type```. Specifically, a flow is a set of features from packets with the same source IP, source UDP/TCP port, destination IP and destination UDP/TCP port and protocol. Flows are bi-directional, therefore, packet (srcIP,srcPort,dstIP,dstPort,proto) belongs to the same flow of (dstIP,dstPort,srcIP,srcPort,proto). The result is a set of intermediate binary files with extension ```.data```.

This first step can be executed with command:

```
python3 lucid_dataset_parser.py --dataset_type DOS2019 --dataset_folder ./sample-dataset/ --packets_per_flow 10 --dataset_id DOS2019 --traffic_type all --time_window 10
```

This will process in parallel the two files, producing a file named ```10t-10n-DOS2019-preprocess.data```. In general, the script loads all the files with extension ```.pcap``` contained in the folder indicated with option ```--dataset_folder```. The files are processed in parallel to minimise the execution time.

Prefix ```10t-10n``` means that the pre-processing has been done using a time window of 10 seconds (10t) and a flow length of 10 packets (10n). Please note that ```DOS2019``` in the filename is the result of option ```--dataset_id DOS2019``` in the command.

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
2023-05-19 15:37:17 | examples (tot,ben,ddos):(7486,3743,3743) | Train/Val/Test sizes: (6060,677,749) | Packets (train,val,test):(24850,2763,3285) | options:--preprocess_folder ./sample-dataset/ |
```

Which means 7486 samples in total (3743 benign and 3743 DDoS), 6060 in the training set, 677 in the validation set and 749 in the test set. The output also shows the total number of packets in the dataset divided in training, validation and test sets and the options used with the script. 

All the output of the ```lucid_dataset_parser.py``` script is saved within the output folder in the ```history.log``` file.

## Training

The LUCID CNN is implemented in script ```lucid_cnn.py```. The script executes a grid search throughout a set of hyperparameters and saves the model that maximises the accuracy on the validation set in ```h5``` format (hierarchical data format).

At each point in the grid (each combination of hyperparameters), the training continues until the maximum number of epochs is reached or after the loss has not decreased for 10 consecutive times. This value is defined with variable ```PATIENCE=10``` at the beginning of the script. Part of the hyperparameters is defined in the script as follows:

- **Learning rate**:  ```"learning_rate": [0.1,0.01],```
- **Batch size**: ```"batch_size": [1024,2048]```
- **Number of convolutional filters**: ```"kernels": [32,64]```
- **L1 and L2 Regularization**: ```"regularization" : [None,'l1']```
- **Dropout**: ```"dropout" : [None,0.5]```

Other two important hyperparameters must be specified during the first step of the data preprocessing (see above):

- **Maximum number of packets/sample (n)**: indicates the maximum number of packets of a flow recorded in chronological order in a sample.
- **Time window (t)**: Time window (in seconds) used to simulate the capturing process of online systems by splitting the flows into subflows of fixed duration.

To tune LUCID with this two hyperparameters, the data preprocessing step must be executed multiple times to produce different versions of the dataset, one for each combination of **n** and **t**. This of course will produce multiple versions of the dataset with different prefixes like: ```10t-10n```, ```10t-100n```, ```100t-10n``` and ```100t-100n``` when testing with ```n=10,100``` and ```t=10,100```.

All these files can be stored into a single folder, or in multiple  subfolders. The script takes care of loading all the versions of the dataset available in the folder (and its subfolders) specified with option ```--dataset_folder```, as described below.

### Command options

To execute the training process, the following parameters can be specified when using ```lucid_cnn.py```:

- ```-t```, ```--train```: Starts the training process and specifies the folder with the dataset
- ```-e```, ```--epochs ```: Maximum number of training epochs for each set of hyperparameters (default=1000)
- ```-cv```,```--cross_validation```: Number of folds used to split the training data for cross-validation (accepted values: 0,2,3, etc., 0 is the default value that means no cross-validation). 


### The training process

To train LUCID, execute the following command:

```
python3 lucid_cnn.py --train ./sample-dataset/
```

This command trains LUCID over the grid of hyperparameters, maximum 1000 epochs for each point in the grid. The training process can stop earlier if no progress towards the minimum loss is observed for PATIENCE=10 consecutive epochs. The model which maximises the accuracy on the validation set is saved in ```h5``` format in the ```output``` folder, along with a ```csv``` file with the performance of the model on the validation set.  The name of the two files is the same (except for the extension) and is in the following format:

```
10t-10n-DOS2019-LUCID.h5
10t-10n-DOS2019-LUCID.csv
```

Where the prefix 10t-10n indicates the values of hyperparameters ```time window``` and ```packets/sample``` that produced the best results in terms of F1 score on the validation set. The values of the other hyperparameters are reported in the ```csv``` file:

|Model|Samples|Accuracy|F1Score|Hyper-parameters|Validation Set|
|-----|-------|--------|-------|----------------|--------------|
|DOS2019-LUCID|677|0.9261|0.9256|"{'batch_size': 1024, 'dropout': 0.5, 'kernels': 64, 'learning_rate': 0.1, 'regularization': None}"|./sample-dataset/10t-10n-DOS2019-dataset-val.hdf5|

along with prediction results obtained on the validation set, such as  accuracy and F1 score. The last column is the path to the validation set used to validate the best model.


## Testing

Testing means evaluating a trained model of LUCID with unseen data (data not used during the training and validation steps), such as the test set in the ```sample-dataset``` folder. For this process,  the ```lucid_cnn.py``` provides a different set of options:

- ```-p```, ```--predict```: Perform prediction on the test sets contained in a given folder specified with this option. The folder must contain files in ```hdf5``` format with the ```test``` suffix
- ```-m```, ```--model```: Model to be used for the prediction. The model in ```h5``` format produced with the training
- ```-i```, ```--iterations```: Repetitions of the prediction process (useful to estimate the average prediction time)

To test LUCID, run the following command:

```
python3 lucid_cnn.py --predict ./sample-dataset/ --model ./output/10t-10n-DOS2019-LUCID.h5
```

The output printed on the terminal and saved in a text file in the ```output``` folder in the following format:

|Model|Time|Packets|Samples|DDOS%|Accuracy|F1Score|TPR|FPR|TNR|FNR|Source|
|-----|----|-------|-------|-----|--------|-------|---|---|---|---|------|
|DOS2019-LUCID|0.016|3285|749|0.437|0.9947|0.9939|0.9909|0.0024|0.9976|0.0091|10t-10n-DOS2019-dataset-test.hdf5|

Where ```Time``` is the execution time on a test set.  The values of ```Packets``` and ```Samples``` are the the total number of packets and samples in the test set respectively. More precisely, ```Packets``` is the total amount of packets represented in the samples (traffic flows) of the test set. ```Accuracy```, ```F1```, ```PPV```  are classification accuracy, F1 and precision scores respectively, ```TPR```, ```FPR```, ```TNR```, ```FNR``` are the true positive, false positive, true negative and false negative rates respectively. 

The last column indicates the name of the test set used for the prediction test. Note that the script loads and process all the test sets in the folder specified with option ``` --predict``` (identified with the suffix ```test.hdf5```). This means that the output might consist of multiple lines, on for each test set. 

## Online Inference

Once trained, LUCID can perform inference on live network traffic or on pre-recorded traffic traces saved in ```pcap``` format. This operational mode is implemented in the ```lucid_cnn.py``` script and leverages on ```pyshark``` and ```tshark``` tools to capture the network packets from one of the network cards of the machine where the script is executed, or to extract the packets from a ```pcap``` file. In both cases, the script simulates an online deployment, where the traffic is collected for a predefined amount of time (```time_window```) and then sent to the neural network for classification.

Online inference can be started by executing ```lucid_cnn.py``` followed by one or more of these options: 

- ```-pl```, ```--predict_live```: Perform prediction on the network traffic sniffed from a network card or from a ```pcap``` file available on the file system. Therefore, this option must be followed by either the name of a network interface (e.g., ```eth0```) or the path to a ```pcap``` file (e.g., ```/home/user/traffic_capture.pcap```)
- ```-m```, ```--model```: Model to be used for the prediction. The model in ```h5``` format produced with the training
- ```-y```, ```--dataset_type```: One between ```DOS2017```, ```DOS2018``` and ```DOS2019``` in the case of ```pcap``` files from the UNB's datasets. This option is not used by LUCID for the classification task, but only to produce the classification statistics (e.g., accuracy, F1 score, etc,) by comparing the ground truth labels with the LUCID's output
- ```-a```, ```--attack_net```: Specifies the subnet of the attack network (e.g., ```192.168.0.0/24```). Like option ```dataset_type```, this is used to generate the ground truth labels. This option is used, along with option ```victim_net```, in the case of custom traffic or pcap file with IP address schemes different from those in the three datasets ```DOS2017```,  ```DOS2018``` or ```DOS2019``` 
- ```-y```, ```--victim_net```: The subnet of the victim network (e.g., ```10.42.0.0/24```), specified along with option ```attack_net``` (see description above).

### Inference on live traffic

If the argument of ```predict_live``` option is a network interface, LUCID will sniff the network traffic from that interface and will return the classification results every time the time window expires. The duration of the time window is automatically detected from the prefix of the model's name (e.g., ```10t``` indicates a 10-second time window). To start the inference on live traffic, use the following command:

```
python3 lucid_cnn.py --predict_live eth0 --model ./output/10t-10n-DOS2019-LUCID.h5 --dataset_type DOS2019
```

Where ```eth0``` is the name of the network interface, while ```dataset_type``` indicates the address scheme of the traffic. This is optional and, as written above, it is only used to obtain the ground truth labels needed to compute the classification accuracy.

In the example, ```DOS2019``` refers to the UNB's CIC-DDoS2019 dataset, which has been built using the  addressing scheme defined in ```lucid_dataset_parser.py```:

```
DOS2019_FLOWS = {'attackers': ['172.16.0.5'], 'victims': ['192.168.50.1', '192.168.50.4']}
```

Of course, the above dictionary can be changed to meet the address scheme of the network where the experiments are executed. Alternatively, one can use the ```attack_net``` and ```victim_net``` options as follows:

```
python3 lucid_cnn.py --predict_live eth0 --model ./output/10t-10n-DOS2019-LUCID.h5 --attack_net 11.0.0.0/24 --victim_net 10.42.0.0/24
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
python3 lucid_cnn.py --predict_live ./sample-dataset/CIC-DDoS-2019-UDPLag.pcap --model ./output/10t-10n-DOS2019-LUCID.h5 --dataset_type DOS2019
```

In this case, the argument of option ```predict_live``` must be the path to a pcap file. The script parses the file from the beginning to the end, printing the classification results every time the time window expires. The duration of the time window is automatically detected from the prefix of the model's name (e.g., ```10t``` indicates a 10-second time window). 

The output of LUCID on the victim machine will be similar to that reported in Section **Testing** above. 


## Acknowledgements

If you are using LUCID's code for scientific research, please cite the related paper in your manuscript as follows:

*R. Doriguzzi-Corin, S. Millar, S. Scott-Hayward, J. Martínez-del-Rincón and D. Siracusa, "Lucid: A Practical, Lightweight Deep Learning Solution for DDoS Attack Detection," in IEEE Transactions on Network and Service Management, vol. 17, no. 2, pp. 876-889, June 2020, doi: 10.1109/TNSM.2020.2971776.*



This work has been partially funded by the European Union’s Horizon 2020 Research and Innovation Programme under grant agreements no. 815141 (DECENTER project), 830929 (CyberSec4Europe project) and n. 833685 (SPIDER project).

## License

The code is released under the Apache License, Version 2.0.