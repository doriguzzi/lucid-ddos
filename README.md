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

Then create a new ```conda``` environment (called ```myenv```) based on Python 3.8 and including part the required packages:

```
conda create -n myenv python=3.8 numpy tensorflow=2.3.0 h5py lxml
```

Activate the new ```myenv``` environment:

```
conda activate myenv
```

And finalise the installation with a few more packages:

```
(myenv)$ pip3 install pyshark sklearn
```

Please note that Pyshark is just Python wrapper for tshark, allowing python packet parsing using wireshark dissectors. This means that ```tshark``` must be also installed. On an Ubuntu-based OS use  the following command:

```
sudo apt install tshark
```

For the sake of simplicity, we omit the command prompt ```(myenv)$``` in the following example commands in this README.   ```(myenv)$``` indicates that we are working inside the ```myenv``` execution environment, which provides all the required libraries and tools. If the command prompt is not visible, re-activate the environment as explained above.

## Traffic pre-processing

LUCID requires a labelled dataset, including the traffic traces in the format of ```pcap``` files. The traffic pre-processing functions are implemented in the ```lucid-dataset-parser.py``` Python script. It currently supports three DDoS datasets from the University of New Brunswick (UNB) (https://www.unb.ca/cic/datasets/index.html): ISCXIDS2012, CIC-IDS2017 and CSE-CIC-IDS2018, plus a custom dataset containing a SYN Flood DDoS attack that will be used for this guide and included in the ```sample-dataset``` folder.

With term *support*, we mean the capability of the script to correctly label the packets and the traffic flows inside the traffic traces as benign or DDoS. In general, this is done by parsing a file with the labels provided with the traffic traces, as in the case of the UNB datasets, or by manually indicating the  IP address(es) of the attacker(s) and the IP address(es) of the victim(s) in the code. Of course, also in the latter case, the script must be tuned with the correct information of the traffic (all the attacker/victim pairs of IP addresses), as this information is very specific to the dataset and to the methodology used to generate the traffic. 

Said that, ```lucid-dataset-parser.py``` implements both approaches, therefore it can be easily extended to support other datasets by replicating the available code. In  the current version, only the dataset  ISCXIDS2012 needs the file with the labels (which can be obtained from the UNB's repository), while for all the others mentioned above we have already included the structures with the pairs attacker/victim. For instance, the following Python dictionary provides the IP addresses of the 254 attackers and the victim involved in the custom SYN Flood attack:   

```
CUSTOM_DDOS_SYN = {'attackers': ['11.0.0.' + str(x) for x in range(1,255)],
                      'victims': ['10.42.0.2']}
```

### Command options

The following parameters can be specified when using ```lucid-dataset-parser.py```:

- ```-d```, ```--dataset_folder```: Folder with the dataset
- ```-o```, ```--output_folder ```: Folder where  the scripts saves the output. The dataset folder is used when this option is not used
- ```-f```, ```--traffic_type ```: Type of flow to process (all, benign, ddos)
- ```-p```, ```--preprocess_folder ```: Folder containing the intermediate files ```*.data```
- ```-t```, ```--dataset_type ```: Type of the dataset. Available options are: IDS2012, IDS2017, IDS2018, SYN2020
- ```-n```, ```--packets_per_flow ```: Maximum number of packets in a sample
- ```-w```, ```--time_window ```: Length of the time window (in seconds)
- ```-i```, ```--dataset_id ```: String to append to the names of output files



### First step

The traffic pre-processing operation comprises two steps. The first parses the file with the labels (if needed) all extracts the features from the packets of all the ```pcap``` files contained in the source directory. The features are grouped in flows, where a flow is a set of features from packets with the same source IP, source UDP/TCP port, destination IP and destination UDP/TCP port and protocol. Flows are bi-directional, therefore, packet (srcIP,srcPort,dstIP,dstPort,proto) belongs to the same flow of (dstIP,dstPort,srcIP,srcPort,proto). The result, is a set of intermediate binary files with extension ```.data```.

This first step can be executed with command:

```
python3 lucid-dataset-parser.py --dataset_type SYN2020 --dataset_folder ./sample-dataset/ --packets_per_flow 10 --dataset_id SYN2020 --traffic_type all --time_window 10
```

This will process in parallel the two files, producing a file named ```10t-10n-SYN2020-preprocess.data```. In general, the script loads all the ```pcap``` files contained in the folder indicated with option ```--dataset_folder``` and starting with prefix ```dataset-chunk-```. The files are processed in parallel to minimise the execution time.

Prefix ```10t-10n``` means that the pro-processing has been done using a time window of 10 seconds (10t) and a flow length of 10 packets (10n). Please note that ```SYN2020``` in the filename is the result of option ```--dataset_id SYN2020``` in the command.

Time window and flow length are two hyperparameters of LUCID. For more information, please refer to the research paper mentioned above. 

### Second step

The second step loads the ```*.data``` files, merges them into a single data structure stored in RAM memory,  balances the dataset so that number of benign and DDoS samples are approximately the same, splits the data structure into training, validation and test sets, normalises the features between 0 and 1 and executes the padding of samples with zeros so that they all have the same shape (since having samples of fixed shape is a requirement for a CNN to be able to learn over a full sample set).

Finally, three files (training, validation and test sets) are saved in *hierarchical data format* ```hdf5``` . 

The second step is executed with command:

```
python3 lucid-dataset-parser.py --preprocess_folder ./sample-dataset/
```

If the option ```--output_folder``` is not used,  the output will be produced in the input folder specified with option ```--preprocess_folder```.

At the end of this operation, the script prints a summary of the pre-processed dataset. In our case, with this tiny traffic traces, the result should be something like:

```
2020-08-27 11:02:20 | examples (tot,ben,ddos):(3518,1759,1759) | Train/Val/Test sizes: (2849,317,352) | Packets (train,val,test):(15325,1677,1761) | options:--preprocess_folder ./sample-dataset/ |
```

Which means 3518 samples in total (1759 benign and 1759 DDoS), 2849 in the training set, 317 in the validation set and 352 in the test set. The output also shows the total number of packets in the dataset divided in training, validation and test sets and the options used with the script. 

All the output of the ```lucid-dataset-parser.py``` script is saved within the output folder in the ```history.log``` file.

## Training

The LUCID CNN  is implemented in script ```lucid-cnn.py```. The script executes a grid search throughout a set of hyperparameters and saves the model that maximises the F1 score metric in ```h5``` format (hierarchical data format).

At each point in the grid (each combination of hyperparameters), the training continues indefinitely and stops when the loss does not decrease for a consecutive 25 times. This value is defined with variable ```MAX_CONSECUTIVE_LOSS_INCREASE=25``` at the beginning of the script. Part of the hyperparameters are defined in the script as follows:

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

To execute the training process, the following parameters can be specified when using ```lucid-cnn.py```:

- ```-t```, ```--train```: Start the training process and specifies the folder with the dataset
- ```-e```, ```--epochs ```: Maximum number of training epochs for each set of hyperparameters. This option overrides the *early stopping* mechanism based on the loss trend described above.

### The training process

To train LUCID, execute the following command:

```
python3 lucid-cnn.py --train ./sample-dataset/  --epochs 100
```

This command trains LUCID over the grid of hyperparameters, 100 epochs for each point in the grid. The output is saved in a text file in the same folder containing the dataset. In that folder, the model that maximises the F1 score on the validation set is also saved in ```h5``` format, along with a ```csv``` file with the performance of the model.  The name of the two files is the same (except for the extension) and is in the following format:

```
10t-10n-SYN2020-LUCID.h5
10t-10n-SYN2020-LUCID.csv
```

Where the prefix 10t-10n indicates the values of hyperparameters ```time window``` and ```packets/sample``` that produced the best results in terms of F1 score on the validation set. The values of the other hyperparameters are reported in the ```csv``` file:

```
Model     TIME    ACC     ERR     PRE     REC     F1      AUC     Parameters
LUCID     000.040 1.00000 0.08495 1.00000 1.00000 1.00000 1.00000 lr=0.100,b=2048,n=010,t=010,k=001,h=(03,11),m=min
```

along with prediction results obtained on the validation set, such as execution time (TIME), accuracy (ACC), loss (ERR), precision (PRE), recall (REC), F1 score and area under the curve (AUC).

The hyperparameters are learning rate (lr), batch size (b), packet/sample (n), time window (t), number of convolutional filters (k), filter size (h), max pooling size (m).

 

## Testing

Testing means evaluating a trained model of LUCID with unseen data (data not used during the training and validation steps), such as the test set in the ```sample-dataset``` folder. For this process,  the ```lucid-cnn.py``` provides a different set of options:

- ```-p```, ```--predict```: Perform prediction on the test sets contained in a given folder specified with this option. The folder must contain files in ```hdf5``` format with the ```test``` suffix
- ```-m```, ```--model```: Model to be used for the prediction. The model in ```h5``` format produced with the training
- ```-i```, ```--iteration```: Repetitions of the prediction process (useful to estimate the average prediction time)

To test LUCID, run the following command:

```
python3 lucid-cnn.py --predict ./sample-dataset/ --model ./sample-dataset/10t-10n-SYN2020-LUCID.h5
```

The output printed on the terminal and saved in a text file in the folder with the dataset. The output has the following format:

```
Model         TIME    PACKETS    PKT/SEC    SAMPLE/SEC ACC     ERR      PRE     REC     F1      AUC     
SYN2020-LUCID 000.002 0000001761 0001122689 0000224410 0.99148 00.29437 0.98305 1.00000 0.99145 0.99157 
```

```
TN      FP      FN      TP      DatasetName
0.98315 0.01685 0.00000 1.00000 10t-10n-SYN2020-dataset-test.hdf5
```

Where ```TIME``` is the execution time, ```PACKETS``` is the number of packets present in the test set as part of the samples, ```PKT/SEC``` is the number of packet processed in a second (useful in online system to understand the max throughput supported by LUCID on the testing machine), ```SAMPLE/SEC``` is the number of samples processed in a second, ```ERR``` is the total loss on the test set (binary cross entropy),  ```PRE     REC     F1      AUC```  are precision, recall, F1 score and area under the curve respectively, ```TN      FP      FN      TP``` are the true negative, false positive, false negative and true positive rates respectively. 

The last column indicates the name of the test set used for the prediction test. Note that the script loads and process all the test sets in the folder specified with option ``` --predict``` (identified with the suffix ```test.hdf5```). This means that the output is formed by multiple lines, on for each test set. 

## 

## Acknowledgements

If you are using LUCID's code for a scientific research, please cite the related paper in your manuscript as follows:

*R. Doriguzzi-Corin, S. Millar, S. Scott-Hayward, J. Martínez-del-Rincón and D. Siracusa, "Lucid: A Practical, Lightweight Deep Learning Solution for DDoS Attack Detection," in IEEE Transactions on Network and Service Management, vol. 17, no. 2, pp. 876-889, June 2020, doi: 10.1109/TNSM.2020.2971776.*



This work has been partially funded by the European Union’s Horizon 2020 Research and Innovation Programme under grant agreement no. 815141 (DECENTER project).

## License

The code is released under the Apache License, Version 2.0.

