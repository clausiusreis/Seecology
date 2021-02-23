![alt text](https://github.com/clausiusreis/Seecology/blob/master/Seecology/static/images/Seecology.png?raw=true)

## A visualization framework for acoustic ecology applications

Author: Clausius Duque Reis - UFV-CRP / ICMC-USP (clausiusreis@gmail.com).

Advisor: Maria Cristina Ferreira de Oliveira - ICMC-USP.

Seecology is a framework for feature extraction and visualization of ecological recordings, providing an easy way to visualize the resulting data immediately after the extraction process. Our solution provides a simple method to extract data with individual settings for each feature.

Among the visualizations available at this moment, we provide a comparative line chart with media player capabilities, a concentric radviz with a labeling system to help create annotated datasets and a boat detection visualization to display the result of a novel algorithm developed during my PhD, called FAVS (Frequency Amplitude Variation Signature), capable of high sensitive detection of the presence (With specific signatures) of vessels, even in noisy conditions.

More visualizations will be added in the next few months.

## Citation
Research published on the content of this framework:

[A Visualization Framework for Feature Investigation in Soundscape Recordings](https://www.researchgate.net/publication/327390554_A_Visualization_Framework_for_Feature_Investigation_in_Soundscape_Recordings) (September 2018)

[Automatic Detection of Vessel Signatures in Audio Recordings with Spectral Amplitude Variation Signature](https://www.researchgate.net/publication/334057825_Automatic_Detection_of_Vessel_Signatures_in_Audio_Recordings_with_Spectral_Amplitude_Variation_Signature) (June 2019)

## Installation procedures
Using Miniconda (Or Anaconda), open a terminal and go to [Seecology/Install](https://github.com/clausiusreis/Seecology/blob/master/Seecology/Install/). Change the prefix path at the end of the [seecologyEnv.yml](https://github.com/clausiusreis/Seecology/blob/master/Seecology/Install/seecologyEnv.yml) file according to where you want to install the new environment, and then run the command: **conda env create -f seecologyEnv.yml**. This should install all the necessary libraries. The framework was designed to work on a Linux Mint 19.1 Cinnamon machine, however, the code should be able to run on most (Linux) machines running a Miniconda/Anaconda Python distribution.

A windows version is on development at the moment, pending specific compatible libraries for the Windows OS.

## Running the framework UI
From the terminar, enter the newly created environment with the command: **source activate seecologyEnv**. Now you can navigate to the [Seecology path](https://github.com/clausiusreis/Seecology/tree/master/Seecology) and run the command: python ./Seecology.py. This will open the Seecology UI in a new browser window.

## Tutorial
Currently in development, will be available shortly. 

## Acknowledgment
We like to acknowledge the financial support of the São Paulo State Research Foundation (FAPESP grants 2017/05838-3 and 2016/02175-0) and the National Council for Scientific and Technological Development (CNPq grant 301847/2017-7). The views expressed do not reflect the official policy or position of either FAPESP or CNPq.
