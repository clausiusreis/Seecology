#!/usr/bin/env python2.7
# coding: utf-8

#Examle of date stamp: %Y.%m.%d_%H.%M.%S

import json
import os
import subprocess
import sys
import shutil
import numpy as np
from flask import Flask, request, render_template, jsonify
import random, threading, webbrowser
import pandas as pd
import Tkinter, tkFileDialog
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import librosa
from librosa import display
import soundfile as sf
#import pywt
from scipy.signal import spectrogram
import scipy.misc
import csv
import copy
import time
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
from scipy import spatial
from scipy import signal
from datetime import datetime

# External http server to provide access to files on the chosen path
from threading import Thread
import http.server
import socketserver

### External functions
sys.path.append("scripts")

### Available feature functions
import Features

### Unify and normalize the individual files
#import UnifiedCSVGenerator as UG

app = Flask('clausius')

print("### "+app.instance_path)

# For the progress bar
global currentPath
global numCurrentFile
global numCurrentSpectrogram
global numFiles

global SDThreshold
global plotImages
global extractSpectrograms1
global extractMP31
global flags
global serverport

global teste
#teste = ""

flags = []
currentPath = ""
numCurrentFile = 1
numCurrentSpectrogram = 1
numFiles = 0
SDThreshold = 1.5
plotImages = True
extractSpectrograms = False
#initialDirectory="~/Documents/DOUTORADO/TEST_DATABASE"
#initialDirectory="/home/clausius/public_html/DOUTORADO_Visualizations_Framework/testData"
initialDirectory="/home/clausius/Documents/DOUTORADO/TEST_DATABASE/teste3"

# Start page
@app.route("/", methods=['GET', 'POST'])
def index_page():
    return render_template("page1.html", currentPath = currentPath, flags=flags)

# Feature extraction page
@app.route("/featureextraction", methods=['GET', 'POST'])
def featureExtraction_page():
    return render_template("page2.html", currentPath = currentPath)
    
# Feature extraction boat page
@app.route("/featureextractionboat", methods=['GET', 'POST'])
def featureExtractionboat_page():
    return render_template("page2-1.html", currentPath = currentPath)

# STRFTIME tool
@app.route("/strftime", methods=['GET', 'POST'])
def strftime_page():
    return render_template("strftime.html", currentPath = currentPath)

# Available visualizations page
@app.route("/datavisualization", methods=['GET', 'POST'])
def datavisualization_page():
    return render_template("page3.html", currentPath = currentPath)

@app.route("/vis1", methods=['GET', 'POST'])
def vis1_page():
    return render_template("vis1.html", currentPath = currentPath, port=serverport+10)
    
@app.route("/vis2", methods=['GET', 'POST'])
def vis2_page():
    return render_template("vis2.html", currentPath = currentPath, port=serverport+10)    
    
@app.route("/vis3", methods=['GET', 'POST'])
def vis3_page():
    return render_template("vis3.html", currentPath = currentPath, port=serverport+10)    

@app.route("/vis4", methods=['GET', 'POST'])
def vis4_page():
    return render_template("vis4.html", currentPath = currentPath, port=serverport+10)    
    
@app.route("/vis5", methods=['GET', 'POST'])
def vis5_page():
    return render_template("vis5.html", currentPath = currentPath, port=serverport+10)    
    
@app.route("/vis6", methods=['GET', 'POST'])
def vis6_page():
    return render_template("vis6.html", currentPath = currentPath, port=serverport+10)    
    
@app.route("/vis7", methods=['GET', 'POST'])
def vis7_page():
    return render_template("vis7.html", currentPath = currentPath, port=serverport+10)    
    
@app.route("/vis8", methods=['GET', 'POST'])
def vis8_page():
    return render_template("vis8.html", currentPath = currentPath, port=serverport+10)    
    
@app.route("/vis9", methods=['GET', 'POST'])
def vis9_page():
    return render_template("vis9.html", currentPath = currentPath, port=serverport+10)        
    

@app.route("/get_current_path", methods=['GET', 'POST'])
def getCurrentPath():
    global currentPath
    global flags
    global serverport
    
    root = Tkinter.Tk()
    currentPath = tkFileDialog.askdirectory(initialdir=initialDirectory,title='Please select a directory with audio recordings')    

    if (os.path.isfile("/".join([currentPath, "Extraction", 'flags_extraction.csv']))):
        flags, headers = loadFlags(currentPath);    

    root.destroy()
    root.mainloop()

    threaded = Thread(target=servidor, args=()).start()

    seecologyPath = os.getcwd()
    os.chdir(seecologyPath)

    return jsonify({"currentPath": currentPath, "status": "Working directory selected", "flags": flags})

################################################################################################################
### HTTP SERVER - Start ########################################################################################
################################################################################################################
class HTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        http.server.SimpleHTTPRequestHandler.end_headers(self)

def server(port):
    httpd = socketserver.TCPServer(('', port), HTTPRequestHandler)
    return httpd

def servidor():    
    global serverport
    global currentPath

    os.chdir(currentPath)
    
    httpd = server(serverport+10)
    try:        
        os.chdir(currentPath)
        print("\nServing at localhost:" + str(serverport+10))
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n...shutting down http server")
        httpd.shutdown()
        sys.exit()    
    
#    Handler = http.server.SimpleHTTPRequestHandler
#    
#    self.send_header('Access-Control-Allow-Origin', '*')    
#    Handler.end_headers(self)
#
#    httpd = socketserver.TCPServer(("", serverport+10), Handler)
#    print("serving at port", serverport+10, currentPath)
#    httpd.serve_forever()

################################################################################################################
### HTTP SERVER - Start ########################################################################################
################################################################################################################


def removeOutliersFromData(data):
    l = data.shape[0];
    c = data.shape[1];    
        
    # Only show data within 3 standard deviations of the mean. 
    # If data is normally distributed 99.7% will fall in this range.        
    for j in range(1, c):
        auxsum = 0; # stores sum of elements
        auxsumsq = 0; # stores sum of squares
        
        for i in range(0, l):
            auxsum += data[data.columns[j]][i];
            auxsumsq += data[data.columns[j]][i] * data[data.columns[j]][i];
    
        mean = auxsum / l;
        variance = auxsumsq / l - mean * mean;
        sd = np.sqrt(variance);
        
        for i in range(0, l):
            if ( (data[data.columns[j]][i] > (mean + 3 * sd)) | (data[data.columns[j]][i] < (mean - 3 * sd)) ):
                data[data.columns[j]][i] = mean;

    return data;

def unifyCSV(mainPath, 
             resultingCSVName="features", 
             groupingOperation="none", 
             normalizationOperation="n4",
             removeOutliers = False):
    #mainPath: Directory where the audio files are stored
    #resultingCSVName: Name of the resulting CSV file
    #groupingOperation: Operation over each file - none, mean, median
    #normalizationOperation: type of normalization: 
    #    n0 - without normalization
    #    n1 - standardization ((x-mean)/sd)
    #    n2 - positional standardization ((x-median)/mad)
    #    n3 - unitization ((x-mean)/range)
    #    n3a - positional unitization ((x-median)/range)
    #    n4 - unitization with zero minimum ((x-min)/range)
    #    n5 - normalization in range <-1,1> ((x-mean)/max(abs(x-mean)))
    #    n5a - positional normalization in range <-1,1> ((x-median)/max(abs(x-median)))
    #    n6 - quotient transformation (x/sd)
    #    n6a - positional quotient transformation (x/mad)
    #    n7 - quotient transformation (x/range)
    #    n8 - quotient transformation (x/max)
    #    n9 - quotient transformation (x/mean)
    #    n9a - positional quotient transformation (x/median)
    #    n10 - quotient transformation (x/sum)
    #    n12 - normalization ((x-mean)/sqrt(sum((x-mean)^2)))
    #    n12a - positional normalization ((x-median)/sqrt(sum((x-median)^2)))
    
    wdir = os.listdir("%s/Extraction/AudioFeatures" % (mainPath))
    wdir.sort()
    csvFiles = [arq for arq in wdir if (arq.lower().endswith(".csv"))]
    csvFiles.sort()
    
    print("#### GENERATING UNIFIED CSV ####")
    print "Processing directory: %s" % mainPath

    firstHeader = 0
    resultingFile = '%s/Extraction/%s.csv' % (mainPath, resultingCSVName)
    with open(resultingFile, 'wb') as outputCsvfile:
        csvwriter = csv.writer(outputCsvfile, delimiter=',')
    
        for f in csvFiles:
            print "   Processing file: %s" % f
            with open("%s/Extraction/AudioFeatures/%s" % (mainPath, f), 'rb') as ff:
                reader = csv.reader(ff)
    
                currentLine = 0
                for row in reader:
                    if (firstHeader == 0):
                        csvwriter.writerow(row)
                        firstHeader = 1
                        currentLine = currentLine + 1
                    else:
                        if (currentLine > 0):
                            csvwriter.writerow(row)
                        currentLine = currentLine + 1

    print("#### UNIFIED CSV GENERATED ####")

    print("#### NORMALIZING CSV ####")
    print "Processing file: %s" % (resultingFile)
    
    #load original CSV
    df = pd.read_csv(resultingFile)

    ### DATA GROUPING ###################################################################################
    if (groupingOperation == "mean"):
        result = df.groupby(['FileName', 'Date', 'Time']).mean()
        result = pd.DataFrame(result)
        result.to_csv('%s_group.csv' % (resultingFile[:-4]), sep=',')

        if (result.shape[0] == 0):        
            result = df.groupby(['FileName']).mean()
            result = pd.DataFrame(result)
            result.to_csv('%s_group.csv' % (resultingFile[:-4]), sep=',')

    if (groupingOperation == "median"):
        result = df.groupby(['FileName', 'Date', 'Time']).median()
        result = pd.DataFrame(result)
        result.to_csv('%s_group.csv' % (resultingFile[:-4]), sep=',')
        
        if (result.shape[0] == 0):
            result = df.groupby(['FileName']).median()
            result = pd.DataFrame(result)
            result.to_csv('%s_group.csv' % (resultingFile[:-4]), sep=',')

    if (groupingOperation == "none"):
        result = df.groupby(['FileName', 'Date', 'Time', 'SecondsFromStart']).mean()
        result = pd.DataFrame(result)
        result.to_csv('%s_group.csv' % (resultingFile[:-4]), sep=',')
        
        if (result.shape[0] == 0):
            result = df.groupby(['FileName', 'SecondsFromStart']).mean()
            result = pd.DataFrame(result)
            result.to_csv('%s_group.csv' % (resultingFile[:-4]), sep=',')

    ### REMOVE OUTLIERS #################################################################################
    if (removeOutliers):
        result = removeOutliersFromData(result)    

    ### NORMALIZATION ###################################################################################
    #def normalize(df):
    #    result = df.copy()
    #    for feature_name in df.columns:
    #        max_value = df[feature_name].max()
    #        min_value = df[feature_name].min()
    #        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    #    return result
    
    # n1 - standardization ((x-mean)/sd)
    if (normalizationOperation == 'n1'):    
        normalized_df=(result-result.mean())/result.std()
        result = pd.DataFrame(normalized_df)
        result.to_csv('%s_norm.csv' % (resultingFile[:-4]), sep=',')
    
    # n2 - positional standardization ((x-median)/mad)
    if (normalizationOperation == 'n2'):
        normalized_df=(result-result.median())/(result.mad())
        result = pd.DataFrame(normalized_df)
        result.to_csv('%s_norm.csv' % (resultingFile[:-4]), sep=',')
        
    # n3 - unitization ((x-mean)/range)
    if (normalizationOperation == 'n3'):    
        normalized_df=(result-result.mean())/(result.max()-result.min())
        result = pd.DataFrame(normalized_df)
        result.to_csv('%s_norm.csv' % (resultingFile[:-4]), sep=',')
    
    # n3a - positional unitization ((x-median)/range)
    if (normalizationOperation == 'n3a'):    
        normalized_df=(result-result.median())/(result.max()-result.min())
        result = pd.DataFrame(normalized_df)
        result.to_csv('%s_norm.csv' % (resultingFile[:-4]), sep=',')
    
    # n4 - unitization with zero minimum ((x-min)/range)
    if (normalizationOperation == 'n4'):    
        normalized_df=(result-result.min())/(result.max()-result.min())
        result = pd.DataFrame(normalized_df)
        result.to_csv('%s_norm.csv' % (resultingFile[:-4]), sep=',')
    
    # n5 - normalization in range <-1,1> ((x-mean)/max(abs(x-mean)))
    if (normalizationOperation == 'n5'):    
        normalized_df=(result-result.mean())/(((result-result.mean()).abs()).max())
        result = pd.DataFrame(normalized_df)
        result.to_csv('%s_norm.csv' % (resultingFile[:-4]), sep=',')
    
    # n5a - positional normalization in range <-1,1> ((x-median)/max(abs(x-median)))
    if (normalizationOperation == 'n5a'):    
        normalized_df=(result-result.median())/(((result-result.median()).abs()).max())
        result = pd.DataFrame(normalized_df)
        result.to_csv('%s_norm.csv' % (resultingFile[:-4]), sep=',')
    
    # n6 - quotient transformation (x/sd)
    if (normalizationOperation == 'n6'):    
        normalized_df=(result)/(result.std())
        result = pd.DataFrame(normalized_df)
        result.to_csv('%s_norm.csv' % (resultingFile[:-4]), sep=',')
    
    # n6a - positional quotient transformation (x/mad)
    if (normalizationOperation == 'n6a'):    
        normalized_df=(result)/(result.mad())
        result = pd.DataFrame(normalized_df)
        result.to_csv('%s_norm.csv' % (resultingFile[:-4]), sep=',')
    
    # n7 - quotient transformation (x/range)
    if (normalizationOperation == 'n7'):    
        normalized_df=(result)/(result.max()-result.min())
        result = pd.DataFrame(normalized_df)
        result.to_csv('%s_norm.csv' % (resultingFile[:-4]), sep=',')
    
    # n8 - quotient transformation (x/max)
    if (normalizationOperation == 'n8'):    
        normalized_df=(result)/(result.max())
        result = pd.DataFrame(normalized_df)
        result.to_csv('%s_norm.csv' % (resultingFile[:-4]), sep=',')
    
    # n9 - quotient transformation (x/mean)
    if (normalizationOperation == 'n9'):    
        normalized_df=(result)/(result.mean())
        result = pd.DataFrame(normalized_df)
        result.to_csv('%s_norm.csv' % (resultingFile[:-4]), sep=',')
    
    # n9a - positional quotient transformation (x/median)
    if (normalizationOperation == 'n9a'):    
        normalized_df=(result)/(result.median())
        result = pd.DataFrame(normalized_df)
        result.to_csv('%s_norm.csv' % (resultingFile[:-4]), sep=',')
    
    # n10 - quotient transformation (x/sum)
    if (normalizationOperation == 'n10'):    
        normalized_df=(result)/(result.sum())
        result = pd.DataFrame(normalized_df)
        result.to_csv('%s_norm.csv' % (resultingFile[:-4]), sep=',')
        
    # n12 - normalization ((x-mean)/sqrt(sum((x-mean)^2)))
    if (normalizationOperation == 'n12'):    
        normalized_df=(result-result.mean())/(((result-result.mean()) ** 2).sum()).apply(np.sqrt)
        result = pd.DataFrame(normalized_df)
        result.to_csv('%s_norm.csv' % (resultingFile[:-4]), sep=',')
    
    # n12a - positional normalization ((x-median)/sqrt(sum((x-median)^2)))
    if (normalizationOperation == 'n12a'):    
        normalized_df=(result-result.median())/(((result-result.median()) ** 2).sum()).apply(np.sqrt)
        result = pd.DataFrame(normalized_df)
        result.to_csv('%s_norm.csv' % (resultingFile[:-4]), sep=',')
        
    print("#### PROCESSED CSV GENERATED ####")

def groupFeatures(feat, fileIdent, fileDateTimeMask, GENERAL_timeWindow):
    #firstCol = False

    global resACI
    global resADI
    global resAEI
    global resBIO
    global resMFCC 
    global resNDSI 
    global resNDSI_Anthro
    global resNDSI_Bio
    global resRMS 
    global resTH
    global resZCR
    global resSpecPropMean
    global resSpecPropSD
    global resSpecPropSEM
    global resSpecPropMedian
    global resSpecPropMode
    global resSpecPropQ25
    global resSpecPropQ50
    global resSpecPropQ75
    global resSpecPropIQR
    global resSpecPropSkewness
    global resSpecPropKurtosis
    global resSpecPropEntropy
    global resSpecPropVariance
    global resSpecChromaSTFT
    global resSpecChromaCQT
    global resSpecCentroid
    global resSpecBandwidth
    #global resSpecContrast
    global resSpecRolloff
    global resSpecPolyFeatures_Zero
    global resSpecPolyFeatures_Linear
    global resSpecPolyFeatures_Quadratic
    global resSpread
    global resSPL
    
    resultMat = {}    
    selectedColNames = []

    if ('ACI' in feat):
        resultMat['ACI'] = resACI 
        selectedColNames = selectedColNames + ['ACI']

    if ('ADI' in feat):
        resultMat['ADI'] = resADI 
        selectedColNames = selectedColNames + ['ADI']

    if ('AEI' in feat):
        resultMat['AEI'] = resAEI 
        selectedColNames = selectedColNames + ['AEI']

    if ('BIO' in feat):
        resultMat['BIO'] = resBIO 
        selectedColNames = selectedColNames + ['BIO']
        
    if ('MFCC' in feat):
        for i in range(0,resMFCC.shape[1]):
            resultMat['MFCC_'+str(i)] = resMFCC[:,i]
        selectedColNames = selectedColNames + ['MFCC_'+str(i) for i in range(resMFCC.shape[1])]

    if ('NDSI' in feat):
        resultMat['NDSI'] = resNDSI
        resultMat['NDSI_Anthro'] = resNDSI_Anthro
        resultMat['NDSI_Bio'] = resNDSI_Bio
        selectedColNames = selectedColNames + ['NDSI', 'NDSI_Anthro', 'NDSI_Bio']
    
    if ('RMS' in feat):
        resultMat['RMS'] = resRMS 
        selectedColNames = selectedColNames + ['RMS']
    
    if ('TH' in feat):
        resultMat['TH'] = resTH 
        selectedColNames = selectedColNames + ['TH']
    
    if ('ZCR' in feat):
        resultMat['ZCR'] = resZCR 
        selectedColNames = selectedColNames + ['ZCR']
    
    if ('SpecMean' in feat):
        resultMat['SpecMean'] = resSpecPropMean 
        selectedColNames = selectedColNames + ['SpecMean']
    
    if ('SpecSD' in feat):
        resultMat['SpecSD'] = resSpecPropSD
        selectedColNames = selectedColNames + ['SpecSD']
    
    if ('SpecSEM' in feat):
        resultMat['SpecSEM'] = resSpecPropSEM
        selectedColNames = selectedColNames + ['SpecSEM']
    
    if ('SpecMedian' in feat):
        resultMat['SpecMedian'] = resSpecPropMedian 
        selectedColNames = selectedColNames + ['SpecMedian']

    if ('SpecMode' in feat):
        resultMat['SpecMode'] = resSpecPropMode 
        selectedColNames = selectedColNames + ['SpecMode']
    
    if ('SpecQuartile' in feat):
        resultMat['SpecQuartile25'] = resSpecPropQ25
        resultMat['SpecQuartile50'] = resSpecPropQ50
        resultMat['SpecQuartile75'] = resSpecPropQ75
        resultMat['SpecQuartileIQR'] = resSpecPropIQR
        selectedColNames = selectedColNames + ['SpecQuartile25', 'SpecQuartile50', 'SpecQuartile75', 'SpecQuartileIQR']

    if ('SpecSkewness' in feat):
        resultMat['SpecSkewness'] = resSpecPropSkewness
        selectedColNames = selectedColNames + ['SpecSkewness']
    
    if ('SpecKurtosis' in feat):
        resultMat['SpecKurtosis'] = resSpecPropKurtosis
        selectedColNames = selectedColNames + ['SpecKurtosis']
    
    if ('SpecEntropy' in feat):
        resultMat['SpecEntropy'] = resSpecPropEntropy
        selectedColNames = selectedColNames + ['SpecEntropy']
    
    if ('SpecVariance' in feat):
        resultMat['SpecVariance'] = resSpecPropVariance
        selectedColNames = selectedColNames + ['SpecVariance']
    
    if ('SpecChromaSTFT' in feat):
        resultMat['SpecChromaSTFT'] = resSpecChromaSTFT
        selectedColNames = selectedColNames + ['SpecChromaSTFT']
    
    if ('SpecChromaCQT' in feat):
        resultMat['SpecChromaCQT'] = resSpecChromaCQT
        selectedColNames = selectedColNames + ['SpecChromaCQT']
    
    if ('SpecCentroid' in feat):
        resultMat['SpecCentroid'] = resSpecCentroid
        selectedColNames = selectedColNames + ['SpecCentroid']
    
    if ('SpecBandwidth' in feat):
        resultMat['SpecBandwidth'] = resSpecBandwidth
        selectedColNames = selectedColNames + ['SpecBandwidth']
    
    #if ('SpecContrast' in feat):
    #    resultMat['SpecContrast'] = resSpecContrast
    #    selectedColNames = selectedColNames + ['SpecContrast']
    
    if ('SpecRolloff' in feat):
        resultMat['SpecRolloff'] = resSpecRolloff
        selectedColNames = selectedColNames + ['SpecRolloff']
    
    if ('SpecPolyFeat' in feat):
        resultMat['SpecPolyFZero'] = resSpecPolyFeatures_Zero
        resultMat['SpecPolyFLinear'] = resSpecPolyFeatures_Linear
        resultMat['SpecPolyFQuadratic'] = resSpecPolyFeatures_Quadratic
        selectedColNames = selectedColNames + ['SpecPolyFZero', 'SpecPolyFLinear', 'SpecPolyFQuadratic']
    
    if ('SpecSpread' in feat):
        resultMat['SpecSpread'] = resSpread
        selectedColNames = selectedColNames + ['SpecSpread']
    
    if ('SPL' in feat):
        resultMat['SPL'] = resSPL
        selectedColNames = selectedColNames + ['SPL']
    
    ### Create first collumns of CSV #############################################################################        
    if (fileDateTimeMask != ""):
        fileDateTime = datetime.strptime(fileIdent, fileDateTimeMask)
        fileDate = fileDateTime.strftime('%Y-%m-%d')
        fileTime = fileDateTime.strftime('%H:%M:%S')
    else:
        fileDate = ""
        fileTime = ""

    firstCols = ["FileName","Date","Time","SecondsFromStart"]
    selectedColNames = firstCols + selectedColNames
    col0 = [fileIdent for i in range( len(resultMat[selectedColNames[4]]) )]
    col1 = [fileDate for i in range( len(resultMat[selectedColNames[4]]) )]
    col2 = [fileTime for i in range( len(resultMat[selectedColNames[4]]) )]
    col3 = [i*GENERAL_timeWindow for i in range( len(resultMat[selectedColNames[4]]) )]
    
    resultMat['FileName'] = col0
    resultMat['Date'] = col1
    resultMat['Time'] = col2
    resultMat['SecondsFromStart'] = col3
    
    return resultMat, selectedColNames

###########################################################################################################
### Calculate the spectrum of samples
###########################################################################################################
def calculate_spectrum(recording, sample_rate):
    fft = np.fft.fft(recording, norm=None)    

    return abs(fft[:sample_rate/2])

def extractionProcess(jsonData, dbname):
    global numCurrentFile
    global numCurrentSpectrogram
    global numFiles      
    global extractSpectrograms1
    global extractMP31
    global extractSpectrograms
    global extractMP3
    global teste

    teste = jsonData
    
    print("#### EXTRACTION PROCESS BEGIN ####")
    print "Processing directory: %s" % jsonData['GENERAL_mainPath']

    #REMOVER TODOS OS ARQUIVOS ANTERIORES

    ### Crio diretórios para armazenar o resultado ###############################################################
    if not os.path.exists("%s/Extraction" % (jsonData['GENERAL_mainPath'])):
        os.makedirs("%s/Extraction" % (jsonData['GENERAL_mainPath']))    
    
    if not os.path.exists("%s/Extraction/AudioFeatures" % (jsonData['GENERAL_mainPath'])):
        os.makedirs("%s/Extraction/AudioFeatures" % (jsonData['GENERAL_mainPath']))
    else:
        shutil.rmtree( "%s/Extraction/AudioFeatures" % (jsonData['GENERAL_mainPath']) )
        time.sleep(2)
        os.makedirs("%s/Extraction/AudioFeatures" % (jsonData['GENERAL_mainPath']))

    if not os.path.exists("%s/Extraction/AudioMP3" % (jsonData['GENERAL_mainPath'])):
        os.makedirs("%s/Extraction/AudioMP3" % (jsonData['GENERAL_mainPath']))

    if not os.path.exists("%s/Extraction/AudioSpectrogram" % (jsonData['GENERAL_mainPath'])):
        os.makedirs("%s/Extraction/AudioSpectrogram" % (jsonData['GENERAL_mainPath']))

    # Reset the extraction error report
    with open("/".join([jsonData['GENERAL_mainPath'], "Extraction/PROCESSING_ERROR_FeatureExtraction.log"]), "w") as myfile:
        myfile.write("####################################################\n")
        myfile.write("### ERROR REPORT - Seecology                     ###\n")
        myfile.write("####################################################\n")    

    wdir = os.listdir(jsonData['GENERAL_mainPath'])
    wdir.sort()
    audios = [arq for arq in wdir if (arq.lower().endswith(".flac") | arq.lower().endswith(".mp3") | arq.lower().endswith(".wav"))]
    audios.sort()
    
    for f in audios:           
        fileExt = {"flac", ".mp3", ".wav"}
    
        global audioType
        audioType = f[-4:]
        
        if (f[-4:] in fileExt): 
            ##Check the extension MP3 need to be extracted
            if (f[-4:].lower() == '.mp3'):
                print "    Extracting MP3 file to WAV: %s" % ( f )
                currentFileMP3 = "%s/%s" % (jsonData['GENERAL_mainPath'], f)                                
                os.system("yes | ffmpeg -i %s %s.wav" % (currentFileMP3, currentFileMP3[:-4]))
    
    ##############################################################################################################
    ### BEGIN OF FILES LOOP ######################################################################################
    ##############################################################################################################
    wdir = os.listdir(jsonData['GENERAL_mainPath'])
    #wdir.sort()
    audios = [arq for arq in wdir if (arq.lower().endswith(".flac") | arq.lower().endswith(".wav"))]
    audios.sort()
    
    numFiles = len(audios)

    # Append the boat extraction errors
    with open("/".join([jsonData['GENERAL_mainPath'], "Extraction/PROCESSING_ERROR_FeatureExtraction.log"]), "a") as myfile:
        myfile.write("\n")    
        myfile.write("### ERROR ON THE EXTRACTION PROCESS ################\n")    
    
    sumTimeFiles = 0
    totAudios = 0
    totSamples = 0 
    for f in audios:
        try:
            if numCurrentFile <= len(audios):
                print("   Processing file %s of %s (%s)" % (numCurrentFile+1, len(audios), f))                
    
            # Teste de tempo        
            tt0 = time.time()
            
            totAudios = totAudios + 1
     
            currentFileWAV = "%s/%s" % (jsonData['GENERAL_mainPath'], f)
            if f[-4:] == 'flac':
                fileIdent = f[:-5]
            else:
                fileIdent = f[:-4]
            currentFile = currentFileWAV
    
            ##############################################################################################################
            ### Convert the files to MP3 for the player ##################################################################
            ##############################################################################################################
            if extractMP3 == True:
                if not os.path.exists("%s/Extraction/AudioMP3/%s.mp3" % (jsonData['GENERAL_mainPath'], fileIdent)):
                    os.system("yes | ffmpeg -i %s -vn -f mp3 %s/Extraction/AudioMP3/%s.mp3" % (currentFile, jsonData['GENERAL_mainPath'], fileIdent))
    
            ##############################################################################################################
            ### Chamada de função para extração dos features #############################################################
            ##############################################################################################################
            #Load an audio file with separated samplerate, left and right channels            
            #rate, frames = wavfile.read(currentFile)    
            frames, rate = sf.read(currentFile)
             
            # Pego um canal apenas
            if (len(np.shape(frames)) == 1):
                selChannel = frames
            else:
                if (jsonData['GENERAL_audioChannel'] == 0):
                    selChannel = frames[:,0]
                else:
                    selChannel = frames[:,1]
    
            ##############################################################################################################
            ### CREATE AN AUDIO SPECTROGRAM OF THE ENTIRE FILE ###########################################################
            ##############################################################################################################
            #if not os.path.exists("%s/Extraction/AudioSpectrogram/%s.png" % (jsonData['GENERAL_mainPath'], fileIdent)):
            #    spec_width = 600
            #    spec_height = 270
            #    speccmap = plt.cm.jet
            #    spectro = librosa.stft(selChannel, n_fft=2048, window=scipy.signal.blackmanharris)
            #    specMatrixDB = librosa.amplitude_to_db(spectro, ref=np.max, top_db=jsonData['GENERAL_dbThreshold'])
            #    specMatrixDB1 = np.flipud(specMatrixDB)
            #    norm = plt.Normalize(vmin=specMatrixDB1.min(), vmax=specMatrixDB1.max())
            #    image = speccmap(norm(specMatrixDB1))        
            #    newImage = scipy.misc.imresize(image, (spec_height, spec_width))
            #    plt.imsave("%s/Extraction/AudioSpectrogram/%s.png" % (jsonData['GENERAL_mainPath'], fileIdent), newImage)
    
            ##############################################################################################################
            # NEW SPECTROGRAM
            # Extract the spectrogram matrix (STFT) for each audio second
            #sampleWindows = len(selChannel)/rate
            #specMatrix = np.zeros([rate/2, sampleWindows])
            #for i in range(sampleWindows):
            #    t1 = i*rate
            #    t2 = (i+1)*rate
            #    specMatrix[:,i] = calculate_spectrum(selChannel[t1:t2], rate)
    
            # Add the mean of the spectrogram for each time sample defined by the user to a new matrix
    
            # CREATE AN AUDIO SPECTROGRAM OF THE ENTIRE FILE (Pass the new matrix instead of this one)
            #spec_width = 1200
            #spec_height = 600
            #speccmap = plt.cm.jet
            #specMatrixDB = librosa.amplitude_to_db(specMatrix[:maxFreq,:], ref=1.0, top_db=topDB)
            #specMatrixDB1 = np.flipud(specMatrixDB)
            #norm = plt.Normalize(vmin=specMatrixDB1.min(), vmax=specMatrixDB1.max())    
            #image = speccmap(norm(specMatrixDB1))        
            #newImage = scipy.misc.imresize(image, (spec_height, spec_width))
            #plt.imsave("%s/Extraction/AudioSpectrogram/%s.png" % (currentPath, fileIdent), newImage)
            ##############################################################################################################
    
            ##############################################################################################################
            ### BEGIN OF TIME WINDOW LOOP ################################################################################
            ##############################################################################################################
            sampleWindow = jsonData['GENERAL_timeWindow'] * rate
            numSamples = len(selChannel)/sampleWindow
            audioSeconds = int(len(frames)/rate)
            
            #TODO: Gerar csv com info do áudio
            print("   ### Printing audio info")
            with open("/".join([jsonData['GENERAL_mainPath'], "Extraction/AudioMP3", fileIdent+'_info.csv']), 'w') as outfileinfo:            
                outfileinfo.write("maxFreq,sampleWindow,audioSeconds\n")
                outfileinfo.write("%s,%s,%s" % (rate/2, jsonData['GENERAL_timeWindow'], int(len(frames)/rate)))         
                    
            ### Define each as global to group them at the end
            global result
            global resACI
            global resADI
            global resAEI
            global resBIO
            global resMFCC 
            global resNDSI 
            global resNDSI_Anthro
            global resNDSI_Bio
            global resRMS 
            global resTH
            global resZCR
            global resSpecPropMean
            global resSpecPropSD
            global resSpecPropSEM
            global resSpecPropMedian
            global resSpecPropMode
            global resSpecPropQ25
            global resSpecPropQ50
            global resSpecPropQ75
            global resSpecPropIQR
            global resSpecPropSkewness
            global resSpecPropKurtosis
            global resSpecPropEntropy
            global resSpecPropVariance
            global resSpecChromaSTFT
            global resSpecChromaCQT
            global resSpecCentroid
            global resSpecBandwidth
            #global resSpecContrast
            global resSpecRolloff
            global resSpecPolyFeatures_Zero
            global resSpecPolyFeatures_Linear
            global resSpecPolyFeatures_Quadratic
            global resSpread
            global resSPL
             
            result = np.zeros(numSamples)        
            resACI = np.zeros(numSamples)
            resADI = np.zeros(numSamples)
            resAEI = np.zeros(numSamples)
            resBIO = np.zeros(numSamples)
            resMFCC = np.zeros(((numSamples,jsonData['MFCC_numcep'])))
            resNDSI = np.zeros(numSamples)
            resNDSI_Anthro = np.zeros(numSamples)
            resNDSI_Bio = np.zeros(numSamples)
            resRMS = np.zeros(numSamples)
            resTH = np.zeros(numSamples)
            resZCR = np.zeros(numSamples)
            resSpecPropMean = np.zeros(numSamples)
            resSpecPropSD = np.zeros(numSamples)
            resSpecPropSEM = np.zeros(numSamples)
            resSpecPropMedian = np.zeros(numSamples)
            resSpecPropMode = np.zeros(numSamples)
            resSpecPropQ25 = np.zeros(numSamples)
            resSpecPropQ50 = np.zeros(numSamples)
            resSpecPropQ75 = np.zeros(numSamples)
            resSpecPropIQR = np.zeros(numSamples)
            resSpecPropSkewness = np.zeros(numSamples)
            resSpecPropKurtosis = np.zeros(numSamples)
            resSpecPropEntropy = np.zeros(numSamples)
            resSpecPropVariance = np.zeros(numSamples)
            resSpecChromaSTFT = np.zeros(numSamples)
            resSpecChromaCQT = np.zeros(numSamples)
            resSpecCentroid = np.zeros(numSamples)
            resSpecBandwidth = np.zeros(numSamples)
            #resSpecContrast = np.zeros(numSamples)
            resSpecRolloff = np.zeros(numSamples)
            resSpecPolyFeatures_Zero = np.zeros(numSamples)
            resSpecPolyFeatures_Linear = np.zeros(numSamples)
            resSpecPolyFeatures_Quadratic = np.zeros(numSamples)
            resSpread = np.zeros(numSamples)
            resSPL = np.zeros(numSamples)
    
            for si in range(numSamples):            
                totSamples = totSamples + 1
                
                # Set the time window to extract audio sample    
                time0 = (si)*sampleWindow
                time1 = (si+1)*sampleWindow
                if time1 > len(selChannel):
                    time1 = len(selChannel)-1
             
                sample  = selChannel[time0:time1]
                 
                sampleDB = {}
                samplePWR = {}
                for ff in jsonData['GENERAL_featuresToCalculate']:                    
                    try:
                        if jsonData[ff+'_fft_w'] not in sampleDB:
                            spectro = librosa.stft( sample, n_fft=jsonData[ff+'_fft_w'] )
                            specDB  = librosa.amplitude_to_db(spectro, ref=np.max, top_db=jsonData['GENERAL_dbThreshold'])
                            specPWR = librosa.db_to_power(specDB, ref=1.0)
                            
                            #TODO: Confirmar esta chamada 
                            sampleDB[jsonData[ff+'_fft_w']]  = specDB
                            samplePWR[jsonData[ff+'_fft_w']] = specPWR
                    except:
                        pass
                
                ### Acoustic Complesity ######################################################################################            
                if ('ACI' in jsonData['GENERAL_featuresToCalculate']):
                    resACI[si] = Features.compute_ACI(
                                sample          = samplePWR[jsonData['ACI_fft_w']],
                                rate            = rate,
                                timeWindow      = jsonData['GENERAL_timeWindow'],
                                min_freq        = jsonData['ACI_min_freq'], 
                                max_freq        = jsonData['ACI_max_freq'], 
                                j_bin           = jsonData['ACI_j_bin'], 
                                fft_w           = jsonData['ACI_fft_w'], 
                                db_threshold    = jsonData['GENERAL_dbThreshold'])
      
                ### Acoustic Diversity #######################################################################################    
                if ('ADI' in jsonData['GENERAL_featuresToCalculate']):
                    resADI[si] = Features.compute_ADI(
                                sample          = sampleDB[jsonData['ADI_fft_w']],
                                rate            = rate,
                                timeWindow      = jsonData['GENERAL_timeWindow'],                            
                                max_freq        = jsonData['ADI_max_freq'], 
                                freq_step       = jsonData['ADI_freq_step'], 
                                fft_w           = jsonData['ADI_fft_w'], 
                                db_threshold    = jsonData['GENERAL_dbThreshold'])
                  
                ### Acoustic Evenness ########################################################################################
                if ('AEI' in jsonData['GENERAL_featuresToCalculate']):
                    resAEI[si] = Features.compute_AEI(
                                sample          = sampleDB[jsonData['AEI_fft_w']],
                                rate            = rate,
                                timeWindow      = jsonData['GENERAL_timeWindow'],
                                max_freq        = jsonData['AEI_max_freq'], 
                                freq_step       = jsonData['AEI_freq_step'], 
                                fft_w           = jsonData['AEI_fft_w'],
                                db_threshold    = jsonData['GENERAL_dbThreshold'])
                  
                ### Bioacoustic Index ########################################################################################
                if ('BIO' in jsonData['GENERAL_featuresToCalculate']):
                    resBIO[si] = Features.compute_BIO(
                                sample          = sampleDB[jsonData['BIO_fft_w']], 
                                rate            = rate,
                                timeWindow      = jsonData['GENERAL_timeWindow'],
                                min_freq        = jsonData['BIO_min_freq'], 
                                max_freq        = jsonData['BIO_max_freq'], 
                                fft_w           = jsonData['BIO_fft_w'],
                                db_threshold    = jsonData['GENERAL_dbThreshold'])    
                  
                ### MFCC - Mel Frequency Cepstral Coefficients ###############################################################
                if ('MFCC' in jsonData['GENERAL_featuresToCalculate']):
                    resMFCC[si] = Features.compute_MFCC(
                            soundMono       = sample, 
                            rate            = rate, 
                            timeWindow      = jsonData['GENERAL_timeWindow'], 
                            numcep          = jsonData['MFCC_numcep'], 
                            min_freq        = jsonData['MFCC_min_freq'], 
                            max_freq        = jsonData['MFCC_max_freq'], 
                            useMel          = False if (jsonData['MFCC_useMel'] == "False") else True )
                   
                ### NDSI - Normalized Difference Soundscape Index ############################################################
                if ('NDSI' in jsonData['GENERAL_featuresToCalculate']):
                    resNDSI[si], resNDSI_Anthro[si], resNDSI_Bio[si] = Features.compute_NDSI(
                                soundMono       = sample,   
                                rate            = rate,
                                timeWindow      = jsonData['GENERAL_timeWindow'],
                                fft_w           = jsonData['NDSI_fft_w'],
                                anthrophony     = jsonData['NDSI_anthrophony'], 
                                biophony        = jsonData['NDSI_biophony'])
       
                ### RMS Energy ###############################################################################################
                if ('RMS' in jsonData['GENERAL_featuresToCalculate']):
                    resRMS[si] = Features.compute_RMS_Energy(
                            soundMono       = sample, 
                            rate            = rate, 
                            timeWindow      = jsonData['GENERAL_timeWindow'] )
                   
                ### Temporal Entropy #########################################################################################
                if ('TH' in jsonData['GENERAL_featuresToCalculate']):
                    resTH[si] = Features.compute_TH(
                            soundMono       = sample, 
                            rate            = rate, 
                            timeWindow      = jsonData['GENERAL_timeWindow'] )
                   
                ### Zero Crossing Rate #######################################################################################
                if ('ZCR' in jsonData['GENERAL_featuresToCalculate']):
                    resZCR[si] = Features.compute_ZCR(
                            soundMono       = sample, 
                            rate            = rate, 
                            timeWindow      = jsonData['GENERAL_timeWindow'] )    
               
                ### Spectral Mean ############################################################################################
                if ('SpecMean' in jsonData['GENERAL_featuresToCalculate']):
                    resSpecPropMean[si] = Features.compute_SpecProp_Mean(
                            sample          = samplePWR[jsonData['SpecMean_fft_w']], 
                            rate            = rate, 
                            timeWindow      = jsonData['GENERAL_timeWindow'], 
                            min_freq        = jsonData['SpecMean_min_freq'], 
                            max_freq        = jsonData['SpecMean_max_freq'], 
                            fft_w           = jsonData['SpecMean_fft_w'], 
                            db_threshold    = jsonData['GENERAL_dbThreshold'] )
               
                ### Spectral Standard Deviation ##############################################################################
                if ('SpecSD' in jsonData['GENERAL_featuresToCalculate']):
                    resSpecPropSD[si] = Features.compute_SpecProp_SD(
                            sample          = samplePWR[jsonData['SpecSD_fft_w']], 
                            rate            = rate, 
                            timeWindow      = jsonData['GENERAL_timeWindow'], 
                            min_freq        = jsonData['SpecSD_min_freq'], 
                            max_freq        = jsonData['SpecSD_max_freq'], 
                            fft_w           = jsonData['SpecSD_fft_w'],
                            db_threshold    = jsonData['GENERAL_dbThreshold'] ) 
               
                ### Spectral Standard Error Mean #############################################################################
                if ('SpecSEM' in jsonData['GENERAL_featuresToCalculate']):
                    resSpecPropSEM[si] = Features.compute_SpecProp_SEM(
                            sample          = samplePWR[jsonData['SpecSEM_fft_w']], 
                            rate            = rate, 
                            timeWindow      = jsonData['GENERAL_timeWindow'], 
                            min_freq        = jsonData['SpecSEM_min_freq'], 
                            max_freq        = jsonData['SpecSEM_max_freq'], 
                            fft_w           = jsonData['SpecSEM_fft_w'],
                            db_threshold    = jsonData['GENERAL_dbThreshold'])
        
                ### Spectral Median ##########################################################################################
                if ('SpecMedian' in jsonData['GENERAL_featuresToCalculate']):
                    resSpecPropMedian[si] = Features.compute_SpecProp_Median(
                            sample          = samplePWR[jsonData['SpecMedian_fft_w']], 
                            rate            = rate, 
                            timeWindow      = jsonData['GENERAL_timeWindow'], 
                            min_freq        = jsonData['SpecMedian_min_freq'], 
                            max_freq        = jsonData['SpecMedian_max_freq'], 
                            fft_w           = jsonData['SpecMedian_fft_w'], 
                            db_threshold    = jsonData['GENERAL_dbThreshold'])
                 
                ### Spectral Mode ############################################################################################
                if ('SpecMode' in jsonData['GENERAL_featuresToCalculate']):
                    resSpecPropMode[si] = Features.compute_SpecProp_Mode(
                            sample          = samplePWR[jsonData['SpecMode_fft_w']], 
                            rate            = rate, 
                            timeWindow      = jsonData['GENERAL_timeWindow'], 
                            min_freq        = jsonData['SpecMode_min_freq'], 
                            max_freq        = jsonData['SpecMode_max_freq'], 
                            fft_w           = jsonData['SpecMode_fft_w'], 
                            db_threshold    = jsonData['GENERAL_dbThreshold'])
               
                ### Spectral Quartile ########################################################################################
                if ('SpecQuartile' in jsonData['GENERAL_featuresToCalculate']):
                    resSpecPropQ25[si], resSpecPropQ50[si], resSpecPropQ75[si], resSpecPropIQR[si] = Features.compute_SpecProp_Quartiles(
                            sample          = samplePWR[jsonData['SpecQuartile_fft_w']], 
                            rate            = rate, 
                            timeWindow      = jsonData['GENERAL_timeWindow'], 
                            min_freq        = jsonData['SpecQuartile_min_freq'], 
                            max_freq        = jsonData['SpecQuartile_max_freq'], 
                            fft_w           = jsonData['SpecQuartile_fft_w'], 
                            db_threshold    = jsonData['GENERAL_dbThreshold'])
           
                ### Spectral Skewness ########################################################################################
                if ('SpecSkewness' in jsonData['GENERAL_featuresToCalculate']):
                       
                    if jsonData['SpecSkewness_power'] == True:
                        sampleChosen = samplePWR[jsonData['SpecSkewness_fft_w']]
                    else:
                        sampleChosen = sampleDB[jsonData['SpecSkewness_fft_w']]                    
                           
                    resSpecPropSkewness[si] = Features.compute_SpecProp_Skewness(
                            sample          = sampleChosen, 
                            rate            = rate, 
                            timeWindow      = jsonData['GENERAL_timeWindow'], 
                            min_freq        = jsonData['SpecSkewness_min_freq'], 
                            max_freq        = jsonData['SpecSkewness_max_freq'], 
                            fft_w           = jsonData['SpecSkewness_fft_w'], 
                            db_threshold    = jsonData['GENERAL_dbThreshold'], 
                            power           = False if (jsonData['SpecSkewness_power'] == "False") else True )
               
                ### Spectral Kurtosis ########################################################################################
                if ('SpecKurtosis' in jsonData['GENERAL_featuresToCalculate']):
                       
                    if jsonData['SpecKurtosis_power'] == True:
                        sampleChosen = samplePWR[jsonData['SpecKurtosis_fft_w']]
                    else:
                        sampleChosen = sampleDB[jsonData['SpecKurtosis_fft_w']]
    
                    resSpecPropKurtosis[si] = Features.compute_SpecProp_Kurtosis(
                            sample          = sampleChosen, 
                            rate            = rate, 
                            timeWindow      = jsonData['GENERAL_timeWindow'], 
                            min_freq        = jsonData['SpecKurtosis_min_freq'],
                            max_freq        = jsonData['SpecKurtosis_max_freq'], 
                            fft_w           = jsonData['SpecKurtosis_fft_w'], 
                            db_threshold    = jsonData['GENERAL_dbThreshold'], 
                            power           = False if (jsonData['SpecKurtosis_power'] == "False") else True )  
               
                ### Spectral Entropy #########################################################################################
                if ('SpecEntropy' in jsonData['GENERAL_featuresToCalculate']):
                       
                    if jsonData['SpecEntropy_power'] == True:
                        sampleChosen = samplePWR[jsonData['SpecEntropy_fft_w']]
                    else:
                        sampleChosen = sampleDB[jsonData['SpecEntropy_fft_w']]
       
                    resSpecPropEntropy[si] = Features.compute_SpecProp_Entropy(
                            sample          = sampleChosen, 
                            rate            = rate, 
                            timeWindow      = jsonData['GENERAL_timeWindow'], 
                            min_freq        = jsonData['SpecEntropy_min_freq'],
                            max_freq        = jsonData['SpecEntropy_max_freq'],
                            fft_w           = jsonData['SpecEntropy_fft_w'], 
                            db_threshold    = jsonData['GENERAL_dbThreshold'], 
                            power           = False if (jsonData['SpecEntropy_power'] == "False") else True ) 
               
                ### Spectral Variance #########################################################################################
                if ('SpecVariance' in jsonData['GENERAL_featuresToCalculate']):
                       
                    if jsonData['SpecVariance_power'] == True:
                        sampleChosen = samplePWR[jsonData['SpecVariance_fft_w']]
                    else:
                        sampleChosen = sampleDB[jsonData['SpecVariance_fft_w']]
                       
                    resSpecPropVariance[si] = Features.compute_SpecProp_Variance(
                            sample          = sampleChosen, 
                            rate            = rate, 
                            timeWindow      = jsonData['GENERAL_timeWindow'], 
                            min_freq        = jsonData['SpecVariance_min_freq'], 
                            max_freq        = jsonData['SpecVariance_max_freq'], 
                            fft_w           = jsonData['SpecVariance_fft_w'], 
                            db_threshold    = jsonData['GENERAL_dbThreshold'], 
                            power           = False if (jsonData['SpecVariance_power'] == "False") else True ) 
               
                ### Spectral Chroma STFT ######################################################################################
                if ('SpecChromaSTFT' in jsonData['GENERAL_featuresToCalculate']):
                       
                    if jsonData['SpecChromaSTFT_power'] == True:
                        sampleChosen = samplePWR[jsonData['SpecChromaSTFT_fft_w']]
                    else:
                        sampleChosen = sampleDB[jsonData['SpecChromaSTFT_fft_w']]
                       
                    resSpecChromaSTFT[si] = Features.compute_SpecProp_Chroma_STFT(
                            sample          = sampleChosen, 
                            rate            = rate, 
                            timeWindow      = jsonData['GENERAL_timeWindow'], 
                            min_freq        = jsonData['SpecChromaSTFT_min_freq'], 
                            max_freq        = jsonData['SpecChromaSTFT_max_freq'], 
                            fft_w           = jsonData['SpecChromaSTFT_fft_w'], 
                            db_threshold    = jsonData['GENERAL_dbThreshold'], 
                            power           = False if (jsonData['SpecChromaSTFT_power'] == "False") else True )
               
                ### Spectral Chroma CQT #######################################################################################
                if ('SpecChromaCQT' in jsonData['GENERAL_featuresToCalculate']):
                    resSpecChromaCQT[si] = Features.compute_SpecProp_Chroma_CQT(
                            soundMono       = sample, 
                            rate            = rate, 
                            timeWindow      = jsonData['GENERAL_timeWindow'])
               
                ### Spectral Centroid #########################################################################################
                if ('SpecCentroid' in jsonData['GENERAL_featuresToCalculate']):
                    resSpecCentroid[si] = Features.compute_SpecProp_Centroid(
                            soundMono       = sample, 
                            rate            = rate, 
                            timeWindow      = jsonData['GENERAL_timeWindow'], 
                            fft_w           = jsonData['SpecCentroid_fft_w'])
               
                ### Spectral Bandwidth ########################################################################################
                if ('SpecBandwidth' in jsonData['GENERAL_featuresToCalculate']):
                    resSpecBandwidth[si] = Features.compute_SpecProp_Bandwidth(
                            soundMono       = sample, 
                            rate            = rate, 
                            timeWindow      = jsonData['GENERAL_timeWindow'], 
                            fft_w           = jsonData['SpecBandwidth_fft_w'])
               
                ### Spectral Contrast ########################################################################################
                #if ('SpecContrast' in jsonData['GENERAL_featuresToCalculate']):
                #    resSpecContrast[si] = Features.compute_SpecProp_Contrast(
                #            soundMono       = sample, 
                #            rate            = rate, 
                #            timeWindow      = jsonData['GENERAL_timeWindow'], 
                #            fft_w           = jsonData['SpecContrast_fft_w'])
       
                ### Spectral Rolloff #########################################################################################
                if ('SpecRolloff' in jsonData['GENERAL_featuresToCalculate']):
                    resSpecRolloff[si] = Features.compute_SpecProp_Rolloff(
                            soundMono       = sample, 
                            rate            = rate, 
                            timeWindow      = jsonData['GENERAL_timeWindow'], 
                            fft_w           = jsonData['SpecRolloff_fft_w'])
               
                ### Spectral Poly Features ###################################################################################
                if ('SpecPolyFeat' in jsonData['GENERAL_featuresToCalculate']):
                       
                    if jsonData['SpecPolyFeat_power'] == True:
                        sampleChosen = samplePWR[jsonData['SpecPolyFeat_fft_w']]
                    else:
                        sampleChosen = sampleDB[jsonData['SpecPolyFeat_fft_w']]
                       
                    resSpecPolyFeatures_Zero[si], resSpecPolyFeatures_Linear[si], resSpecPolyFeatures_Quadratic[si] = Features.compute_SpecProp_PolyFeatures(
                            sample          = sampleChosen, 
                            rate            = rate, 
                            timeWindow      = jsonData['GENERAL_timeWindow'], 
                            min_freq        = jsonData['SpecPolyFeat_min_freq'], 
                            max_freq        = jsonData['SpecPolyFeat_max_freq'], 
                            fft_w           = jsonData['SpecPolyFeat_fft_w'], 
                            db_threshold    = jsonData['GENERAL_dbThreshold'], 
                            power           = False if (jsonData['SpecPolyFeat_power'] == "False") else True )
           
                ### Spectral Spread ##########################################################################################
                if ('SpecSpread' in jsonData['GENERAL_featuresToCalculate']):
                    resSpread[si] = Features.compute_SpecSpread(
                            soundMono       = sample,
                            sample          = sampleDB[jsonData['SpecSpread_fft_w']],                         
                            rate            = rate, 
                            timeWindow      = jsonData['GENERAL_timeWindow'], 
                            fft_w           = jsonData['SpecSpread_fft_w'], 
                            db_threshold    = jsonData['GENERAL_dbThreshold'])
               
                ### SPL ######################################################################################################
                if ('SPL' in jsonData['GENERAL_featuresToCalculate']):
                    resSPL[si] = Features.compute_SPL(
                            sample          = sampleDB[jsonData['SPL_fft_w']], 
                            rate            = rate, 
                            timeWindow      = jsonData['GENERAL_timeWindow'], 
                            min_freq        = jsonData['SPL_min_freq'], 
                            max_freq        = jsonData['SPL_max_freq'], 
                            fft_w           = jsonData['SPL_fft_w'], 
                            db_threshold    = jsonData['GENERAL_dbThreshold'])
    
            ### Assemble all arrays into a matrix ########################################################################
            result, selectedColNames = groupFeatures(
                jsonData['GENERAL_featuresToCalculate'], 
                fileIdent, 
                jsonData['GENERAL_fileDateTimeMask'],
                jsonData['GENERAL_timeWindow'])
    
            ### Save CSV with our extraction format
            dataframe = pd.DataFrame(result, columns = selectedColNames)
            dataframe.to_csv("/".join([jsonData['GENERAL_mainPath'], "Extraction/AudioFeatures", fileIdent+'.csv']), index=False)        
    
            numCurrentFile = numCurrentFile + 1    
            if extractSpectrograms == False:
                numCurrentSpectrogram = numCurrentSpectrogram + 1

            tt1 = time.time()
            totalTime = tt1 - tt0
            sumTimeFiles = sumTimeFiles + totalTime
            print("        Saving Time: %s" % totalTime)

        except:
            print("#####################################################")
            print("#####################################################")
            print("### Error on file: %s" % currentFile)
            print("#####################################################")
            print("#####################################################")
            numCurrentFile = numCurrentFile + 1
            # Append the faulty filename 
            with open("/".join([jsonData['GENERAL_mainPath'], "Extraction/PROCESSING_ERROR_FeatureExtraction.log"]), "a") as myfile:
                myfile.write("   %s\n" % currentFile)


    ### Save DB info
    dbnamesplit = dbname.split("/")
    currentdb = dbnamesplit[len(dbnamesplit)-1]
    with open("/".join([jsonData['GENERAL_mainPath'], "Extraction/dbinfo.csv"]), 'w') as outfileinfo:            
        outfileinfo.write("dbname,numfiles,numsamples,samplewindow\n")
        outfileinfo.write("%s,%s,%s,%s" % (currentdb, totAudios, totSamples, jsonData['GENERAL_timeWindow']))         

    ##############################################################################################################
    ### END OF LOOP ##############################################################################################
    ##############################################################################################################
  
    unifyCSV(mainPath = jsonData['GENERAL_mainPath'],
             resultingCSVName = "features",
             groupingOperation = jsonData['GENERAL_groupingOperation'], 
             normalizationOperation = jsonData['GENERAL_normalizationOperation'], 
             removeOutliers = jsonData['GENERAL_removeOutliers'])

    #Apago o resulting file para não ocupar muito espaço.
    #print("REMOVE features.csv: %s/Extraction/features.csv" % (jsonData['GENERAL_mainPath'])  )
    #os.remove("%s/Extraction/features.csv" % (jsonData['GENERAL_mainPath']))    

    ##############################################################################################################################
    ### EXTRACT SPECTROGRAMS #####################################################################################################
    ##############################################################################################################################    
    # Append the boat extraction errors
    with open("/".join([currentPath, "Extraction/PROCESSING_ERROR_FeatureExtraction.log"]), "a") as myfile:
        myfile.write("\n")
        myfile.write("### ERROR ON THE SPECTROGRAM GENERATION PROCESS ####\n")
    
    if extractSpectrograms == True:
        #Load an audio file with separated samplerate, left and right channels        
        for currentFile in audios:
            print("   PROCESSING: %s" % (currentFile))
            try:
                #timeSpec = time.time()
                
                if numCurrentSpectrogram <= len(audios):
                    #numCurrentSpectrogram = numCurrentSpectrogram + 1
                    print("   Generating Spectrogram %s of %s (%s)" % (numCurrentSpectrogram+1, len(audios), currentFile))
                
                if currentFile[-4:] == 'flac':
                    fileIdent = currentFile[:-5]
                else:
                    fileIdent = currentFile[:-4]
                
                frames, rate = sf.read("/".join([jsonData['GENERAL_mainPath'],currentFile]))
                
                # Pego um canal apenas
                if (len(np.shape(frames)) == 1):                    
                    selChannel = frames
                else:
                    if (jsonData['GENERAL_audioChannel'] == 0):                        
                        selChannel = frames[:,0]                        
                    else:
                        selChannel = frames[:,1]
                
                numMinutes = int(np.ceil((len(frames)/rate)/60))
                
                # Extract the spectrogram matrix (STFT)
                #sampleWindows = len(selChannel)/(rate)
                #specMatrix = np.zeros([rate/2, sampleWindows])
                #for i in range(sampleWindows):
                #    t1 = i*rate
                #    t2 = (i+1)*rate
                #    specMatrix[:,i] = calculate_spectrum(selChannel[t1:t2], rate)
                specRate = 4096
                sampleWindows = len(selChannel)/(specRate)
                specMatrix = np.zeros([specRate/2, sampleWindows])
                for i in range(sampleWindows):
                    t1 = i*specRate
                    t2 = (i+1)*specRate
                    specMatrix[:,i] = calculate_spectrum(selChannel[t1:t2], specRate)
                
                print('clausius 0')
                # CREATE AN AUDIO SPECTROGRAM OF THE ENTIRE FILE
                spec_width = 1200
                spec_height = 600
                speccmap = plt.cm.jet
                specMatrixDB = librosa.amplitude_to_db(specMatrix, ref=1.0, top_db=jsonData['GENERAL_dbThreshold'])
                specMatrixDB1 = np.flipud(specMatrixDB)
                norm = plt.Normalize(vmin=specMatrixDB1.min(), vmax=specMatrixDB1.max())    
                image = speccmap(norm(specMatrixDB1))        
                newImage = scipy.misc.imresize(image, (spec_height, spec_width))
                plt.imsave("%s/Extraction/AudioSpectrogram/%s.png" % (jsonData['GENERAL_mainPath'], fileIdent), newImage)
                
                #with open("%s/Extraction/AudioSpectrogram/%s.csv" % (jsonData['GENERAL_mainPath'], fileIdent), 'wb') as csvfile:
                #    fieldnames = ['sampleWindow', 'audioSeconds', 'maxFreq']
                #    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')            
                #    writer.writeheader()
                #    writer.writerow({'sampleWindow': (numMinutes*60)/sampleWindows,
                #                     'audioSeconds': numMinutes*60,
                #                     'maxFreq': rate/2})                
                
                print('clausius 1')
                numCurrentSpectrogram = numCurrentSpectrogram + 1
                print('clausius 2')                
                #timeSpec1 = time.time()
                print('clausius 3')                
                #sumTimeSpectrograms = sumTimeSpectrograms + (timeSpec1-timeSpec)
                print('clausius 4')                
            except:
                print("#####################################################")
                print("#####################################################")
                print("### Error on file: %s" % currentFile)
                print("#####################################################")
                print("#####################################################")
                numCurrentSpectrogram = numCurrentSpectrogram + 1
                # Append the faulty filename 
                with open("/".join([currentPath, "Extraction/PROCESSING_ERROR_FeatureExtraction.log"]), "a") as myfile:
                    myfile.write("   %s\n" % currentFile)
        
    print("###############################################################")
    print("### Process Completed #########################################")
    print("###############################################################")
    #timeEnd = time.time()
    #timeToProcess = timeEnd-timeStart
    #meanTimeToProcess = sumTimeFiles/numFiles
    #meanTimeToProcessSpec = sumTimeSpectrograms/numFiles
    #print("Time to process %s files: %.2f seconds" % (numFiles, timeToProcess))
    #print("Mean time to process each file: %.2f seconds" % (meanTimeToProcess))
    #print("Mean time to generate each spectrogram: %.2f seconds" % (meanTimeToProcessSpec))
    #print(" ")


    print "#### EXTRACTION PROCESS FINISH ####"
      
    time.sleep(1)    

    if (audioType == '.mp3'):
        os.system("rm -rf %s/*.wav" % (jsonData['GENERAL_mainPath']))

    ### Update system flags: ##################################################################
    extractionProcess = True
    extractBoat       = False
    extract24h        = False
    vis               = [False] *10
        
    if (os.path.isfile("/".join([jsonData['GENERAL_mainPath'], "Extraction", 'flags_extraction.csv']))):
        vis = []
        data, headers = loadFlags(jsonData['GENERAL_mainPath'])
        extractionProcess    = str2bool(data[0])
        extractSpectrograms1 = str2bool(data[1])
        extractMP31          = str2bool(data[2])
        extractBoat          = str2bool(data[3])
        extract24h           = str2bool(data[4])
        vis.append(str2bool(data[5]))
        vis.append(str2bool(data[6]))
        vis.append(str2bool(data[7]))
        vis.append(str2bool(data[8]))
        vis.append(str2bool(data[9]))
        vis.append(str2bool(data[10]))
        vis.append(str2bool(data[11]))
        vis.append(str2bool(data[12]))
        vis.append(str2bool(data[13]))
        vis.append(str2bool(data[14]))
    
    if extractSpectrograms == True:
        extractSpectrograms1 = True
        
    if extractMP3 == True:
        extractMP31 = True

    ################################################################################################
    ### Code to enable each visualization will go here ##############################################    
    ################################################################################################
    
    print("   ### Printing extraction flags")
    with open("/".join([jsonData['GENERAL_mainPath'], "Extraction", 'flags_extraction.csv']), 'w') as outfileinfo:            
        outfileinfo.write("features,spectrograms,mp3,boat,24h,vis0,vis1,vis2,vis3,vis4,vis5,vis6,vis7,vis8,vis9\n")
        outfileinfo.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % 
            (extractionProcess, extractSpectrograms, extractMP3, extractBoat, extract24h,
             vis[0],vis[1],vis[2],vis[3],vis[4],vis[5],vis[6],vis[7],vis[8],vis[9]))
                     
    print(" ")
    print("### FINISH ###")

def loadFlags(path):
    with open("%s/Extraction/flags_extraction.csv" % (path), 'rb') as f:
        reader = csv.reader(f)        
        headers = next(reader, None)                
        
        for row in reader:
            data = row;
    return data, headers

def str2bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError

@app.route("/start_feature_extraction", methods=['GET', 'POST'])
def startFeatureExtraction():    
    global currentPath
    global numCurrentFile
    global numCurrentSpectrogram
    global numFiles
    global extractSpectrograms    
    global extractMP3 
    global teste

    print("### TESTE DE PRINT NA TELA ###")

    numCurrentFile = 0
    numCurrentSpectrogram = 0
    numFiles = 1

    # Load features settings from HTML
    featArray = ["ACI","ADI","AEI","BIO","MFCC","NDSI","RMS","TH","ZCR","SpecMean","SpecSD",
                 "SpecSEM","SpecMedian","SpecMode","SpecQuartile","SpecSkewness",
                 "SpecKurtosis","SpecEntropy","SpecVariance","SpecChromaSTFT",
                 "SpecChromaCQT","SpecCentroid","SpecBandwidth",
                 "SpecRolloff","SpecPolyFeat","SpecSpread","SPL"]

    parameters = request.form.to_dict()

    if 'action' in request.form.keys():
        action = parameters['action']
        
        json_data = {}                       

        json_data['GENERAL_mainPath'] = currentPath #"../../FILES/FET/".$_REQUEST['GENERAL_mainPath'])                
        json_data['GENERAL_fileDateTimeMask'] = parameters['GENERAL_fileDateTimeMask']
        json_data['GENERAL_audioChannel'] = int(parameters['GENERAL_audioChannel'])
        json_data['GENERAL_timeWindow'] = int(parameters['GENERAL_timeWindow'])
        json_data['GENERAL_groupingOperation'] = parameters['GENERAL_groupingOperation']
        json_data['GENERAL_normalizationOperation'] = parameters['GENERAL_normalizationOperation']
        json_data['GENERAL_removeOutliers'] = parameters['GENERAL_removeOutliers']
        json_data['GENERAL_dbThreshold'] = int(parameters['GENERAL_dbThreshold'])
        
        featuresSelected = []
        for value in featArray:
            if ("enable%s" % (value)) in request.form.keys():
                featuresSelected.append(value)
        json_data['GENERAL_featuresToCalculate'] = featuresSelected
        
        json_data['ACI_min_freq'] = int(parameters['ACI_min_freq'])
        json_data['ACI_max_freq'] = int(parameters['ACI_max_freq'])
        json_data['ACI_j_bin'] = int(parameters['ACI_j_bin'])
        json_data['ACI_fft_w'] = int(parameters['ACI_fft_w'])     
        
        json_data['ADI_max_freq'] = int(parameters['ADI_max_freq'])
        json_data['ADI_freq_step'] = int(parameters['ADI_freq_step'])
        json_data['ADI_fft_w'] = int(parameters['ADI_fft_w'])

        json_data['AEI_max_freq'] = int(parameters['AEI_max_freq'])
        json_data['AEI_freq_step'] = int(parameters['AEI_freq_step'])
        json_data['AEI_fft_w'] = int(parameters['AEI_fft_w'])
        
        json_data['BIO_min_freq'] = int(parameters['BIO_min_freq'])
        json_data['BIO_max_freq'] = int(parameters['BIO_max_freq'])
        json_data['BIO_fft_w'] = int(parameters['BIO_fft_w'])
        
        json_data['MFCC_numcep'] = int(parameters['MFCC_numcep'])
        json_data['MFCC_min_freq'] = int(parameters['MFCC_min_freq'])
        json_data['MFCC_max_freq'] = int(parameters['MFCC_max_freq'])
        json_data['MFCC_useMel'] = parameters['MFCC_useMel']
        
        json_data['NDSI_fft_w'] = int(parameters['NDSI_fft_w'])
        json_data['NDSI_anthrophony'] = [int(parameters['NDSI_anthrophony_min']), int(parameters['NDSI_anthrophony_max'])]
        json_data['NDSI_biophony'] = [int(parameters['NDSI_biophony_min']), int(parameters['NDSI_biophony_max'])]
        
        json_data['SpecSpread_fft_w'] = int(parameters['SpecSpread_fft_w'])
        
        json_data['SpecMean_min_freq'] = int(parameters['SpecMean_min_freq'])
        json_data['SpecMean_max_freq'] = int(parameters['SpecMean_max_freq'])
        json_data['SpecMean_fft_w'] = int(parameters['SpecMean_fft_w'])
        
        json_data['SpecSD_min_freq'] = int(parameters['SpecSD_min_freq'])
        json_data['SpecSD_max_freq'] = int(parameters['SpecSD_max_freq'])
        json_data['SpecSD_fft_w'] = int(parameters['SpecSD_fft_w'])
        
        json_data['SpecSEM_min_freq'] = int(parameters['SpecSEM_min_freq'])
        json_data['SpecSEM_max_freq'] = int(parameters['SpecSEM_max_freq'])
        json_data['SpecSEM_fft_w'] = int(parameters['SpecSEM_fft_w'])
        
        json_data['SpecMedian_min_freq'] = int(parameters['SpecMedian_min_freq'])
        json_data['SpecMedian_max_freq'] = int(parameters['SpecMedian_max_freq'])
        json_data['SpecMedian_fft_w'] = int(parameters['SpecMedian_fft_w'])
        
        json_data['SpecMode_min_freq'] = int(parameters['SpecMode_min_freq'])
        json_data['SpecMode_max_freq'] = int(parameters['SpecMode_max_freq'])
        json_data['SpecMode_fft_w'] = int(parameters['SpecMode_fft_w'])
        
        json_data['SpecQuartile_min_freq'] = int(parameters['SpecQuartile_min_freq'])
        json_data['SpecQuartile_max_freq'] = int(parameters['SpecQuartile_max_freq'])
        json_data['SpecQuartile_fft_w'] = int(parameters['SpecQuartile_fft_w'])
        
        json_data['SpecSkewness_min_freq'] = int(parameters['SpecSkewness_min_freq'])
        json_data['SpecSkewness_max_freq'] = int(parameters['SpecSkewness_max_freq'])
        json_data['SpecSkewness_fft_w'] = int(parameters['SpecSkewness_fft_w'])        
        json_data['SpecSkewness_power'] = parameters['SpecSkewness_power']        
        
        json_data['SpecKurtosis_min_freq'] = int(parameters['SpecKurtosis_min_freq'])
        json_data['SpecKurtosis_max_freq'] = int(parameters['SpecKurtosis_max_freq'])
        json_data['SpecKurtosis_fft_w'] = int(parameters['SpecKurtosis_fft_w'])
        json_data['SpecKurtosis_power'] = parameters['SpecKurtosis_power']
        
        json_data['SpecEntropy_min_freq'] = int(parameters['SpecEntropy_min_freq'])
        json_data['SpecEntropy_max_freq'] = int(parameters['SpecEntropy_max_freq'])
        json_data['SpecEntropy_fft_w'] = int(parameters['SpecEntropy_fft_w'])
        json_data['SpecEntropy_power'] = parameters['SpecEntropy_power']
        
        json_data['SpecVariance_min_freq'] = int(parameters['SpecVariance_min_freq'])
        json_data['SpecVariance_max_freq'] = int(parameters['SpecVariance_max_freq'])
        json_data['SpecVariance_fft_w'] = int(parameters['SpecVariance_fft_w'])
        json_data['SpecVariance_power'] = parameters['SpecVariance_power']
        
        json_data['SpecChromaSTFT_min_freq'] = int(parameters['SpecChromaSTFT_min_freq'])
        json_data['SpecChromaSTFT_max_freq'] = int(parameters['SpecChromaSTFT_max_freq'])
        json_data['SpecChromaSTFT_fft_w'] = int(parameters['SpecChromaSTFT_fft_w'])
        json_data['SpecChromaSTFT_power'] = parameters['SpecChromaSTFT_power']
        
        json_data['SpecCentroid_fft_w'] = int(parameters['SpecCentroid_fft_w'])
        
        json_data['SpecBandwidth_fft_w'] = int(parameters['SpecBandwidth_fft_w'])
        
        #json_data['SpecContrast_fft_w'] = int(parameters['SpecContrast_fft_w'])
        
        json_data['SpecRolloff_fft_w'] = int(parameters['SpecRolloff_fft_w'])
        
        json_data['SpecPolyFeat_min_freq'] = int(parameters['SpecPolyFeat_min_freq'])
        json_data['SpecPolyFeat_max_freq'] = int(parameters['SpecPolyFeat_max_freq'])
        json_data['SpecPolyFeat_fft_w'] = int(parameters['SpecPolyFeat_fft_w'])
        json_data['SpecPolyFeat_power'] = parameters['SpecPolyFeat_power']
        
        json_data['SPL_min_freq'] = int(parameters['SPL_min_freq'])
        json_data['SPL_max_freq'] = int(parameters['SPL_max_freq'])
        json_data['SPL_fft_w'] = int(parameters['SPL_fft_w'])

        extractSpectrogramsBool = parameters['GENERAL_extractSpectrograms']
        if extractSpectrogramsBool == 'True':    
            extractSpectrograms = True
        else:
            extractSpectrograms = False        
        
        extractMP3Bool = parameters['GENERAL_extractMP3']
        if extractMP3Bool == 'True':    
            extractMP3 = True
        else:
            extractMP3 = False                
        
        json_final = json.dumps(json_data)

        outputPath = "/".join([currentPath, "Extraction"])
        if (os.path.isdir(outputPath) == False):
            os.system("mkdir %s" % (outputPath))

        with open("/".join([currentPath, "Extraction", "parameters.json"]), 'w') as outfile:
            json.dump(json_data, outfile)

        # Start the ACTUAL extraction process
        jsonData = json.load(open("/".join([currentPath, "Extraction", "parameters.json"])))
        extractionProcess(jsonData=jsonData, dbname=currentPath)

    # TODO: Create the progress bar

    return jsonify({"status": "Extraction finished"})

@app.route("/get_progress", methods=['GET', 'POST'])
def getProgress():    
    global currentPath
    global numCurrentFile
    global numCurrentSpectrogram
    global numFiles
    
    #print("PROGRESS: %s - %s" % (numCurrentFile, numCurrentSpectrogram))    
        
    return jsonify({"numCurrentFile": numCurrentFile, 
                    "numCurrentSpectrogram": numCurrentSpectrogram, 
                    "numFiles": numFiles})

@app.route('/addLabels', methods=['POST'])
def addLabels():
    global currentPath
    
    labelName = request.form.get('labelName')
    labelList = request.form.getlist('labelList')
    
    print(labelList[0])
    
    #break the labelList by ";"
    labelList1 = labelList[0].split(";")
    
    # Load the file if exists, otherwise, create it
    df = pd.DataFrame()
    df1 = pd.DataFrame()
    if os.path.exists("%s/Extraction/features_labels.csv" % (currentPath)):
        df = pd.read_csv("%s/Extraction/features_labels.csv" % (currentPath))
    else:    
        df.to_csv("%s/Extraction/features_labels.csv" % (currentPath), sep=";")
    
    dfFeatures = pd.read_csv("%s/Extraction/features_norm.csv" % (currentPath))
    
    if (df.empty):        
        df = pd.DataFrame(columns=[labelName])
        for i in range( len(dfFeatures.index) ):
            df = df.append({labelName: 0}, ignore_index=True)
        
        for i in range( len(dfFeatures.index) ):
            for j in range( len(labelList1) ):
                auxLL = labelList1[j].split("|")        
                if (auxLL[0] == dfFeatures.values[i][0]) & (int(auxLL[1]) == dfFeatures.values[i][3]):                
                    df.values[i][0] = 1
    else:
        if labelName in df.columns:
            df.drop(labelName, axis=1, inplace=True)
        
        df1 = pd.DataFrame(columns=[labelName])
        for i in range( len(dfFeatures.index) ):
            df1 = df1.append({labelName: 0}, ignore_index=True)
        
        for i in range( len(dfFeatures.index) ):
            for j in range( len(labelList1) ):
                auxLL = labelList1[j].split("|")
                if (auxLL[0] == dfFeatures.values[i][0]) & (int(auxLL[1]) == dfFeatures.values[i][3]):                
                    df1.values[i][0] = 1

    # Add the column
    df = df.join(df1)
        
    # Finally, save the CSV        
    df.to_csv("%s/Extraction/features_labels.csv" % (currentPath), sep=",", index=None)

    return jsonify({"status": "Labels added"})



#@app.route("/set_data", methods=['GET', 'POST'])
#def setData():    
#    #global SDThreshold
#    #global maximumFrequency
#    #global plotImages
#    #global extractSpectrograms
#    global settings1
#    
#    parameters = request.form.to_dict()
#
#    #SDThreshold = float(parameters['threshold'])
#    #maximumFrequency = int(parameters['maximumFrequency'])
#    #plotImagesInt = int(parameters['plotImages'])
#    #extractSpectrogramsInt = int(parameters['extractSpectrograms'])
#    settings1 = float(parameters['aaa'])
#    
#    #if plotImagesInt == 1:    
#    #    plotImages = True
#    #else:
#    #    plotImages = False 
#    
#    #if extractSpectrogramsInt == 1:    
#    #    extractSpectrograms = True
#    #else:
#    #    extractSpectrograms = False    
#
#    return jsonify({"status": "Settings defined"})
#    #return render_template("page2.html")


if __name__ == "__main__":

    global serverport    
    
    serverport = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:{0}".format(serverport)    
    
    threading.Timer(1.25, lambda: webbrowser.open(url) ).start()
    
    import logging
    log = logging.getLogger('werkzeug')
    log.disabled = True
    app.logger.disabled = True    
    
    app.run(port=serverport, debug=True, use_reloader=True)
    
    
# Para matar um servidor que ficou ativo:        
#    ps -fA | grep python 
#    kill <id>