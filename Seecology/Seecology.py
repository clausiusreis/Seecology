#!/usr/bin/env python2.7
# coding: utf-8

#Examle of date stamp: %Y.%m.%d_%H.%M.%S

import json
import os
#import subprocess
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
#from librosa import display
import soundfile as sf
#import pywt
#from scipy.signal import spectrogram
import scipy.misc
import csv
#import copy
import time
#import scipy.cluster.hierarchy as sch
#from scipy.cluster.hierarchy import fcluster
#from scipy import spatial
#from scipy import signal
from datetime import datetime
from datetime import timedelta
import copy
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
from scipy import spatial

# External http server to provide access to files on the chosen path
from threading import Thread
#import http.server # Python3
import SimpleHTTPServer
import SocketServer

### External functions
sys.path.append("scripts")

### Available feature functions
import Features

### Unify and normalize the individual files
#import UnifiedCSVGenerator as UG

app = Flask('Seecology_ClausiusReis')

print("### "+app.instance_path)

# For the progress bar
global currentPath
global numCurrentFile
global numCurrentSpectrogram
global numFiles
global numFiles24h
global numFilesSpec24h
global numCurrentFiles24h
global numCurrentFilesSpec24h

global extractSpectrograms24h
global maximumFrequency24h
global sampleWindow24h
global audioChannel
global dateFormat24h
global numCurrentSpectrogram24h

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
numFiles24h = 0
numFilesSpec24h = 0
numCurrentFiles24h = 0
numCurrentFilesSpec24h = 0
SDThreshold = 1.5
plotImages = True
extractSpectrograms = False
#initialDirectory="~/Documents/DOUTORADO/TEST_DATABASE"
#initialDirectory="/home/clausius/public_html/DOUTORADO_Visualizations_Framework/testData"
initialDirectory="/home/clausius/Documents/DOUTORADO/DadosTeste/PEMLS"

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

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

################################################################################################################
### HTTP SERVER - Start ########################################################################################
################################################################################################################
#class HTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
#    def end_headers(self):
#        self.send_header('Access-Control-Allow-Origin', '*')
#        http.server.SimpleHTTPRequestHandler.end_headers(self)
#
#def server(port):
#    httpd = socketserver.TCPServer(('', port), HTTPRequestHandler)
#    return httpd
#
#def servidor():    
#    global serverport
#    global currentPath
#
#    os.chdir(currentPath)
#    
#    httpd = server(serverport+10)
#    try:        
#        os.chdir(currentPath)
#        print("\nServing at localhost:" + str(serverport+10))
#        httpd.serve_forever()
#    except KeyboardInterrupt:
#        print("\n...shutting down http server")
#        httpd.shutdown()
#        sys.exit()    
    
class HTTPRequestHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        SimpleHTTPServer.SimpleHTTPRequestHandler.end_headers(self)

def server(port):
    httpd = SocketServer.TCPServer(('', port), HTTPRequestHandler)
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

################################################################################################################
### HTTP SERVER - End ##########################################################################################
################################################################################################################

#TODO: Corrigir esta função
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
    print("Processing directory: %s" % mainPath)

    firstHeader = 0
    resultingFile = '%s/Extraction/%s.csv' % (mainPath, resultingCSVName)
    with open(resultingFile, 'wb') as outputCsvfile:
        csvwriter = csv.writer(outputCsvfile, delimiter=',')
    
        for f in csvFiles:
            print("   Processing file: %s" % f)
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
    print("Processing file: %s" % resultingFile)
    
    #load original CSV
    df = pd.read_csv(resultingFile)

    ### DATA GROUPING ###################################################################################    
    result = df.groupby(['FileName', 'Date', 'Time', 'SecondsFromStart']).mean()
    result = pd.DataFrame(result)    
    if (result.shape[0] == 0):
        result = df.groupby(['FileName', 'SecondsFromStart']).mean()
        result = pd.DataFrame(result)

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

    ### DATA GROUPING ###################################################################################
    df = pd.read_csv("%s/Extraction/features_norm.csv" % (currentPath))
    
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

    ### Generate individual normalized files ##############################################################
    df = pd.DataFrame()
    df = pd.read_csv("%s/Extraction/features_norm.csv" % (currentPath))
    
    auxLoop = df.groupby(['FileName', 'Date', 'Time']).mean()
    if (auxLoop.shape[0] == 0):
        auxLoop = df.groupby(['FileName', 'SecondsFromStart']).mean()

    for index, row in auxLoop.iterrows():
        df2 = df[df['FileName'] == index[0]]
        df2.to_csv('%s/Extraction/AudioFeatures/%s.nor' % (currentPath, index[0]), sep=',', index=False)
            
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
### Smooth function to allow simple detection within spectrogram
###########################################################################################################
def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]  

###########################################################################################################
### Algoritmo de detecção por variação da amplitude e desvio padrão (Minuto)
###########################################################################################################
def extractBoatSignatures():
    global SDThreshold
    global timeToProcess
    global meanTimeToProcess
    global meanTimeToProcessSpec
    global plotImages
    global extractSpectrograms
    global audioChannel
    
    sumTimeFiles = 0
    sumTimeSpectrograms = 0
    timeStart = time.time()
    print("###############################################################")
    print("### Process Started ###########################################")
    print("###############################################################")

    ###########################################################################################################
    ### Variáveis de configuração
    ###########################################################################################################
    #audioChannel = 0
    topDB = 50
    W = 13
    H = 5
    #plotImages = True
    #extractSpectrograms = True

    global currentPath
    global numCurrentFile
    global numCurrentSpectrogram
    global numFiles
    global maximumFrequency
    global dateFormat24h
    
    ###########################################################################################################
    ### Create the necessary directories
    ###########################################################################################################
    outputPath = "/".join([currentPath, "Extraction"])
    if (os.path.isdir(outputPath) == False):
        os.system("mkdir %s" % (outputPath))
    
    outputPath = "/".join([currentPath, "Extraction", "BoatSpectrogram"])
    if (os.path.isdir(outputPath) == False):
        os.system("mkdir %s" % (outputPath))
    
    outputPath = "/".join([currentPath, "Extraction", "DetectionImg"])
    if (os.path.isdir(outputPath) == False):
        os.system("mkdir %s" % (outputPath))    
    
    outputPath = "/".join([currentPath, "Extraction", "Detection"])
    if (os.path.isdir(outputPath) == False):
        os.system("mkdir %s" % (outputPath))    
    
    #outputPath = "/".join([currentPath, "Extraction", "DetectionSignature"])
    #if (os.path.isdir(outputPath) == False):
    #    os.system("mkdir %s" % (outputPath))    
        
    #outputPath = "/".join([currentPath, "Extraction", "DetectionInfo"])
    #if (os.path.isdir(outputPath) == False):
    #    os.system("mkdir %s" % (outputPath))

    #outputPath = "/".join([currentPath, "Extraction", "DetectionSpectrum"])
    #if (os.path.isdir(outputPath) == False):
    #    os.system("mkdir %s" % (outputPath))
        
    #outputPath = "/".join([currentPath, "Extraction", "DetectionVariance"])
    #if (os.path.isdir(outputPath) == False):
    #    os.system("mkdir %s" % (outputPath))
    
    # Reset the extraction error report
    with open("/".join([currentPath, "Extraction/PROCESSING_ERROR_BoatDetection.log"]), "w") as myfile:
        myfile.write("####################################################\n")
        myfile.write("### ERROR REPORT - Seecology                     ###\n")
        myfile.write("####################################################\n")    
    
    ###########################################################################################################
    ### List all the allowed files on the directory
    ###########################################################################################################
    wdir = os.listdir("%s" % (currentPath))
    wdir.sort()
    currentFiles = [arq for arq in wdir if (arq.lower().endswith(".wav") | 
                                            arq.lower().endswith(".raw") | 
                                            arq.lower().endswith(".flac"))]

    numFiles = len(currentFiles)

    ##############################################################################################################################
    ### EXTRACT BOAT SIGNATURE ###################################################################################################
    ##############################################################################################################################    
    # Append the boat extraction errors
    with open("/".join([currentPath, "Extraction/PROCESSING_ERROR_BoatDetection.log"]), "a") as myfile:
        myfile.write("\n")    
        myfile.write("### ERROR ON THE EXTRACTION PROCESS ################\n")
    
    #Summary file for the visualization control
    fieldnamesSummary = ['Index', 'IndexSpec', 'Filename', 'Date', 'NumPeaks', 'Detection', 'Multiple']
    with open("/".join([currentPath, "Extraction/BoatDetectionSummary.csv"]), "w") as BDSummary:        
        BDSummaryWriter = csv.DictWriter(BDSummary, fieldnames=fieldnamesSummary, delimiter=',')       
        BDSummaryWriter.writeheader()    

    currentBDSummary = 0
    currentIndexSpec = 0
    previousFile = ""
    normalizationMin = 1000;
    normalizationMax = -1000;
    for currentFile in currentFiles:
        try:        
            timeFile = time.time()
            
            if numCurrentFile <= len(currentFiles):
                #numCurrentFile = numCurrentFile + 1
                print("   Processing file %s of %s (%s)" % (numCurrentFile+1, len(currentFiles), currentFile))
            detectionData = []        
            boatDetection = []
            boatDetectionP = []    
            
            if currentFile[-4:] == "flac":
                fileIdent = currentFile[:-5]
            else:
                fileIdent = currentFile[:-4]
        
            #Load an audio file with separated samplerate, left and right channels
            frames, rate = sf.read("/".join([currentPath,currentFile]))
        
            # Pego um canal apenas
            if (len(np.shape(frames)) == 1):
                selChannel = frames
            else:
                if (audioChannel == 0):
                    selChannel = frames[:,0]
                else:
                    selChannel = frames[:,1]
        
            numMinutes = int(np.ceil((len(frames)/rate)/60))
            
            ##############################################################################################################################
            ### Get the min and max values for normalization 
            #if extractSpectrograms == True:            
            #    specRate1 = rate/2
            #    sampleWindows1 = len(selChannel)/(specRate1)
            #    specMatrix1 = np.zeros([specRate1/2, sampleWindows1])
            #    for i in range(sampleWindows1):
            #        t11 = i*specRate1
            #        t22 = (i+1)*specRate1
            #        specMatrix1[:,i] = calculate_spectrum(selChannel[t11:t22], specRate1)            
    
            #    if np.min(specMatrix1) < normalizationMin:
            #        normalizationMin = np.min(specMatrix1)
            #    if np.max(specMatrix1) > normalizationMax:
            #        normalizationMax = np.max(specMatrix1)
            ##############################################################################################################################
            
            ##############################################################################################################################
            ### EXTRACT BOAT DETECTION SIGNATURES ########################################################################################
            ##############################################################################################################################
            #fileSpectrumNormal = []        
            fileSpectrumSmooth = []
            fileVariances = []
            for minute in range(numMinutes):
                sample = selChannel[(rate*60*minute):(rate*60*(minute+1))]
        
                ####### TESTE #######
                # Calcular windowFFT dado o rate (Testar)
                #windowFFT = int(np.floor(rate / 1.34583))
                windowFFT = rate
                if windowFFT%2!=0:
                    windowFFT += 1
    
                # Extract the spectrogram matrix (STFT)
                #specMatrix = librosa.stft(selChannel, n_fft=windowFFT)
                specMatrix = librosa.stft(sample, n_fft=windowFFT)
            
                # Convert the spectrogram matrix to DB scale
                specMatrixDB = librosa.amplitude_to_db(specMatrix, ref=np.max, top_db=topDB)
        
                # Spectrogram Power
                #specPower = librosa.db_to_power(specMatrixDB, ref=1.0)
            
                #specMatrixDB = specPower

                ###########################################################################################################
                ### Teste de detecção com eliminação de outliers em DB
                ###########################################################################################################
                #SDThreshold = 1.5 # 3 normal, 4 with noise
                #SDThreshold1 = 1
                #SDFish = 2
        
                specMatrixMean = np.mean(specMatrixDB, axis=1)     
                #specMatrixMeanS = specMatrixMean        
                specMatrixMeanS = smooth(np.mean(specMatrixDB, axis=1), window='blackman', window_len=8)
                #specMatrixMeanSD = np.std(specMatrixMean)
                specMatrixMeanSD = np.std(specMatrixMeanS)
        
                #specPowerMean = np.mean(specPower, axis=1)
                #specPowerMeanS = smooth(np.mean(specPower, axis=1),32)
                #specPowerMeanSD = np.std(specPowerMean)
            
                freqs = librosa.core.fft_frequencies(sr=rate, n_fft=(windowFFT)+1)
    
                #Encontro o índice da frequência menor que 1000Hz
                maxFreq=0
                while (freqs[maxFreq] < maximumFrequency):
                    maxFreq += 1
        
                specMatrixMean = specMatrixMean[0:maxFreq]
                specMatrixMeanS = specMatrixMeanS[0:maxFreq]
        
                #fileSpectrumNormal.append(specMatrixMean)
                fileSpectrumSmooth.append(specMatrixMeanS)
        
                # Extraio a variação SD e detecto eventos tipo barco
                boatDet = np.zeros(len(specMatrixMean))
                #boatDet1 = np.zeros(len(specMatrixMean))
                variance = np.zeros(len(specMatrixMean))
                #for i in xrange(5,len(specMatrixMean)-5, 1): # Era 3 o 1
                for i in range(5,len(specMatrixMean)-5, 1): # Era 3 o 1
                    #vAmp = specMatrixMean[i] - specMatrixMean[i+1]
                    vAmp = specMatrixMeanS[i] - specMatrixMeanS[i+1]
        
                    #Teste da detecção apenas da saída ou entrada do pico            
                    if vAmp > 0:
                        vAmp = 0
                    variance[i] = abs(vAmp*vAmp*vAmp) # Adicionei mais um vAmp e abs
                    #variance[i] = abs(vAmp*vAmp)
            
                    # Detecção dos eventos significantes  (o -topDB é só para mostrar, poderia ser 1 e 0)
                    if (variance[i] >= specMatrixMeanSD * SDThreshold):
                        #for j in xrange(-2,3,1): #-6,7 # Janela de 5 Hz para poder fazer cruzamento de informações na classificação
                        for j in range(-2,3,1): #-6,7 # Janela de 5 Hz para poder fazer cruzamento de informações na classificação
                            #if plotImages:                    
                            #    boatDet[i+j] = -topDB
                            #else:
                            boatDet[i+j] = 1
                    #if (variance[i] >= specMatrixMeanSD * SDThreshold1):
                    #    for j in xrange(-4,5,1): # Janela de 5 para cada lado para poder fazer cruzamento de informações na classificação
                    #        if plotImages:                    
                    #            boatDet1[i+j] = -topDB
                    #        else:
                    #            boatDet1[i+j] = 1
    
                fileVariances.append(variance)
        
                # Ignoring the detections bellow 20Hz e últimas frequências
                i=0
                while (freqs[i] < 20):
                    i += 1
                boatDet[:i-1] = boatDet[-10:] = 0
                #boatDet1[:i-1] = boatDet1[-10:] = 0
                
                boatDetection.append([currentFile, minute, boatDet])
                #boatDetectionP.append([currentFile, minute, boatDet1])
    
                ##############################################################################################################################
                ### MULTIPLE SIGNATURES DETECTION ############################################################################################
                ##############################################################################################################################
                # Identifico se são múltiplas assinaturas e gero o detectionData
                sigFreq = []
                numPeaks = 0
                previous = 0
                multipleSignatures = 0
                for i in range(len(boatDet[:maxFreq])):    
                    if (boatDet[previous] != 0) & (boatDet[i]==0):
                        sigFreq.append(i)
                    previous = i
    
                if len(sigFreq) > 0:
                    numPeaks = len(sigFreq)
                    if numPeaks > 0:
                        detectionData.append( [minute ,boatDet.astype(int)] )
                        
                if len(detectionData) > 1:                
                    ### Compute the Distance Matrix
                    orderMethod = 6
                    numberOfClusters = 2
            
                    dist = np.zeros(shape=(len(detectionData), len(detectionData)))
                    for x in range(len(detectionData)):
                        for y in range(len(detectionData)):
                            dist[x,y] = 1 - spatial.distance.cosine(detectionData[x][1], detectionData[y][1])
                            
                    ### Matriz de distância (similaridade) ordenada.
                    m = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']
                    distMatrix = sch.linkage(dist, method=m[orderMethod])
                    Z1 = sch.dendrogram(distMatrix, no_plot=True)
                    featureNameOrderedID = Z1['leaves']
        
                    ### Dado um número de clusters classificar os features em grupos.
                    clusterObj = numberOfClusters
                    clusterDet = 10000
                    max_d = 0
                    while clusterDet > clusterObj:
                        max_d = max_d + 1
                        clusters = fcluster(distMatrix, max_d, criterion='distance')
                        clusterDet = len( range(1, clusters.max()+1 ) )
                    
                    clusters = fcluster(distMatrix, max_d, criterion='distance')
                    
                    #print(clusters)
      
                    i=0
                    newSignatures = np.zeros(shape=(len(detectionData[0][1]), 2), dtype=np.int8)
                    for cl in clusters:
                        newSignatures[:, cl-1] = (np.logical_or(newSignatures[:, cl-1].astype(int), detectionData[i][1].astype(int)) ).astype(int)
                        i = i+1
                    
                    distance = 1 - spatial.distance.cosine(newSignatures[:,0], newSignatures[:,1])
                    
                    #print( "###############################################################" )
                    #print( "### DETECTION #################################################" )
                    #print( "###############################################################" )
                    #print( "File: %s" % (currentFile) )
                    #print( "Distance: %s" % (distance) )
                    
                    if distance > 0.6:
                        #print("Single boat signature")
                        fileSignature = (np.logical_or(newSignatures[:, 0].astype(int), newSignatures[:, 1].astype(int)) ).astype(int)
                        multipleSignatures = 0                                                
                    else:
                        #print("Multiple boat signatures")
                        multipleSignatures = 1
                else:
                    if len(detectionData) == 1:
                        #print( "###############################################################" )
                        #print( "### DETECTION #################################################" )
                        #print( "###############################################################" )
                        #print( "File: %s" % (currentFile) )
                        #print( "Distance: %s" % (1.0) )
                        #print( "Single boat signature")
                        multipleSignatures = 0
                        fileSignature = ( detectionData[0][1] ).astype(int)
                ###########################################################################################################################
    
                if plotImages:
                    plt.ioff()
                    
                    sdPlot = [specMatrixMeanSD * SDThreshold for sx in specMatrixMean]
                    boatDetVis = copy.copy(boatDet)
                    boatDetVis[boatDetVis > 0] = -topDB
                        
                    fig, ax = plt.subplots()
                    fig.set_size_inches(W, H)
                    leg1 = mpatches.Patch(color='red', label="Boat Det. (%s x SD)" % (SDThreshold))
                    leg3 = mpatches.Patch(color='blue', label="Spectrum dB")
                    leg4 = mpatches.Patch(color='purple', label='Threshold %s x SD: %s' % (SDThreshold, "%.3f" % (SDThreshold*specMatrixMeanSD) ))
                    leg5 = mpatches.Patch(color='green', label="Spec. dB Amp. Variance")
                    plt.legend(handles=[leg1, leg3, leg4, leg5], loc=1)
                    plt.plot(freqs[:maxFreq], specMatrixMeanS[:maxFreq], lineWidth=0.5, color='blue')
                    plt.plot(freqs[:maxFreq], sdPlot[:maxFreq], linestyle='--', lineWidth=0.5, color='purple')
                    plt.plot(freqs[:maxFreq], variance[:maxFreq], lineWidth=0.5, color='green')
                    plt.fill(freqs[:maxFreq], boatDetVis[:maxFreq], lineWidth=0.5, color='red', alpha=0.5)    
                    plt.xscale('linear')
                    plt.title('Boat detection on SD Variance of Spectrum DB Mean: %s (Min. %s)' % (currentFile, minute))
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('Amplitude (dB)')
                    
                    fig.savefig(("%s/Extraction/DetectionImg/%s_M%s.png" % (currentPath, currentFile[:-4], minute)), bbox_inches='tight')
                    plt.close(fig)        
    
            # Save spectrum for each file
            #with open("%s/Extraction/DetectionSpectrum/%s_Normal.csv" % (currentPath, fileIdent), 'wb') as csvfile:
            #    spamwriter = csv.writer(csvfile, delimiter=',')
            #    for sig in fileSpectrumNormal:
            #        spamwriter.writerow(sig)
    
            #with open("%s/Extraction/DetectionSpectrum/%s_Smooth.csv" % (currentPath, fileIdent), 'wb') as csvfile:
            with open("%s/Extraction/Detection/%s_spectrum.csv" % (currentPath, fileIdent), 'wb') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',')
                for sig in fileSpectrumSmooth:
                    spamwriter.writerow(sig)
                
            # Save variance for each file
            #with open("%s/Extraction/DetectionVariance/%s.csv" % (currentPath, fileIdent), 'wb') as csvfile:
            with open("%s/Extraction/Detection/%s_variance.csv" % (currentPath, fileIdent), 'wb') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',')
                for sig in fileVariances:
                    spamwriter.writerow(sig)            
                
            # Save signatures for each file
            #fileSig = np.zeros(len(specMatrixMean))
            #with open("%s/Extraction/DetectionSignature/%s.csv" % (currentPath, fileIdent), 'wb') as csvfile:
            with open("%s/Extraction/Detection/%s_signature.csv" % (currentPath, fileIdent), 'wb') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',')
                #spamwriter.writerow(['Minute'] + range(maxFreq))
                for sig in boatDetection:
                    spamwriter.writerow(sig[2].astype(int).astype('S10'))
                    #fileSig = (np.logical_or(fileSig,sig[2].astype(int))).astype(int)
        
            # Save the file signature            
            #with open("%s/Extraction/DetectionSignature/%s.csv" % (currentPath, fileIdent), 'wb') as csvfile:
                #spamwriter = csv.writer(csvfile, delimiter=',')        
                #spamwriter.writerow(fileSig.astype('S10'))
        
            # Save signature info for each minute
            #with open("%s/Extraction/DetectionInfo/%s.csv" % (currentPath, fileIdent), 'wb') as csvfile:
            with open("%s/Extraction/Detection/%s_info.csv" % (currentPath, fileIdent), 'wb') as csvfile:
                fieldnames = ['Filename', 'Minute', 'LowFreq', 'HighFreq', 'DistFreqs', 'NumPeaks', 'Detection', 'Multiple', 'Cluster', 'SD']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')        
            
                writer.writeheader()            
                currentMinute = 0
                for sig in boatDetection:
                    sigFreq = []
                    lowFreq = 0
                    highFreq = 0
                    distFreqs = 0
                    numPeaks = 0
                    detection = 0
                    mSig = 0
        
                    previous = 0
                    for i in range(len(sig[2][:maxFreq])):    
                        if (sig[2][previous] != 0) & (sig[2][i]==0):
                            sigFreq.append(i)
                        previous = i
                    
                    if len(sigFreq) > 0:
                        lowFreq = sigFreq[0]
                        highFreq = sigFreq[len(sigFreq)-1]
                        distFreqs = highFreq - lowFreq
                        numPeaks = len(sigFreq)
                        if numPeaks > 0:
                            #detectionData.append( [sig[1] ,sig[2].astype(int)] )
                            detection = 1
                            if multipleSignatures == 1:
                                mSig = 1
                    
                    clusterID = 0
                    for dt in range(len(detectionData)):
                        if detectionData[dt][0] == sig[1]:
                            clusterID = clusters[dt]
        
                    writer.writerow({'Filename': currentFile, 
                                     'Minute': currentMinute,
                                     'LowFreq': lowFreq, 
                                     'HighFreq': highFreq, 
                                     'DistFreqs': distFreqs, 
                                     'NumPeaks': numPeaks,
                                     'Detection': detection,
                                     'Multiple': mSig,
                                     'Cluster': clusterID,
                                     'SD': (specMatrixMeanSD * SDThreshold)})

                    currentMinute = currentMinute + 1
        
                    with open("/".join([currentPath, "Extraction/BoatDetectionSummary.csv"]), "a") as BDSummary:
                        BDSummaryWriter = csv.DictWriter(BDSummary, fieldnames=fieldnamesSummary, delimiter=',')

                        #TODO: Adicionar funcionalidade de data para isto funcionar adequadamente
                        if (previousFile[0:10] == currentFile[0:10]):
                            currentIndexSpec = currentIndexSpec + 1
                        else:
                            currentIndexSpec = 0
                        previousFile = currentFile
                        
                        BDSummaryWriter.writerow({'Index': currentBDSummary,
                                                  'IndexSpec': currentIndexSpec,
                                                  'Filename': currentFile, 
                                                  'Date': (datetime.strptime(fileIdent, dateFormat24h)).strftime("%Y.%m.%d"),
                                                  'NumPeaks': numPeaks,
                                                  'Detection': detection,
                                                  'Multiple': mSig})
                    currentBDSummary = currentBDSummary + 1        

                ## Save signature of the file (Full file signature)        
                #sigFreq = []
                #lowFreq = 0
                #highFreq = 0
                #distFreqs = 0
                #numPeaks = 0
                #detection = 0
                #
                #previous = 0
                #for i in range(len(fileSig[:maxFreq])):    
                #    if (fileSig[previous] != 0) & (fileSig[i]==0):
                #        sigFreq.append(i)
                #    previous = i
                #
                #if len(sigFreq) > 0:
                #    lowFreq = sigFreq[0]
                #    highFreq = sigFreq[len(sigFreq)-1]
                #    distFreqs = highFreq - lowFreq
                #    numPeaks = len(sigFreq)
                #    if numPeaks > 0:
                #        detection = 1
                #
                #writer.writerow({'Filename': currentFile, 
                #                 'Minute': 100,
                #                 'LowFreq': lowFreq, 
                #                 'HighFreq': highFreq, 
                #                 'DistFreqs': distFreqs, 
                #                 'NumPeaks': numPeaks,
                #                 'Detection': detection,
                #                 'Multiple': mSig,
                #                 'Cluster': 0})
        
            numCurrentFile = numCurrentFile + 1
            if extractSpectrograms == False:
                numCurrentSpectrogram = numCurrentSpectrogram + 1

            timeFile1 = time.time()
            sumTimeFiles += (timeFile1-timeFile)
        except:
            print("#####################################################")
            print("#####################################################")
            print("### Error on file: %s" % currentFile)
            print("#####################################################")
            print("#####################################################")
            numCurrentFile = numCurrentFile + 1
            if extractSpectrograms == False:
                numCurrentSpectrogram = numCurrentSpectrogram + 1
            # Append the faulty filename 
            with open("/".join([currentPath, "Extraction/PROCESSING_ERROR_BoatDetection.log"]), "a") as myfile:
                myfile.write("   %s\n" % currentFile)

    ##############################################################################################################################
    ### EXTRACT SPECTROGRAMS #####################################################################################################
    ##############################################################################################################################    
    # Append the boat extraction errors
    with open("/".join([currentPath, "Extraction/PROCESSING_ERROR_BoatDetection.log"]), "a") as myfile:
        myfile.write("\n")
        myfile.write("### ERROR ON THE SPECTROGRAM GENERATION PROCESS ####\n")

    if extractSpectrograms == True:
        
        #Load an audio file with separated samplerate, left and right channels        
        for currentFile in currentFiles:
            try:
                timeSpec = time.time()
                
                if numCurrentSpectrogram <= len(currentFiles):
                    #numCurrentSpectrogram = numCurrentSpectrogram + 1
                    print("   Generating Spectrogram %s of %s (%s)" % (numCurrentSpectrogram+1, len(currentFiles), currentFile))
                boatDetection = []
                boatDetectionP = []        
                
                if currentFile[-4:] == 'flac':
                    fileIdent = currentFile[:-5]
                else:
                    fileIdent = currentFile[:-4]
                
                frames, rate = sf.read("/".join([currentPath,currentFile]))
            
                # Pego um canal apenas
                if (len(np.shape(frames)) == 1):
                    selChannel = frames
                else:
                    if (audioChannel == 0):
                        selChannel = frames[:,0]
                    else:
                        selChannel = frames[:,1]
                
                numMinutes = int(np.ceil((len(frames)/rate)/60))
            
                # Extract the spectrogram matrix (STFT)
                #sampleWindows = len(selChannel)/rate
                #specMatrix = np.zeros([rate/2, sampleWindows])
                #for i in range(sampleWindows):
                #    t1 = i*rate
                #    t2 = (i+1)*rate
                #    specMatrix[:,i] = calculate_spectrum(selChannel[t1:t2], rate)                            

                if numMinutes == 1:
                    specRate = 8192
                    sampleWindows = len(selChannel)/(specRate)
                    specMatrix = np.zeros([specRate/2, sampleWindows])
                    for i in range(sampleWindows):
                        t1 = i*specRate
                        t2 = (i+1)*specRate
                        specMatrix[:,i] = calculate_spectrum(selChannel[t1:t2], specRate)
                else:
                    # Extract the spectrogram matrix (STFT)
                    specRate = rate
                    sampleWindows = len(selChannel)/specRate
                    specMatrix = np.zeros([specRate/2, sampleWindows])
                    for i in range(sampleWindows):
                        t1 = i*specRate
                        t2 = (i+1)*specRate
                        specMatrix[:,i] = calculate_spectrum(selChannel[t1:t2], specRate)

                freqs1 = librosa.core.fft_frequencies(sr=rate, n_fft=(specRate)+1)
                maxFreq1=0
                while (freqs1[maxFreq1] < maximumFrequency):
                    maxFreq1 += 1

                # CREATE AN AUDIO SPECTROGRAM OF THE ENTIRE FILE
                spec_width = 1200
                spec_height = 600
                speccmap = plt.cm.gray_r
                specMatrixDB = librosa.amplitude_to_db(specMatrix[:maxFreq1,:], ref=1.0, top_db=topDB)
                #specMatrixPWR = np.power(specMatrixDB, 2)
                specMatrixDB1 = np.flipud(specMatrixDB)
                #specMatrixDB1 = np.flipud(specMatrixPWR)             
                norm = plt.Normalize(vmin=specMatrixDB1.min(), vmax=specMatrixDB1.max())    
                #norm = plt.Normalize(vmin=normalizationMin, vmax=normalizationMax)
                image = speccmap(norm(specMatrixDB1))        
                #image = specMatrixDB1
                newImage = scipy.misc.imresize(image, (spec_height, spec_width))
                plt.imsave("%s/Extraction/BoatSpectrogram/%s.png" % (currentPath, fileIdent), newImage)

                with open("%s/Extraction/BoatSpectrogram/%s.csv" % (currentPath, fileIdent), 'wb') as csvfile:
                    fieldnames = ['sampleWindow', 'audioSeconds', 'maxFreq']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')            
                    writer.writeheader()
                    writer.writerow({'sampleWindow': 60,
                                     'audioSeconds': numMinutes*60,
                                     'maxFreq': freqs1[maxFreq1]})                  
                
                numCurrentSpectrogram = numCurrentSpectrogram + 1
                
                timeSpec1 = time.time()
                sumTimeSpectrograms += (timeSpec1-timeSpec)
                
            except:
                print("#####################################################")
                print("#####################################################")
                print("### Error on file: %s" % currentFile)
                print("#####################################################")
                print("#####################################################")
                numCurrentSpectrogram = numCurrentSpectrogram + 1
                # Append the faulty filename 
                with open("/".join([currentPath, "Extraction/PROCESSING_ERROR_BoatDetection.log"]), "a") as myfile:
                    myfile.write("   %s\n" % currentFile)

    print("###############################################################")
    print("### Process Completed #########################################")
    print("###############################################################")
    timeEnd = time.time()
    timeToProcess = timeEnd-timeStart
    meanTimeToProcess = sumTimeFiles/numFiles
    meanTimeToProcessSpec = sumTimeSpectrograms/numFiles
    print("Time to process %s files: %.2f seconds" % (numFiles, timeToProcess))
    print("Mean time to process each file: %.2f seconds" % (meanTimeToProcess))
    print("Mean time to generate each spectrogram: %.2f seconds" % (meanTimeToProcessSpec))
    print(" ")

@app.route("/start_extraction_boat", methods=['GET', 'POST'])
def startExtractionBoat():    
    global currentPath
    global numCurrentFile
    global numCurrentSpectrogram
    global numFiles    
    global extractSpectrograms24h
    global numCurrentSpectrogram24h
    
    numCurrentFile = 0
    numCurrentSpectrogram = 0
    numCurrentSpectrogram24h = 0

    extractBoatSignatures()
    
    if (extractSpectrograms24h):
        extract24hSpectrogram()
        
    return jsonify({"status": "Extraction finished"})

@app.route("/set_data_boat", methods=['GET', 'POST'])
def setDataBoat():    
    global SDThreshold
    global maximumFrequency
    global plotImages
    global extractSpectrograms
    global extractSpectrograms24h
    global maximumFrequency24h
    global sampleWindow24h
    global audioChannel
    global dateFormat24h

    parameters = request.form.to_dict()

    SDThreshold = float(parameters['threshold'])
    maximumFrequency = int(parameters['maximumFrequency'])
    plotImagesInt = 0 # = int(parameters['plotImages'])
    extractSpectrogramsInt = int(parameters['extractSpectrograms'])
    extractSpectrograms24hInt = int(parameters['extract24HSpectrograms'])
    maximumFrequency24h = int(parameters['maximumFrequency24h'])
    sampleWindow24h = int(parameters['timesample24h'])    
    audioChannel = int(parameters['audiochannel'])
    dateFormat24h = parameters['fileDateTimeMask']
    
    if plotImagesInt == 1:    
        plotImages = True
    else:
        plotImages = False
    
    if extractSpectrogramsInt == 1:    
        extractSpectrograms = True
    else:
        extractSpectrograms = False    

    if extractSpectrograms24hInt == 1:    
        extractSpectrograms24h = True
    else:
        extractSpectrograms24h = False   

    return jsonify({"status": "Settings defined"})

###########################################################################################################
### Algoritmo de detecção por variação da amplitude e desvio padrão (Minuto)
###########################################################################################################
def extract24hSpectrogram():
    global SDThreshold
    global timeToProcess
    global meanTimeToProcess
    global meanTimeToProcessSpec
    global plotImages
    global extractSpectrograms
    global timeSample
    global dateFormat24h
    global maximumFrequency24h
    global sampleWindow24h
    global audioChannel
    
    #TODO: Pegar estas variáveis do nome do arquivo
    global year
    global startDay
    global endDay
    global month

    spec_width = 800
    spec_height = 300   
    sumTimeFiles = 0
    sumTimeSpectrograms = 0
    timeStart = time.time()
    print("###############################################################")
    print("### Process Started ###########################################")
    print("###############################################################")

    ###########################################################################################################
    ### Variáveis de configuração 
    ###########################################################################################################
    # TODO: Passar por parâmetro essas variáveis    
    topDB = 60
    W = 13
    H = 7
    #audioChannel = 0
    #spec_width = 1200
    #spec_height = 600
    #timeSample = 60 # Seconds
    windowFFT = 2048*2

    # TODO: Tentar extrair esas informacoes direto do nome dos arquivos... criar uma lista de datas a partir dos nomes.
    #year = 2015
    #startDay = 3
    #endDay = 4
    #month = 2

    global currentPath
    global numCurrentFile24h
    global numCurrentSpectrogram24h
    global numFiles24h
    global numFilesSpec24h
    global numCurrentFiles24h
    global numCurrentFilesSpec24h
    global maximumFrequency
    global specMatrixDB
    
    #dateTimeFormat = "%Y.%m.%d_%H.%M.%S"
    
    ###########################################################################################################
    ### Create the necessary directories
    ###########################################################################################################
    outputPath = "/".join([currentPath, "Extraction"])
    if (os.path.isdir(outputPath) == False):
        os.system("mkdir %s" % (outputPath))
    
    outputPath = "/".join([currentPath, "Extraction", "24hSpectrogram"])
    if (os.path.isdir(outputPath) == False):
        os.system("mkdir %s" % (outputPath))
    
    # Reset the extraction error report
    with open("/".join([currentPath, "Extraction/PROCESSING_ERROR_24hSpectrogram.log"]), "w") as myfile:
        myfile.write("####################################################\n")
        myfile.write("### ERROR REPORT - Seecology                     ###\n")
        myfile.write("####################################################\n")    
    
    ###########################################################################################################
    ### List all the allowed files on the directory
    ###########################################################################################################
    wdir = os.listdir("%s" % (currentPath))
    wdir.sort()
    currentFiles = [arq for arq in wdir if (arq.lower().endswith(".wav") | 
                                            arq.lower().endswith(".raw") | 
                                            arq.lower().endswith(".flac"))]
    currentFiles.sort()
    
    #Calcular o número de dias
    #Get first and last dates
    if currentFiles[0][-4:] == 'flac':
        day1 = currentFiles[0][:-5]
        day2 = currentFiles[len(currentFiles)-1][:-5]
    else:
        day1 = currentFiles[0][:-4]
        day2 = currentFiles[len(currentFiles)-1][:-4]
    
    #dateFormat24h = "015089_%Y%m%d_%H%M%S"
    dateTimeFormat = "%Y.%m.%d_%H.%M.%S"
    
    fromTime = "%s_00.00.00" % ((datetime.strptime(day1, dateFormat24h)).strftime("%Y.%m.%d"))
    toTime =   "%s_23.59.59" % ((datetime.strptime(day2, dateFormat24h)).strftime("%Y.%m.%d"))
    
    fromTimeObj = datetime.strptime(fromTime, dateTimeFormat)
    toTimeObj = datetime.strptime(toTime, dateTimeFormat)
    delta = toTimeObj - fromTimeObj
    numFiles24h = delta.days + 1
    #print delta.days + 1

    datelist = pd.date_range(fromTimeObj, toTimeObj).tolist()
    
    #Percorrer esse vetor
    
    #Montar o fromTimeObj e toTimeObj usando o dateFormat24h e o nome do arquivo
        # FormatDate(ParseDate(nome do arquivo, dateFormat24h))
    
    
    # Get the number of spectrograms to generate and the number of files needed
    #numFiles24h = len(range(startDay,endDay+1))    
    numFilesSpec24h = 0    
    #for d in range(startDay,endDay+1):
    for d in datelist:
        #TODO: Tornar esta parte dinâmica, seguindo o padrão de data passado em dateTimeFormat
        #day = "%d.%02d.%02d" % (year,month, d)
        #fromTime = "%s_00.00.00" % day
        #toTime =   "%s_23.59.59" % day
        #fromTimeObj = datetime.strptime(fromTime, dateTimeFormat)
        #toTimeObj = datetime.strptime(toTime, dateTimeFormat)
        
        for f in currentFiles:
            if f[0] != ".": # Elimino arquivos ocultos gerados por alguns sistemas/softwares
                if f[-4:] == 'flac':
                    fileIdent = f[:-5]
                else:
                    fileIdent = f[:-4]
                #if ((fromTimeObj <= datetime.strptime(fileIdent, dateFormat24h)) & (toTimeObj >= datetime.strptime(fileIdent, dateFormat24h)) ):
                if (d.strftime("%Y.%m.%d") == (datetime.strptime(fileIdent, dateFormat24h)).strftime("%Y.%m.%d") ):
                    numFilesSpec24h = numFilesSpec24h + 1    

    ##############################################################################################################################
    ### EXTRACT 24H SPECTROGRAM ##################################################################################################
    ##############################################################################################################################    
    # Append the spectrogram generation errors
    with open("/".join([currentPath, "Extraction/PROCESSING_ERROR_24hSpectrogram.log"]), "a") as myfile:
        myfile.write("\n")    
        myfile.write("### ERROR ON THE 24H SPECTROGRAM GENERATION ################\n")

    numCurrentFiles24h = 0
    numCurrentFilesSpec24h = 0
    #Percorro cada data do mês/ano selecionado        
    timeTotalStart = time.time()
    #for d in range(startDay,endDay+1):
    
    for d in datelist:        
        print("### Day %s " % d.strftime("%Y.%m.%d"))
        timeStart = time.time()        
               
        #TODO: Tornar esta parte dinâmica, seguindo o padrão de data passado em dateTimeFormat
        #day = "%d.%02d.%02d" % (year,month, d)
        #fromTime = "%s_00.00.00" % day
        #toTime =   "%s_23.59.59" % day
        #fromTimeObj = datetime.strptime(fromTime, dateTimeFormat)
        #toTimeObj = datetime.strptime(toTime, dateTimeFormat)
        
        firstOk = False
        firstFile = ""
        lastFile = ""
        
        specMatrixMean = []    

        for f in currentFiles:
            try:
                if f[0] != ".": # Elimino arquivos ocultos gerados por alguns sistemas/softwares
                    if f[-4:] == 'flac':
                        fileIdent = f[:-5]
                    else:
                        fileIdent = f[:-4]
                    #if ((fromTimeObj <= datetime.strptime(fileIdent, dateFormat24h)) & (toTimeObj >= datetime.strptime(fileIdent, dateFormat24h)) ):
                    if (d.strftime("%Y.%m.%d") == (datetime.strptime(fileIdent, dateFormat24h)).strftime("%Y.%m.%d") ):
                        print("   Processing file %s of %s (%s)" % (numCurrentFilesSpec24h+1, numFilesSpec24h, f))            
                
                        # Save the first and last file names to extract date/time
                        if firstOk == False:
                            firstFile = fileIdent
                            firstOk = True
                        lastFile = fileIdent
                        
                        frames, rate = sf.read("/".join([currentPath,f]))
                        
                        if (len(np.shape(frames)) == 1):    #Test if audio is mono
                            selChannel = frames                            
                        else:
                            if (audioChannel == 0):         #Select channel if stereo
                                selChannel = frames[:,0]
                            else:
                                selChannel = frames[:,1]
                                
                        # Extract the spectrogram matrix (STFT)
                        specRate = windowFFT
                        sampleWindows = len(selChannel)/specRate
                        specMatrix = np.zeros([specRate/2, sampleWindows])
                        for i in range(sampleWindows):
                            t1 = i*specRate
                            t2 = (i+1)*specRate
                            fft = np.fft.fft(selChannel[t1:t2], norm=None)    
                            specMatrix[:,i] = abs(fft[:specRate/2])

                        sampleWindowsOnFile = int(np.floor(len(frames)/rate)/(sampleWindow24h))
                        samplesPerWindow = int(np.shape(specMatrix)[1]/sampleWindowsOnFile)

                        for i in range(sampleWindowsOnFile):
                            t1 = i*samplesPerWindow
                            t2 = (i+1)*samplesPerWindow
                            specMatrixMean.append(np.mean(specMatrix[:,t1:t2], axis=1))
                                                
                        numCurrentFilesSpec24h = numCurrentFilesSpec24h + 1
            except:
                print("#####################################################")
                print("#####################################################")
                print("### Error on file: %s" % f)
                print("#####################################################")
                print("#####################################################")
                numCurrentFiles24h = numCurrentFiles24h + 1
                with open("/".join([currentPath, "Extraction/PROCESSING_ERROR_24hSpectrogram.log"]), "a") as myfile:
                    myfile.write("   %s\n" % f)

        numCurrentFiles24h = numCurrentFiles24h + 1

        freqs = librosa.core.fft_frequencies(sr=rate, n_fft=(specRate)+1)
        if maximumFrequency24h != 0:            
            maxFreq=1
            while (freqs[maxFreq] < maximumFrequency24h):
                maxFreq += 1
        else: 
            maxFreq = len(freqs)
        maxFreq = maxFreq - 1
        
        # Convert the specMatrixMean tuple to an image matrix
        print("Saving spectrogram")
        #resImage = np.zeros(shape=(np.shape(specMatrixMean)[1], np.shape(specMatrixMean)[0]))
        resImage = np.zeros(shape=(maxFreq, np.shape(specMatrixMean)[0]))
        for i in range(np.shape(specMatrixMean)[0]):
            resImage[0:maxFreq,i] = specMatrixMean[i][0:maxFreq]
        
        #Cores: Accent, gist_earth, gist_ncar (Perfeito), terrain (Ótimo), nipy_spectral, viridis    
        #speccmap = plt.cm.terrain
        #speccmap = plt.cm.gist_ncar
        #speccmap = plt.cm.nipy_spectral
        specMatrixDB = librosa.amplitude_to_db(resImage, ref=1.0, top_db=topDB)
        specMatrixDB1 = np.flipud(specMatrixDB)
        #specMatrixPWR = librosa.db_to_power(resImage, ref=1.0)
        #specMatrixDB1 = np.flipud(specMatrixPWR)
        #norm = plt.Normalize(vmin=normalizationMin, vmax=normalizationMax)
        
        #DB Mean of the file
        specMatrixDBMean = np.mean(specMatrixDB, axis=0)
        specMatrixDBMeanSmooth = smooth(specMatrixDBMean, window_len=128)
        np.savetxt("%s/Extraction/24hSpectrogram/SpecPeriod_%s_DBMean.csv" % (currentPath, d.strftime("%Y.%m.%d")), specMatrixDBMeanSmooth)
        
        #image = speccmap(norm(specMatrixDB1))        
        #image = speccmap(specMatrixDB1)
        #image = norm(specMatrixDB1)
        image = specMatrixDB1
    
        newImage = scipy.misc.imresize(image, (spec_height, spec_width))
        plt.imsave("%s/Extraction/24hSpectrogram/SpecPeriod_%s.png" % (currentPath, d.strftime("%Y.%m.%d")), newImage)
        
        # Save file information for the visualization
        print("Saving spectrogram info")
        with open("%s/Extraction/24hSpectrogram/SpecPeriod_%s.csv" % (currentPath, d.strftime("%Y.%m.%d")), 'wb') as csvfile:
            fieldnames = ['from', 'to', 'maxFreq']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')            
            writer.writeheader()
    
            auxFrom = datetime.strptime(firstFile, dateFormat24h)
            auxTo   = datetime.strptime(lastFile, dateFormat24h)
    
            #TODO: Confirmar se a data final foi acrescida do tamanho do último arquivo e se não deu erro...
            writer.writerow({'from': "{:%Y-%m-%d %H:%M:%S}".format(auxFrom),
                             'to': "{:%Y-%m-%d %H:%M:%S}".format(auxTo + timedelta(seconds=int(len(frames)/rate))), 
                             'maxFreq': np.ceil(freqs[maxFreq])})
    
        numCurrentSpectrogram24h = numCurrentSpectrogram24h + 1    
        
        timeEnd = time.time()
        timeToProcess = timeEnd-timeStart
        print("Time to process: %.2f seconds" % (timeToProcess))
    
    print("###############################################################")
    print("### Process Completed #########################################")
    print("###############################################################")
    timeTotalEnd = time.time()
    timeToProcess = timeTotalEnd-timeTotalStart
    print("Time to process %s 24h spectrograms: %.2f seconds" % (numFiles24h, timeToProcess))
    print(" ")

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
    print("Processing directory: %s" % jsonData['GENERAL_mainPath'])

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

    if os.path.exists("%s/Extraction/features_labels.csv" % (jsonData['GENERAL_mainPath'])):
        os.remove("%s/Extraction/features_labels.csv" % (currentPath))
        df = pd.DataFrame()
        df.to_csv("%s/Extraction/features_labels.csv" % (currentPath), sep=";")

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
                print("    Extracting MP3 file to WAV: %s" % ( f ))
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
        print("### EXTRACTING SPECTROGRAM ####")
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


    print("#### EXTRACTION PROCESS FINISH ####")
      
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
    global numFiles24h
    global numFilesSpec24h
    global numCurrentFiles24h
    global numCurrentFilesSpec24h

    return jsonify({"numCurrentFile": numCurrentFile, 
                    "numCurrentSpectrogram": numCurrentSpectrogram, 
                    "numFiles": numFiles,
                    "numFiles24h": numFiles24h,
                    "numFilesSpec24h": numFilesSpec24h,
                    "numCurrentFiles24h": numCurrentFiles24h,
                    "numCurrentFilesSpec24h": numCurrentFilesSpec24h})

@app.route('/addLabels', methods=['POST'])
def addLabels():
    global currentPath
    
    labelName = request.form.get('labelName')
    labelList = request.form.getlist('labelList')
        
    #break the labelList by ";"
    labelList1 = labelList[0].split(";")

    dfFeatures = pd.read_csv("%s/Extraction/features_group.csv" % (currentPath))

    # Load the file if exists, otherwise, create it
    df = pd.DataFrame()
    df1 = pd.DataFrame()
    if os.path.exists("%s/Extraction/features_labels.csv" % (currentPath)):
        df = pd.read_csv("%s/Extraction/features_labels.csv" % (currentPath))
    else:
        df.to_csv("%s/Extraction/features_labels.csv" % (currentPath), sep=";")

    if (df.empty):
        df = pd.DataFrame(columns=[labelName])
        for i in range( len(dfFeatures.index) ):
            df = df.append({labelName: 0}, ignore_index=True)
        
        for i in range( len(dfFeatures.index) ):
            for j in range( len(labelList1) ):
                auxLL = labelList1[j].split("|")
                if (auxLL[0] == dfFeatures.values[i][0]) & (float(auxLL[1]) == dfFeatures.values[i][3]):
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
                if (auxLL[0] == dfFeatures.values[i][0]) & (float(auxLL[1]) == dfFeatures.values[i][3]):                
                    df1.values[i][0] = 1

    # Add the column
    df = df.join(df1)
        
    # Finally, save the CSV        
    df.to_csv("%s/Extraction/features_labels.csv" % (currentPath), sep=",", index=None)

    return jsonify({"status": "Labels added"})

@app.route('/removeLabels', methods=['POST'])
def removeLabels():
    global currentPath
    
    labelName = request.form.get('labelName')

    df = pd.DataFrame()
    df = pd.read_csv("%s/Extraction/features_labels.csv" % (currentPath))
    
    if labelName in df.columns:
        df.drop(labelName, axis=1, inplace=True)
    
    df.to_csv("%s/Extraction/features_labels.csv" % (currentPath), sep=",", index=None)

    return jsonify({"status": "Labels removed"})

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
    
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

    app.run(port=serverport, debug=True, use_reloader=False)
    
    
# Para matar um servidor que ficou ativo:        
#    ps -fA | grep python 
#    kill <id>