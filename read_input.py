# helper utils for reading in data

import numpy as np
import os

# reads a list of filepaths, stores in list
def read_list(path):
    collection = []
    with open(path) as infile:
        for line in infile:
            newline = line.strip()
            collection.append(newline)
    return collection

# read table of lat+long coords, store in list
def read_locs(path):
    collection = []
    with open(path) as infile:
        for line in infile:
            newline = line.strip().split()
            newline = list(map(float,newline))
            collection.append(newline)
    
    return collection

# convert list to keys in a dictionary, values all zero
def list2dict(mylist):
    collection = {}
    for i in range(len(mylist)):
        collection[mylist[i]] = 0
    return collection

# parse tree sequence provenance for sigma and map width
def parse_provenance(ts, param):
    prov = str(ts.provenance(0)).split()
    for i in range(len(prov)):
        if param+"=" in prov[i]:
            val = float(prov[i].split("=")[1].split("\"")[0])
            break
    return(val)
            
# read in the training targets and input paths from a preprocessed, hierarchical folder
def dict_from_preprocessed(path):
    targets,genos,locs,counter={},{},{},0
    for root, subdir, files in os.walk(path+"/Genos/"): # everything under Genos/
        if subdir == []: # excluding the Maps/ folder itself
            for f in files:
                genopath = os.path.join(root, f)                     
                mappath = "Maps".join(genopath.rsplit("Genos",1))   
                mappath = "target".join(mappath.rsplit("genos",1))
                locpath = "Locs".join(genopath.rsplit("Genos",1))     
                locpath = "locs".join(locpath.rsplit("genos",1))                   
                targets[counter] = mappath
                genos[counter] = genopath
                locs[counter] = locpath
                counter += 1
    return targets,genos,locs

# getting exactly one preprocessed dataset per tree sequence (instead of multiple)
def preds_from_preprocessed(path):
    targets,genos,locs,counter={},{},{},0
    for root, subdir, files in os.walk(path+"/Genos/1/"): # note the "1"
        if subdir == []: # excluding the Maps/ folder itself                           
            for f in files:
                genopath = os.path.join(root, f)                                       
                mappath = genopath.replace("Genos", "Maps").replace("genos","target") 
                locpath = genopath.replace("Genos", "Locs").replace("genos","locs")    
                targets[counter] = mappath
                genos[counter] = genopath
                locs[counter] = locpath
                counter += 1
    return targets,genos,locs









