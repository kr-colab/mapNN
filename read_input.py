# helper utils for reading in data

import numpy as np
import os

# reads in a list
def read_list(path, trans=None):
    collection = []
    with open(path) as infile:
        for line in infile:
            newline = line.strip().split()
            if len(newline) == 1:
                newline = newline[0]
            if trans is not None:
                newline = list(map(trans,newline))
            collection.append(newline)
    return collection

# convert list to keys in a dictionary, values all zero
def list2dict(mylist):
    collection = {}
    for i in range(len(mylist)):
        collection[mylist[i]] = 0
    return collection

# read in the training targets and input paths from a preprocessed, hierarchical folder
def dict_from_preprocessed(path, prediction=False):
    targets,genos,locs,counter={},{},{},0
    if prediction is not True:
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
                    
    # getting exactly one preprocessed dataset per tree sequence (instead of multiple) 
    else:
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
