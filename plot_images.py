import numpy as np
import sys, os
from geopy import distance
import random
from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
from PIL import ImageFont
from numpy import asarray
from skimage.measure import block_reduce
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm,colors
import cv2


# just what it sounds like
def cookie_cutter(data, outline, fill=None, fxn=None):
    # if 2d, add temporary dim    
    if len(data.shape) == 2:
        data = np.reshape(data,(data.shape[0],data.shape[1],1))

    # apply mask
    if fill is not None:
        for i in range(data.shape[0]):       
            for j in range(data.shape[1]):   
                if outline[i,j] == 0:
                    data[i,j,:] = fill

    # apply log or other fxn
    if fxn is not None:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if outline[i,j] == 1:
                    data[i,j,:] = fxn(data[i,j,:])

    # remove extra dim if it was 2d 
    if data.shape[2] == 1:
        data = np.reshape(data,(data.shape[0],data.shape[1])) 
    return data


# read PNG
def read_map(png, width):
    data=np.load(png)
    if data.shape[0] % width > 0 and width % data.shape[0] > 0:
        print("make sure old map size is divisible by new size")
        exit()
    factor = int(float(data.shape[0]) / float(width))
    if factor > 1: # compress
        data = block_reduce(data, (factor,factor,1), np.mean)
    elif factor < 1: # blow up
        factor = int(1. / factor)
        data = data.repeat(factor, axis=0).repeat(factor, axis=1)
    return data


# reading black and white PNG of the habitat
def read_habitat_map(habitat_map, target_width):
    outline=Image.open(habitat_map)
    temp=asarray(outline)
    outline = np.copy(temp)
    outline = outline.astype(float)

    # compress to target dims
    rat = int(round(outline.shape[0]/target_width))
    outline = block_reduce(outline, block_size=(rat,rat,1), func=np.mean)

    # assign each pixel to land or water
    mask = np.zeros((outline.shape[0],outline.shape[1]))
    for i in range(outline.shape[0]):
        for j in range(outline.shape[1]):
            mean_val = np.mean(outline[i,j,0:3])
            if mean_val < (255.0/2.0):
                mask[i,j] = 1

    return mask


# plotting fxns                                                                                                   
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
def get_concat_bar(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))#, (255,255,255))                                  
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, im1.height-im2.height))
    return dst
def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


# grab min and max values for rescaling sigma
def get_min_max(the_map, habi_map=None):
    if habi_map is None:
        # # (this approach won't work until you plot the heatmap in log scale)
        # mean_sigma = np.mean(the_map[:,:,0])                                                                    
        # mean_k = np.mean(the_map[:,:,1])                                                                        
        # min_sigma = mean_sigma / 10                                                                             
        # max_sigma = mean_sigma * 10                                                                             
        # min_k = mean_k / 10                                                                                     
        # max_k = mean_k * 10                                                                                     
        #                                                                                                         
        min_sigma = np.min(the_map[:,:,0])
        max_sigma = np.max(the_map[:,:,0])
        min_k = np.min(the_map[:,:,1])
        max_k = np.max(the_map[:,:,1])
    else: # find range of sigma, and range of K inside the habitat for empirical interpretation
        min_sigma,max_sigma,min_k,max_k=1e16,0,1e16,0  # defaults                                                 
        for j in range(the_map.shape[0]):
            for k in range(the_map.shape[1]):
                if habi_map[j,k] == 1:
                    min_sigma = np.min([min_sigma,the_map[j,k,0]])
                    max_sigma = np.max([max_sigma,the_map[j,k,0]])
                    min_k = np.min([min_k,the_map[j,k,1]])
                    max_k = np.max([max_k,the_map[j,k,1]])
    return min_sigma,max_sigma,min_k,max_k


# plot heat map
def heatmap(demap, plot_width, tmpfile, cb_params=None, habitat_map_plot=None, habitat_border=None, locs=None):
    # plot map
    img = Image.fromarray(demap)
    img = img.convert('L')
    img = img.resize((plot_width,plot_width))
    img.save(tmpfile)
    img = cv2.imread(tmpfile, cv2.IMREAD_GRAYSCALE)
    colormap = plt.get_cmap('coolwarm_r')
    img = (colormap(img) * 2**16).astype(np.uint16)[:,:,:3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if locs is not None:
        for l in range(locs.shape[1]):
            img = cv2.circle(img, (locs[0,l],locs[1,l]), radius=3, color=(0,0,0), thickness=1)
    if habitat_map_plot is not None:
        img = cookie_cutter(img, habitat_map_plot, fill=65535)
    cv2.imwrite(tmpfile, img) # write temp file                                                               
    img = Image.open(tmpfile) # read as PIL again                                                             
    if habitat_border is not None:
        im_border = Image.open(habitat_border)
        im_border = im_border.resize((plot_width,plot_width))
        img.paste(im_border, (0,0), ImageOps.invert(ImageOps.grayscale(im_border)))
    img = ImageOps.expand(img, border=10, fill='white')

    # color bar
    if cb_params is not None:
        fig = plt.figure()
        ax = fig.add_axes([0, 0.05, 0.06, 1]) # left, bottom, width, height                                       
        norm = colors.Normalize(cb_params[0],cb_params[1])
        colormap = plt.get_cmap('coolwarm_r') # _r for reverse (don't ask)                                        
        cb = mpl.colorbar.ColorbarBase(ax, cm.ScalarMappable(norm=norm, cmap=colormap))#, label=r'$\sigma$')      
        plt.savefig(tmpfile, bbox_inches='tight')
        plt.close()
        fig.clear()
        cb = Image.open(tmpfile)
        white_background = Image.new("RGB", (cb.size[0], 50), (255, 255, 255)) # adding some white space above bar
        cb  = get_concat_v(white_background, cb)
        cb = cb.resize((75,520))
        img = get_concat_bar(img, cb)
        os.remove(tmpfile)

        # text label
        font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans-Oblique.ttf')               
        myfont = ImageFont.truetype(font_path, size=24)                                       
        t = ImageDraw.Draw(img) # (this syntax doesn't make any sense but it works?) 
        t.text((540, 26), cb_params[2], fill=(0,0,0), font=myfont) # sigma                        

    return img
