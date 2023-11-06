
# creates a 2-channel map (.npy) with random spatial segments

# e.g., python create_maps.py --help  # help message
# e.g., python create_maps.py --out temp1.npy --seed 123 --w 50 --min_c1 0.2 --max_c1 3 --min_c2 4 --max_c2 4
# e.g., python create_maps.py --out temp1.npy --seed 123 --w 500 --png  # output optional PNG 


import argparse
import numpy as np
import PIL.Image as Image
import random
import sys
from scipy.stats import loguniform


parser=argparse.ArgumentParser()
parser.add_argument("--out",help="output path (with or without .npy)", required=True)
parser.add_argument("--w",help="width of square map", type=int, default=50)
parser.add_argument("--seed",help="random number seed", type=int)
parser.add_argument("--min_c1",help="min for channel-1", type=float, default=0.2)
parser.add_argument("--max_c1",help="max for channel-1", type=float, default=3)
parser.add_argument("--min_c2",help="min for channel-2", type=float, default=4)
parser.add_argument("--max_c2",help="max for channel-2", type=float, default=40)
parser.add_argument("--max_degree",help="maximum degree for polynomial function (default=3)", type=int, default=3)
parser.add_argument("--png",help="output optional red & blue PNG for visualization, alongside the primary output (.npy)", action="store_true")
args=parser.parse_args()


def random_points():
    degree = random.randint(0, args.max_degree+1) # includes lower bound, excludes upper bound. 0=flat line, 1=sloped line, 2=curves, 3=up and downs.
    num_points = degree+1
    x_pos = [random.randint(0, args.w-1) for p in range(0, num_points)]
    y_pos = [random.randint(0, args.w-1) for p in range(0, num_points)]
    return degree,x_pos,y_pos


def psuedo_log_uniform(min_val, max_val):
    if min_val == 0:
        sys.stderr.write("min_val=0; drawing from pseudo log-uniform with +1\n")
        x = loguniform.rvs(min_val+1, max_val+1) - 1
    else:
        x = loguniform.rvs(min_val, max_val)
    return x


def fit_curve(degree, x_pos, y_pos):
    model1 = np.poly1d(np.polyfit(x_pos, y_pos, degree))
    xs = np.linspace(0, args.w, args.w)
    ys = model1(xs)
    
    # count segments
    num_segments = 1
    intercepts = []
    for i in ys:
        if i < 0:
            intercepts.append(0)
        elif i > args.w:
            intercepts.append(2)
        else:
            intercepts.append(1)
            
    # loop back through and count x intercepts
    current_pos = int(intercepts[0])
    previous_intercept = int(intercepts[0])
    for i in range(1, args.w):
        if previous_intercept == 0 and intercepts[i] == 2:
            num_segments += 1
            previous_intercept = int(intercepts[i])
        elif previous_intercept == 2 and intercepts[i] == 0:
            num_segments += 1
            previous_intercept = int(intercepts[i])
        elif current_pos == 1 and intercepts[i] != 1:
            num_segments += 1            
        current_pos = int(intercepts[i])

    # y intercept
    if intercepts[-1] == 1:
        num_segments += 1
    
    return num_segments,intercepts,ys


def assign_values(num_segments, min_val, max_val):    
    values = []
    mag = np.random.uniform(0, max_val-min_val) # this will represent the RANGE of values. I want this to be uniform, I think.
    start = psuedo_log_uniform(min_val, max_val-mag)
    end = start + mag
    values.append(start)
    values.append(end)
    for i in range(num_segments-2):
        values.append(np.random.uniform(start, end))
    np.random.shuffle(values)
    values = list(values)
    values.append(end)
    return values


def map_segments(num_segments,intercepts):
    segment_map = []
    if num_segments == 1:
        segment_map.append()
    else:
        internal_loops = 0
        if intercepts[0] != 1:
            segment_map.append([0]) # one segment
        else:
            segment_map.append([0,1]) # two segments
        for n in range(1,args.w):
            if intercepts[n] != 1: # one segment for this column
                if len(segment_map[n-1]) == 1: # if the previous column was one segment too, then same segment
                    if intercepts[n] == intercepts[n-1]: 
                        segment_map.append( list(segment_map[n-1]) )
                    else: # this is for very steep curves that change segments within a pixel
                        segment_map.append( [ segment_map[n-1][0]+1+internal_loops] )     
                        if internal_loops > 0:
                            internal_loops -= 1 # used up a loop                    
                else: # previous column was two segments, consolidating down to one segment
                    if segment_map[0] == segment_map[n-1]: # started with y intercept, depends on slope
                        if intercepts[n] == 0:
                            segment_map.append([segment_map[n-1][1]])
                        if intercepts[n] == 2:
                            segment_map.append([segment_map[n-1][0]])
                    else:
                        intercepts_copy = list(intercepts[0:n])
                        while 1 in intercepts_copy:
                            intercepts_copy.remove(1)
                        if len(intercepts_copy) > 0:
                            previous_intercept = int(intercepts_copy[-1])
                        else:
                            previous_intercept = None                        
                        if intercepts[n] != previous_intercept and previous_intercept != None:
                            segment_map.append([segment_map[n-1][1]])
                        else: # this is the interesting case of a loop inside the map
                            segment_map.append([segment_map[n-1][0]])
                            internal_loops += 1
            else: # two segments in this column
                if len(segment_map[n-1]) == 2: 
                    segment_map.append(list(segment_map[n-1]))
                else: # previous column one segment, introducing new segment, here
                    segment_map.append([segment_map[n-1][0], segment_map[n-1][0]+1+internal_loops])
                    if internal_loops > 0:
                        internal_loops -= 1 # used up a loop
    return(segment_map)


def make_mat(segment_map, values, ys):
    mat = np.zeros((args.w,args.w,1))
    for m in range(args.w):                                
        previous_pos = None 
        first_segment = True
        for n in range(0, args.w):
            my_segments = segment_map[n]
            crossed_the_line = False

            # getting position relative to the curve
            if m < ys[n]:
                current_pos = "below"
            else:
                current_pos = "above"
            if current_pos != previous_pos:
                if previous_pos != None:
                    crossed_the_line = True
                previous_pos = str(current_pos)

            # grab appropriate segment
            if len(my_segments) == 1:
                current_segment = int(my_segments[0])
                mat[m,n] = values[current_segment]
                if first_segment == True: # just reset this, was only important for two segments
                    first_segment = False
            else:
                if first_segment == True:
                    if current_pos == "below":
                        current_segment = int(my_segments[0])
                        mat[m,n] = values[current_segment]
                    elif current_pos == "above":
                        current_segment = int(my_segments[1])
                        mat[m,n] = values[current_segment]
                    first_segment = False
                else:
                    if crossed_the_line == False:
                        mat[m,n] = values[current_segment]
                    else:
                        segments_copy = list(my_segments)
                        segments_copy.remove(current_segment)
                        current_segment = int(segments_copy[0])
                        mat[m,n] = values[current_segment]
                        crossed_the_line = False # reset
                        
    return mat


def flip(mat):
    if random.randint(0,2) == 0:
        mat = np.flip(mat, axis=0)
    if random.randint(0,2) == 0:
        mat = np.flip(mat, axis=1)
    if random.randint(0,2) == 0:
        mat = np.rot90(mat)
    return mat


def make_png(s_mat, k_mat):

    # rescale to 0,1                                       
    s_mat = (s_mat-args.min_c1) / (args.max_c1-args.min_c1)
    k_mat = (k_mat-args.min_c2) / (args.max_c2-args.min_c2)

    # scale to 0,255                                       
    s_mat *= 255
    k_mat *= 255

    out = np.concatenate([
        k_mat,
        np.full((args.w, args.w, 1), 0, dtype='uint8'),
        s_mat,
        np.full((args.w, args.w, 1), 255, dtype='uint8'),
    ], axis=-1)
    out = out.astype("uint8")
    im = Image.fromarray(out)

    if args.out[-4:] == ".npy":
        outfile = args.out[:-4]
    else:
        outfile = str(args.out)
    im.save(outfile + ".png")

    return


####### main bit ###########
np.random.seed(args.seed)
random.seed(args.seed)

# sigma channel
degree,x_pos,y_pos = random_points() # draw random polynomial degree and points to draw lines through
if degree == 0: # if zero degree, create a flat map
    sigma = psuedo_log_uniform(args.min_c1,args.max_c1)
    s_mat = np.full((args.w,args.w,1), sigma)
else:
    num_segments,intercepts,ys = fit_curve(degree, x_pos, y_pos) # fit curve, extract some data from plot
    sigmas = assign_values(num_segments, args.min_c1, args.max_c1) # loop through columns first, find which segments belong to each column
    segment_map = map_segments(num_segments,intercepts)
    s_mat = make_mat(segment_map, sigmas, ys)
    s_mat = flip(s_mat) # random flips and transpositions
    
# K channel
degree,x_pos,y_pos = random_points()    
if degree == 0:
    k = psuedo_log_uniform(args.min_c2,args.max_c2)
    k_mat = np.full((args.w,args.w,1), k)
else:
    num_segments,intercepts,ys = fit_curve(degree, x_pos, y_pos)
    ks = assign_values(num_segments,args.min_c2,args.max_c2)
    segment_map = map_segments(num_segments,intercepts)
    k_mat = make_mat(segment_map, ks, ys)
    k_mat = flip(k_mat)

# write
mat = np.concatenate([s_mat,k_mat], axis=2)
np.save(args.out, mat)

# png
if args.png:
    make_png(s_mat, k_mat)

