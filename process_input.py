import numpy as np
import sys
from geopy import distance
import random
from PIL import Image
from numpy import asarray
from skimage.measure import block_reduce


# convert vcf to genotype array
# filters:
#     1. biallelic change the alelles to 0 and 1 before inputting.
#     2. no missing data: filter or impute.
#     3. ideally no sex chromosomes, and only look at one sex at a time.
def vcf2genos(vcf_path, n, num_snps, phase):
    geno_mat = []
    vcf = open(vcf_path, "r")
    current_chrom = None
    for line in vcf:
        if line[0:2] == "##":
            pass
        elif line[0] == "#":
            header = line.strip().split("\t")
            if n == None:  # option for getting sample size from vcf
                n = len(header)-9
        else:
            newline = line.strip().split("\t")
            genos = []
            for field in range(9, len(newline)):
                geno = newline[field].split(":")[0].split("/")
                geno = [int(geno[0]), int(geno[1])]
                if phase == 1:
                    genos.append(sum(geno))
                elif phase == 2:
                    genos.append(geno[0])
                    genos.append(geno[1])
                else:
                    print("problem")
                    exit()
            for i in range((n * phase) - len(genos)):  # pad with 0s
                genos.append(0)
            geno_mat.append(genos)

    # check if enough snps
    if len(geno_mat) < num_snps:
        print("not enough snps")
        exit()
    if len(geno_mat[0]) < (n * phase):
        print("not enough samples")
        exit()

    # sample snps
    geno_mat = np.array(geno_mat)
    return geno_mat[np.random.choice(geno_mat.shape[0], num_snps, replace=False), :]


# calculate isolation by distance
def ibd(genos, coords, phase, num_snps):

    # subset for n samples (avoiding padding-zeros)
    n = 0
    for i in range(genos.shape[1]):
        reverse_index = genos.shape[1]-i-1
        if len(set(genos[:, reverse_index])) > 1:
            n += reverse_index
            break
    n += 1  # for 0 indexing
    if phase == 2:
        n = int(n/2)
    genos = genos[:, 0:n*phase] 

    # if collapsed genos, make fake haplotypes for calculating Rousset's statistic
    if phase == 1:
        geno_mat2 = []
        for i in range(genos.shape[1]):
            geno1, geno2 = [], []
            for s in range(genos.shape[0]):
                combined_geno = genos[s, i]
                if combined_geno == 0.0:
                    geno1.append(0)
                    geno2.append(0)
                elif combined_geno == 2:
                    geno1.append(1)
                    geno2.append(1)
                elif combined_geno == 1:
                    alleles = [0, 1]
                    # assign random allele to each haplotype
                    geno1.append(alleles.pop(random.choice([0, 1])))
                    geno2.append(alleles[0])
                else:
                    print("bug", combined_geno)
                    exit()
            geno_mat2.append(geno1)
            geno_mat2.append(geno2)
        geno_mat2 = np.array(geno_mat2)
        genos = geno_mat2.T

    # denominator for "a"
    locus_specific_denominators = np.zeros((num_snps))
    P = (n*(n-1))/2  # number of pairwise comparisons
    for i1 in range(0, n-1):
        X11 = genos[:, i1*2]
        X12 = genos[:, i1*2+1]
        X1_ave = (X11+X12)/2  # average allelic does within individual-i
        for i2 in range(i1+1, n):
            X21 = genos[:, i2*2]
            X22 = genos[:, i2*2+1]
            X2_ave = (X21+X22)/2
            #
            SSw = (X11-X1_ave)**2 + (X12-X1_ave)**2 + \
                (X21-X2_ave)**2 + (X22-X2_ave)**2
            locus_specific_denominators += SSw
    locus_specific_denominators = locus_specific_denominators / (2*P)
    denominator = np.sum(locus_specific_denominators)

    # numerator for "a"
    gendists = []
    for i1 in range(0, n-1):
        X11 = genos[:, i1*2]
        X12 = genos[:, i1*2+1]
        X1_ave = (X11+X12)/2  # average allelic does within individual-i
        for i2 in range(i1+1, n):
            X21 = genos[:, i2*2]
            X22 = genos[:, i2*2+1]
            X2_ave = (X21+X22)/2
            #
            SSw = (X11-X1_ave)**2 + (X12-X1_ave)**2 + \
                (X21-X2_ave)**2 + (X22-X2_ave)**2
            Xdotdot = (X11+X12+X21+X22)/4  # average allelic dose for the pair
            # a measure of between indiv
            SSb = (X1_ave-Xdotdot)**2 + (X2_ave-Xdotdot)**2
            locus_specific_numerators = ((2*SSb)-SSw) / 4
            numerator = np.sum(locus_specific_numerators)
            a = numerator/denominator
            gendists.append(a)

    # geographic distance
    geodists = []
    for i in range(0, n-1):
        for j in range(i+1, n):
            d = distance.distance(coords[i, :], coords[j, :]).km
            d = np.log(d)
            geodists.append(d)

    # regression
    from scipy import stats
    geodists = np.array(geodists)
    gendists = np.array(gendists)
    b = stats.linregress(geodists, gendists)[0]
    r = stats.pearsonr(geodists, gendists)[0]
    r2 = r**2
    Nw = (1 / b)
    print("IBD r^2, slope, Nw:", r2, b, Nw)

    
# just what it sounds like
def cookie_cutter(data, outline, fill=0.0):
    if len(data.shape) == 2:
        data = np.reshape(data,(data.shape[0],data.shape[1],1))  # add dim
    for i in range(data.shape[0]):       
        for j in range(data.shape[1]):   
            if outline[i,j] == 0:
                data[i,j,:] = fill
    if data.shape[2] == 1:
        data = np.reshape(data,(data.shape[0],data.shape[1]))  # remove extra dim
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


# performs sampling with weights according to the frequency of distances in the sample
def weighted_sample_dists(locs,loc_range,n,pairs,num_bins):
    # get dists                                                                             
    dists = []
    locs = locs.T
    for i in range(n-1):
        for j in range(i+1,n):
          d = np.linalg.norm(locs[i,:] - locs[j,:])
          dists.append(d)

    # counts bins                                                                           
    bins = []
    max_dist = (loc_range**2 + loc_range**2)**(0.5)
    bin_size = float(max_dist) / float(num_bins)
    for i in range(num_bins):
        new_count = 0
        start = bin_size*i
        end = bin_size*(i+1)
        for d in dists:
            if d >= start and d < end:
                new_count += 1
        bins.append(new_count)

    # get props for bins                                                                    
    props = []
    all_pairs = int((float(n)*(float(n)-1))/2)
    for i in range(num_bins):
        p = 1 - float(bins[i]) / all_pairs # invert                                             
        props.append(p)

    # assign props to pairs                                                                 
    weights = []
    for d in dists:
        for i in range(num_bins):
            start = bin_size*i
            end = bin_size*(i+1)
            if d >= start and d < end:
                weights.append(props[i])

    # normalize                                                                             
    weights = np.array(weights)
    total =np.sum(weights)
    weights /= total

    # finally, sample                                     
    sample = np.random.choice(np.arange(all_pairs), size=pairs, replace=False, p=weights)

    return sample


def grid_density(slim_output,w,grid_coarseness,current_gen):
    De = np.zeros((grid_coarseness,grid_coarseness))                                     
    distances = np.zeros((grid_coarseness,grid_coarseness))
    offspring = np.zeros((grid_coarseness,grid_coarseness))    
    num_dispersions = np.zeros((grid_coarseness,grid_coarseness))
    min_dispersions = 1 # minimum number of dispersions to calculate dispersal rate (at least 1, to avoid 0-division)
    gens = 0 # counting gens                                                                             

    # find pixel on W width map from x,y
    def find_pixel(x,y,w,grid_coarseness):
        i = int(np.floor((x/w) * grid_coarseness))
        j = int(np.floor((y/w) * grid_coarseness))
        if i == grid_coarseness: # sometimes they do land right on the edge                                
            i = grid_coarseness-1
        if j ==grid_coarseness:
            j = grid_coarseness-1

        # swap x,y to convert between slim and PNG, and flip (new) i-value
        i,j = j,i
        i = grid_coarseness-i-1 # -1 for zero indexing

        return i,j

    # parse data
    with open(slim_output) as infile:                                                                              
        infile.readline()  # header                                                                                     
        for line in infile:                                                                                             
            newline = line.strip().split()                                                                              
            gen = int(newline[0])                                                                                       
            x,y = float(newline[2]),float(newline[3])
            i,j = find_pixel(x,y,w,grid_coarseness)
            if gen > current_gen:                                                                                   
                current_gen = int(gen)
                gens+=1                                                                                             
            if newline[1] == "ALIVE":                                                                               
                De[(i,j)] += 1                                                                                      
            elif newline[1] == "DEAD":                                                                              
                count,distance = float(newline[4]),float(newline[5])
                weighted_dist = distance*count
                if weighted_dist > 0:
                    distances[(i,j)] += weighted_dist
                    offspring[(i,j)] += count
                    num_dispersions[(i,j)] += 1
            elif newline[1] == "MATING":
                pass
            else:                                                                                                   
                print("issue")                                                                                      
                exit()                                                                                              

    # final calculations
    for i in range(grid_coarseness):
        for j in range(grid_coarseness):
            if De[(i,j)] < min_dispersions: 
                De[(i,j)] = 0
    De /= gens
    sigmas = np.zeros((grid_coarseness,grid_coarseness))
    for i in range(grid_coarseness):
        for j in range(grid_coarseness):
            if num_dispersions[(i,j)] >= min_dispersions:
                sigma = np.sqrt(distances[(i,j)] / offspring[(i,j)])
                sigmas[(i,j)] = sigma
                
    counts = np.stack([sigmas,De],axis=2)

    return counts


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
            # # this approach won't work until you log-scale the heatmap                                              
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
        else:
            # find range of sigma, and range of K *inside the habitat* (not in the water) for empirical interpretation
            min_sigma,max_sigma,min_k,max_k=1e16,0,1e16,0  # defaults                                                 
            for j in range(the_map.shape[0]):
                for k in range(the_map.shape[1]):
                    if habi_map[j,k] == 1:  # land                                                                    
                        min_sigma = np.min([min_sigma,the_map[j,k,0]])
                        max_sigma = np.max([max_sigma,the_map[j,k,0]])
                        min_k = np.min([min_k,the_map[j,k,1]])
                        max_k = np.max([max_k,the_map[j,k,1]])
        return min_sigma,max_sigma,min_k,max_k

    
# main
def main():
    vcf_path = sys.argv[1]
    n = sys.argv[2]
    if n == "None":
        n = None
    else:
        n = int(n)
    num_snps = int(sys.argv[3])
    outname = sys.argv[4]
    phase = int(sys.argv[5])
    geno_mat = vcf2genos(vcf_path, n, num_snps, phase)
    np.save(outname + ".genos", geno_mat)


if __name__ == "__main__":
    main()
