import numpy as np
import sys, os

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
    geno_mat = geno_mat[np.random.choice(geno_mat.shape[0], num_snps, replace=False), :]

    return geno_mat


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


def grid_density(slim_output,grid_coarseness):
    # read data
    with open(slim_output) as infile:                                                                                       
        De = np.array(list(map(float,infile.readline().strip().split())))
        sigmas = np.array(list(map(float,infile.readline().strip().split())))

    # make map shape
    De = np.reshape(De, (int(grid_coarseness),int(grid_coarseness)))
    sigmas = np.reshape(sigmas, (int(grid_coarseness),int(grid_coarseness)))
    counts = np.stack([sigmas,De],axis=2) 

    np.save('temp1NEW',counts)

    return counts


# convert locations from geographic coordinates to row and column positions in a 2d array
#
# What are the locs, exactly?  (updated 12.13.23)                                                                                
# 1. We start with a randomly generated map array (np.savetxt to CSV); or .npy if applying habitat mask.                         
# 2. If habitat PNG, arrange top left pointing northwest. Load PNG (PIL.Image.open()), mask map pixel 0,0 with habitat pixel 0,0.
# 3. Load into SLiM (readCSV().asMatrix()), and it's already oriented the way we want in GUI with topleft pointing northwest.    
# 4. In terms of array (row,col) indices, pixel 0,0 is topleft; pixel 0,50 is topright, etc. (np.asarray(Image.open()))          
# 5. Individual locs from SLiM use cartesian coordinates: bottom left 0,0, top left 0,W, etc (p1.individuals.spatialPosition)    
# 6. Convert x,y to array indices: (i) reverse first dim (W-i), and then (ii) swap the first and second dim (i,j=j,i).           
def coords2array(locs, w):
    new_locs = np.array(locs)
    new_locs[0,:] = w - new_locs[0,:]  # flip first dimension (to match PNG)
    new_locs = np.flip(new_locs, axis=0) # swap x and y (to match PNG indices)
    return new_locs

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
