# data generator code for training CNN

import sys
import numpy as np
import tensorflow as tf
import msprime
import tskit
from attrs import define,field
from read_input import *
import gc # garbage collect

@define
class DataGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"

    list_IDs: list
    targets: dict 
    num_snps: int
    n: int
    batch_size: int
    mu: float
    shuffle: bool
    baseseed: int
    sampling_width: float
    phase: int
    polarize: int
    genos: dict
    locs: dict
    map_width: int
    sample_grid: int
    empirical_locs: list
    slim_width: float
    vcf: bool
    out: str
    simid: int
    chroms: int

    def __attrs_post_init__(self):
        "Initialize a few things"
        self.on_epoch_end()
        np.random.seed(self.baseseed)

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def cropper(self, ts, W, sample_width, alive_inds, scaling_factor):
        "Cropping the map, returning individuals inside sampling window"
        cropped = [] 
        left_edge = np.random.uniform(
            low=0, high=W-sample_width
        )
        right_edge = left_edge + sample_width
        bottom_edge = np.random.uniform(
            low=0, high=W-sample_width
        )
        top_edge = bottom_edge + sample_width

        for i in alive_inds:
            ind = ts.individual(i.id)
            loc = ind.location[0:2]
            loc /= scaling_factor
            if (
                loc[0] > left_edge
                and loc[0] < right_edge
                and loc[1] > bottom_edge
                and loc[1] < top_edge
            ):
                cropped.append(i.id)

        return cropped


    def grid_sample(self, ts, sampled_inds, W, scaling_factor):
        if len(sampled_inds) < self.sample_grid**2:
            print(W,"your sample grid is too fine,", self.sample_grid, ",not enough samples to fill it",len(sampled_inds))
            exit()
        if self.n % self.sample_grid**2 != 0:
            print("n not divisible by sample grid; don't know what to do")
            exit()
        bin_size = W / self.sample_grid
        inds_per_bin = int(np.ceil(self.n / self.sample_grid**2))
        keepers = []
        for i in range(int(self.sample_grid)):
            for j in range(int(self.sample_grid)):
                hole_count = 0
                for ind in sampled_inds:
                    indiv = ts.individual(ind)
                    loc = indiv.location[0:2]
                    loc /= scaling_factor
                    if (
                        loc[0] > (i*bin_size)
                        and loc[0] < ((i+1)*bin_size)
                        and loc[1] > (j*bin_size)
                        and loc[1] < ((j+1)*bin_size)
                    ):
                        keepers.append(ind)
                        hole_count+=1
                    if hole_count == inds_per_bin:
                        break
        if len(keepers) < self.n:
            print("unfilled grid holes; try looser grid, or adjust simulation")
            exit()
        return keepers


    def unpolarize(self, snp):
        "Change 0,1 encoding to major/minor allele. Also filter no-biallelic"
        alleles = {}                                                                          
        for i in range(self.n * 2):  
            a = snp[i]                                                               
            if a not in alleles:                                                              
                alleles[a] = 0                                                                
            alleles[a] += 1                                                                   
        if len(alleles) == 2:                                                                 
            new_genotypes = []                                                                
            major, minor = list(set(alleles))  # set() gives random order                     
            if alleles[major] < alleles[minor]:                                               
                major, minor = minor, major                                                   
            for i in range(self.n * 2):  # go back through and convert genotypes                   
                a = snp[i]                                                           
                if a == major:                                                                
                    new_genotype = 0                                                          
                elif a == minor:                                                              
                    new_genotype = 1                                                          
                new_genotypes.append(new_genotype)
        else:
            new_genotypes = False
            
        return new_genotypes


    def empirical_sample(self, ts, sampled_inds, n, N, W, scaling_factor):
        locs = np.array(self.empirical_locs)
        keep_indivs = []

        # ### nearest indiv
        # np.random.shuffle(locs)
        # indiv_dict = {} 
        # for i in sampled_inds:
        #     indiv_dict[i] = 0
        # for pt in range(n): # for each sampling location
        #     dists = {}
        #     for i in indiv_dict:
        #         ind = ts.individual(i)
        #         loc = ind.location[0:2]
        #         loc /= scaling_factor
        #         d = ( (loc[0]-locs[pt,0])**2 + (loc[1]-locs[pt,1])**2 )**(0.5)
        #         dists[d] = i # see what I did there?
        #     nearest = dists[min(dists)]
        #     print(ts.individual(nearest).location[0:2] / scaling_factor)
        #     keep_indivs.append(nearest)
        #     del indiv_dict[nearest]
        # ###

        ### arbitrary radius
        np.random.shuffle(sampled_inds)
        for pt in range(n): # for each sampling location
            radius=1
            sampled=False
            while sampled == False:
                for i in sampled_inds:
                    ind = ts.individual(i)                                        
                    loc = ind.location[0:2]                                       
                    loc /= scaling_factor                                         
                    d = ( (loc[0]-locs[pt,0])**2 + (loc[1]-locs[pt,1])**2 )**(0.5)
                    if d <= radius:
                        keep_indivs.append(i)
                        sampled_inds.remove(i)
                        sampled=True
                        break
                # (unindent)
                radius *= 2
        ###
        return keep_indivs    



    def sample_ts(self, filepath, seed):
        "The meat: load in and fully process a tree sequence"

        # read input
        np.random.seed(seed)
        tss=[]
        if self.chroms is None:
            tss.append(tskit.load(filepath))
        else:
            fp = filepath.replace(".trees", "_chr1.trees") # (we'll hit the other chroms further down)
            tss.append(tskit.load(fp))
        
        # for converting from SLiM map to new, pixelated map size
        W = float(self.map_width)  # new map width, e.g., pixels
        scaling_factor = self.slim_width / W

        # crop map
        sampled_inds = []
        failsafe = 0
        while (
            len(sampled_inds) < self.n
        ):  # keep looping until you get a map with enough samples
            if self.sampling_width != None:
                sample_width = (float(self.sampling_width) * W)
            else:
                sample_width = np.random.uniform(0, W)
            sampled_inds = self.cropper(tss[0], W, sample_width, tss[0].individuals(), scaling_factor)
            failsafe += 1
            if failsafe > 100:
                print("\tnot enough samples, killed while-loop after 100 loops:", filepath)
                sys.stdout.flush()
                exit()

        # sampling
        if self.empirical_locs is not None:
            keep_indivs = self.empirical_sample(tss[0], sampled_inds, self.n, len(sampled_inds), W, scaling_factor)
        elif self.sample_grid is not None:
            keep_indivs = self.grid_sample(tss[0], sampled_inds, W, scaling_factor)
        else:
            keep_indivs = np.random.choice(sampled_inds, self.n, replace=False)
        keep_nodes = []
        for i in keep_indivs:
            ind = tss[0].individual(i)
            keep_nodes.extend(ind.nodes)

        # load remaining chroms
        tss[0] = tss[0].simplify(keep_nodes)
        if self.chroms is not None:
            for c in range(1+1,self.chroms+1):
                fp = filepath.replace(".trees", "_chr"+str(c)+".trees")
                ts_ = tskit.load(fp)
                ts_ = ts_.simplify(keep_nodes)
                tss.append(ts_)
        
        # mutate
        mu = float(self.mu)
        counter = 0
        current_snp_count = 0
        while current_snp_count < (self.num_snps * 2): # extra SNPs because need to filter a few non-biallelic
            if counter == 10:
                print("\n\nsorry, Dude. Didn't generate enough snps. \n\n")
                sys.stdout.flush()
                exit()
            for t in range(len(tss)):
                prev_snps = tss[t].num_sites
                tss[t] = msprime.sim_mutations(
                    tss[t],
                    rate=mu,
                    random_seed=seed,
                    model=msprime.SLiMMutationModel(type=0),
                    keep=True,
                )
                current_snp_count += tss[t].num_sites-prev_snps
            # (unindent)
            counter += 1
            mu *= 10

        # grab genos
        geno_mat0 = tss[0].genotype_matrix()
        if self.chroms is not None:
            for t in range(1,self.chroms):
                geno_mat0 = np.concatenate([geno_mat0, tss[t].genotype_matrix()], axis=0)

        # free mem                 
        for t in reversed(range(len(tss),1)):
            del tss[t]

        # grab spatial locations
        sample_dict = {}
        locs = []
        for samp in tss[0].samples():
            node = tss[0].node(samp)
            indID = node.individual
            if indID not in sample_dict:
                sample_dict[indID] = 0
                loc = tss[0].individual(indID).location[0:2]
                loc /= scaling_factor
                locs.append(loc)
        locs = np.array(locs)
        locs = locs.T

        ### things for comparing with other methods (single chrom only) ###
        # write vcf
        if self.vcf is True:
            indivlist = []
            sample_dict = {}
            for samp in tss[0].samples():
                node = tss[0].node(samp)
                indID = node.individual
                if indID not in sample_dict:
                    sample_dict[indID] = 0
                    indivlist.append(indID)
            # (unindent)
            os.makedirs(self.out+"/VCFs", exist_ok=True)
            with open(self.out+"/VCFs/snps_"+str(self.simid)+".vcf", "w") as vcf_file:
                tss[0].write_vcf(vcf_file, individuals=indivlist)
            # write locs
            np.save(self.out+"/VCFs/snps_"+str(self.simid)+"_raw.locs", locs)
        ###

        # change 0,1 encoding to major/minor allele  
        if self.polarize == 2:
            shuffled_indices = np.arange(current_snp_count)
            np.random.shuffle(shuffled_indices) 
            geno_mat1 = []    
            snp_counter = 0   
            snp_index_map = {}
            for s in range(self.num_snps): 
                new_genotypes = self.unpolarize(geno_mat0[shuffled_indices[s]])
                if new_genotypes != False: # if bi-allelic, add in the snp
                    geno_mat1.append(new_genotypes)            
                    snp_index_map[shuffled_indices[s]] = int(snp_counter)
                    snp_counter += 1
            while snp_counter < self.num_snps: # likely need to replace a few non-biallelic sites
                s += 1
                new_genotypes = self.unpolarize(geno_mat0[shuffled_indices[s]])
                if new_genotypes != False:
                    geno_mat1.append(new_genotypes)
                    snp_index_map[shuffled_indices[s]] = int(snp_counter)
                    snp_counter += 1
            geno_mat0 = [] 
            sorted_indices = list(snp_index_map) 
            sorted_indices.sort() 
            for snp in range(self.num_snps):
                geno_mat0.append(geno_mat1[snp_index_map[sorted_indices[snp]]])
            geno_mat0 = np.array(geno_mat0)
                                                
        # sample SNPs
        else:
            mask = [True] * self.num_snps + [False] * (current_snp_count - self.num_snps)
            np.random.shuffle(mask)
            geno_mat0 = geno_mat0[mask, :]

        # collapse genotypes, change to minor allele dosage (e.g. 0,1,2)
        if self.phase == 1:
            geno_mat1 = np.zeros((self.num_snps, self.n))
            for ind in range(self.n):
                geno_mat1[:, ind] += geno_mat0[:, ind * 2]
                geno_mat1[:, ind] += geno_mat0[:, ind * 2 + 1]
            geno_mat0 = np.array(geno_mat1) # (change variable name)

        # sample SNPs
        mask = [True] * self.num_snps + [False] * (self.num_snps - self.num_snps)
        np.random.shuffle(mask)
        geno_mat1 = geno_mat0[mask, :]
        geno_mat2 = np.zeros((self.num_snps, self.n * self.phase)) # pad
        geno_mat2[:, 0 : self.n * self.phase] = geno_mat1
        
        # garbage collect, free memory
        del tss
        del geno_mat0
        del geno_mat1
        del mask
        gc.collect()
        
        return geno_mat2, locs


    def __data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples"        
        X1 = np.empty((self.batch_size, self.num_snps, self.n), dtype="int8")  # genos 
        X2 = np.empty((self.batch_size, 2, self.n), dtype=float)  # locs
        y = np.empty((self.batch_size, self.map_width, self.map_width, 2), dtype=float)

        # generate shuffled indices
        shuffled_indices = np.arange(self.n)
        np.random.shuffle(shuffled_indices)

        for i, ID in enumerate(list_IDs_temp):
            # load map
            y[i] = np.load(self.targets[ID])

            # shuffle genos
            genomat = np.load(self.genos[ID])
            genomat = genomat[:, shuffled_indices]
            X1[i, :] = genomat

            # shuffle, flip, reorient, and rescale locs
            locs = np.load(self.locs[ID])  # load
            locs = locs[:, shuffled_indices]  # shuffle        
            locs[0,:] = self.map_width - locs[0,:]  # flip first dimension (to match PNG)
            locs = np.flip(locs, axis=0) # swap x and y (to match PNG indices)
            X2[i, :] = locs
            
        # (unindent)
        X = [X1, X2]                                  

        return (X, y)
