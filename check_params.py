# checking over the command line params
import os


def check_params(args):

    # avoid overwriting saved weights or other output files
    if args.train == True:
        if os.path.exists(args.out + "/mapNN_" + str(args.seed) + "_model.hdf5"):
            print("saved model with specified output name already exists (i.e. --out)")
            exit()

    # params shared across pipelines
    if args.preprocess==True or args.preprocess_density_grid==True or args.train==True or args.predict==True or args.bootstrap==True:
        if args.seed == None:
            print("specify seed via --seed")
            exit()
        if args.out == None:
            print("specify seed via --seed")
            exit()

    if args.preprocess==True or args.preprocess_density_grid==True or args.train==True or args.predict==True:
        if args.num_snps == None:
            print("specify num snps via --num_snps")
            exit()
        if args.n == None:
            print("specify max sample size via --n")
            exit()
            

