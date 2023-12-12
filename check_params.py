# checking over the command line params
import os


def check_params(args):

    # avoid overwriting saved weights or other output files
    if args.train == True:
        if os.path.exists(args.out + "/mapNN_" + str(args.seed) + "_model.hdf5"):
            print("saved model with specified output name already exists (i.e. --out)")
            exit()

    # arguments for training
    if args.train == True:
        if args.num_snps == None:
            print("specify num snps via --num_snps")
            exit()
        if args.n == None:
            print("specify max sample size via --n")
            exit()

    # other param combinations
    if args.predict == True and args.empirical == None:
        if args.n == None:
            print("missing sample size, via --n")
            exit()
