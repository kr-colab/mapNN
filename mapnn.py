import os
import argparse
from sklearn.model_selection import train_test_split
from check_params import *
from read_input import *
from process_input import *
from plot_images import *
from data_generation import DataGenerator
import gpustat
import itertools
import PIL.Image as Image
from matplotlib import pyplot as plt
from matplotlib import cm,colors
import matplotlib as mpl
import math
import PIL.Image as Image

def load_dl_modules():
    print("loading bigger modules")
    import numpy as np
    global tf
    import tensorflow as tf
    from tensorflow import keras
    if args.seed is not None:
        tf.random.set_seed(args.seed)
        tf.keras.utils.set_random_seed(args.seed)
    return

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train", action="store_true", default=False, help="run training pipeline"
)
parser.add_argument(
    "--predict", action="store_true", default=False, help="run prediction pipeline"
)
parser.add_argument(
    "--preprocess",
    action="store_true",
    default=False,
    help="create preprocessed tensors from tree sequences",
)
parser.add_argument("--empirical", default=None, type=str,
                    help="prefix for vcf and locs")
parser.add_argument(
    "--target_list", help="list of filepaths to targets (sigma).", default=None)
parser.add_argument(
    "--tree_list", help="list of tree filepaths.", default=None)
parser.add_argument(
    "--counts_list", help="list of recorded density counts.", default=None)
parser.add_argument(
    "--sampling_width", help="just the sampling area", default=1.0, type=float
)
parser.add_argument(
    "--num_snps",
    default=None,
    type=int,
    help="maximum number of SNPs across all datasets (for pre-allocating memory)",
)
parser.add_argument(
    "--num_pred", default=None, type=int, help="number of datasets to predict on"
)
parser.add_argument(
    "--n",
    default=None,
    type=int,
    help="sample size",
)
parser.add_argument(
    "--mu",
    help="beginning mutation rate: mu is increased until num_snps is achieved",
    default=1e-15,
    type=float,
)
parser.add_argument(
    "--num_reps",
    default=1,
    type=int,
    help="number of replicate-draws from the genotype matrix of each sample",
)
parser.add_argument(
    "--validation_split",
    default=0.2,
    type=float,
    help="0-1, proportion of samples to use for validation.",
)
parser.add_argument("--batch_size", default=1, type=int, help="batch size for training")
parser.add_argument("--max_epochs", default=1000,
                    type=int, help="max epochs for training")
parser.add_argument(
    "--patience",
    type=int,
    default=100,
    help="n epochs to run the optimizer after last improvement in validation loss.",
)
parser.add_argument(
    "--out", help="file name stem for output", default=None, required=True
)
parser.add_argument("--seed", default=None, type=int, help="random seed.")
parser.add_argument("--simid", default=None, type=int, help="specific simulation id for preprocessing: 1-indexed, corresponds to line number in tree_list.txt")
parser.add_argument("--gpu_index", default="-1", type=str,
                    help="index of gpu. To avoid GPUs, skip this flag or say '-1'. To use any available GPU say 'any' ")
parser.add_argument(
    "--load_weights",
    default=None,
    type=str,
    help="Path to a _weights.hdf5 file to load weight from previous run.",
)
parser.add_argument(
    "--phase",
    default=1,
    type=int,
    help="1 for unknown phase, 2 for known phase",
)
parser.add_argument(
    "--polarize",
    default=2,
    type=int,
    help="2 for major/minor, 1 for ancestral/derived",
)
parser.add_argument(
    "--keras_verbose",
    default=1,
    type=int,
    help="verbose argument passed to keras in model training. \
                    0 = silent. 1 = progress bars for minibatches. 2 = show epochs. \
                    Yes, 1 is more verbose than 2. Blame keras.",
)
parser.add_argument(
    "--threads",
    default=1,
    type=int,
    help="num threads.",
)
parser.add_argument(
    "--training_params", help="params used in training: sigma mean and sd, n, num_snps", default=None
)
parser.add_argument(
    "--learning_rate",
    default=1e-4,
    type=float,
    help="learning rate.",
)
parser.add_argument("--combinations", help="", default=2, type=int)
parser.add_argument("--map_width", help="for preprocessing, the target size", type=int)
parser.add_argument("--sample_grid", help="coarseness of grid for grid-sampling", default=None, type=float)
parser.add_argument("--pairs", help="number of pairs to subsample", default=45, type=int)
parser.add_argument("--pairs_encode", help="number of pairs (<= pairs_encode) to use for gradient in the first part of the network", type=int)
parser.add_argument("--habitat_map", help="path to png file with habitat shaded—for cropping.", default=None)
parser.add_argument("--habitat_border", help="path to png file with outline of habitat—only for final visualization.", default=None)
parser.add_argument("--slim_width", help="range of locs from simulation, if different than target maps", default=None, type=float)
parser.add_argument("--filts1", help="num filters convolvulator", type=int, default = 126)
parser.add_argument("--filts2", help="num filters continuous filter conv", type=int, default = 64)
parser.add_argument("--vcf",default=False,action="store_true",help="output vcf and other files for methods comparison pipeline")
parser.add_argument("--ranges",default=None,type=float,help="for plotting: --ranges <min_sigma> <max_sigma> <min_k> <max_k>", nargs=4)
parser.add_argument("--preprocess_density_grid", help="calcualte effective density in a grid", default=False, action="store_true",)
parser.add_argument("--chroms",default=None, type=int,help="num chroms to preprocess multiple chroms")
parser.add_argument(
    "--plot_history",
    default=False,
    type=str,
    help="plot training history? default: False",
)
parser.add_argument("--bootstrap",default=False,action="store_true",help="treat test dir as bootstrap simulations, output uncertainty map")

args = parser.parse_args()
check_params(args)





def load_network(map_width,habitat_map):
    if args.gpu_index != 'any':  # 'any' will search for any available GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    else:
        stats = gpustat.GPUStatCollection.new_query()
        ids = map(lambda gpu: int(gpu.entry['index']), stats)
        ratios = map(lambda gpu: float(
            gpu.entry['memory.used'])/float(gpu.entry['memory.total']), stats)
        bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(bestGPU)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    
    # update conv+pool iterations based on number of SNPs
    num_conv_iterations = int(np.floor(np.log10(args.num_snps))-1) + 1
    if num_conv_iterations < 0:
        num_conv_iterations = 0

    ### *** wolf hack for 10627 snps
    num_conv_iterations = 3
    print("\n\n\n\n\n\t\tCURRENTLY DOING A CUSTOM 3 CONV ITERATIONS, PROBABLY CHANGE THIS\n\n\n\n\n")
    ###

    # organize pairs of individuals
    combinations = list(itertools.combinations(range(args.n), 2))
    combinations = random.sample(combinations, args.pairs)
    combinations_encode = random.sample(combinations, args.pairs_encode)
    combinations = list2dict(combinations) # (using tuples as dict keys seems to work)
    combinations_encode = list2dict(combinations_encode)

    # load inputs
    geno_input = tf.keras.layers.Input(shape=(args.num_snps, args.n)) 
    loc_input = tf.keras.layers.Input(shape=(2, args.n))

    # initialize shared layers
    CONV_LAYERS = []
    conv_kernal_size = 2
    pooling_size = 10    
    for i in range(num_conv_iterations):                                             
        filter_size = 20 + 44*(i+1)
        CONV_LAYERS.append(tf.keras.layers.Conv1D(filter_size, kernel_size=conv_kernal_size, activation="relu", name="extract_CONV1d_"+str(i)))
    DENSE_0 = tf.keras.layers.Dense(args.filts1, activation="relu", name="extract_DENSE_0")
    POOL = tf.keras.layers.AveragePooling1D(pool_size=pooling_size)
    FLATTEN_1 = tf.keras.layers.Flatten()
    FLATTEN_2 = tf.keras.layers.Flatten()
    
    # CONVOLVULATOR
    hs = []
    ls = []
    for comb in combinations:
        h = tf.gather(geno_input, comb, axis = 2)
        if comb in combinations_encode:
            for i in range(num_conv_iterations):
                h = CONV_LAYERS[i](h)
                h = POOL(h)
            h = FLATTEN_1(h)
            h = DENSE_0(h)
        else: # cut gradient tape on some pairs to save memory                 
            for i in range(num_conv_iterations):
                h = tf.stop_gradient(CONV_LAYERS[i](h))
                h = POOL(h)
            h = FLATTEN_1(h)
            h = tf.stop_gradient(DENSE_0(h))
        hs.append(h)
        l = tf.gather(loc_input, comb, axis = 2)
        l = FLATTEN_2(l)
        ls.append(l)

    # stack conv output and locs
    feature_block = tf.stack(hs, axis=1) # stack geno summaries grom all pairs
    print("\nfeature block:", feature_block.shape)
    l = tf.stack(ls, axis=1) # stack locs from all pairs 

    # initialize shared layers
    pixel_stack = []
    DENSE_loc_disp_0 = tf.keras.layers.Dense(args.filts1, activation="relu", name="locs_DENSE_disp_0")
    DENSE_loc_disp_1 = tf.keras.layers.Dense(args.filts1, activation="relu", name="locs_DENSE_disp_1")
    DENSE_weighted = tf.keras.layers.Dense(args.filts2, activation="relu", name="weightedFeatures_DENSE_disp_0")
    DENSE_pooled_0 = tf.keras.layers.Dense(args.filts2, activation="relu",   name="weightedFeatures_DENSE_disp_1")
    DENSE_pooled_1 = tf.keras.layers.Dense(args.filts2, activation="relu",   name="weightedFeatures_DENSE_disp_2")
    DENSE_linear = tf.keras.layers.Dense(2,     activation="linear", name="weightedFeatures_DENSE_disp_3")

    # build locs table    
    pixels,mask = [],[]
    for i in range(map_width):
        for j in range(map_width):
            for k in combinations:
                pixels.append([0,0,0,0, float(i), float(j), map_width])
                mask.append(  [1,1,1,1, 0,        0,        0])
    pixels,mask =np.array(pixels),np.array(mask)
    padding = [[0,0],[0,0],[0,3]]
    locs_table = tf.tile(l,[1,map_width**2,1])
    locs_table = tf.pad(locs_table, padding)
    locs_table = locs_table * mask + pixels * (1 - mask) # hack to get custom values in
    print("location stacks (grad)", locs_table.shape)

    # DENSE
    DENSE_loc_disp_0 = tf.keras.layers.Dense(args.filts1, activation="relu", name="locs_DENSE_disp_0") 
    DENSE_loc_disp_1 = tf.keras.layers.Dense(args.filts1, activation="relu", name="locs_DENSE_disp_1")
    spatial_scores = DENSE_loc_disp_0(locs_table)
    spatial_scores = DENSE_loc_disp_1(spatial_scores)
    print("spatial scores", spatial_scores.shape)

    # TILE - rows for each pair X each pixel, cols for each genotype summary 
    g = tf.tile(feature_block,[1,map_width**2,1])
    print("TILE", g.shape)

    # MULTIPLY by spatial scores
    g = g * spatial_scores
    g = tf.keras.layers.ReLU()(g)
    print("MULT", g.shape)

    # DENSE (beggining of the continuous filter conv operation)
    p = tf.keras.layers.Dense(args.filts2, activation="relu", name="weightedFeatures_DENSE_disp_0")(g)
    print("DENSE", p.shape)

    # POOL - as part of the conv operation
    #>>> import tensorflow as tf; import numpy as np; k=5;f=10;p=3; a=np.ones((1,k,f)); b=np.ones((1,k,f))+1; c=np.ones((1,k,f))+2; a[:,:,9]=0; b[:,:,9]=0; c[:,:,9]=0; a=np.concatenate([a,b,c], axis=1); a; a.shape; a=np.reshape(a,(1,k*p,f,1)); a=tf.keras.activations.linear(a); a=tf.keras.layers.AveragePooling2D(pool_size=(k,1))(a); a=np.reshape(a,(1,p,f)); a; a.shape
    p = tf.keras.layers.Reshape(((map_width**2)*args.pairs,args.filts2,1))(p) # add extra dim
    p = tf.keras.layers.AveragePooling2D(pool_size=(args.pairs,1))(p)
    p = tf.keras.layers.Reshape((map_width**2,args.filts2))(p) # remove extra dim
    print("POOL", p.shape)

    # DENSE
    p = tf.keras.layers.Dense(args.filts2, activation="relu",   name="weightedFeatures_DENSE_disp_1")(p)
    p = tf.keras.layers.Dense(args.filts2, activation="relu",   name="weightedFeatures_DENSE_disp_2")(p)
    p = tf.keras.layers.Dense(2,     activation="linear", name="weightedFeatures_DENSE_disp_3")(p)
    print("DENSE", p.shape)
        
    # RESHAPE — into 3d
    p = tf.keras.layers.Reshape((map_width,map_width,2))(p)
    print("RESHAPE", p.shape)

    # custom loss
    if args.habitat_map is None: 
        cost = 'mse'
    else:                                                                                                                       
        def cost(y, y_):
            pixels = np.sum(habitat_map.flatten())
            squared_error = tf.square(y-y_) # squared error for every value in target across both channels
            map1,map2 = squared_error[:,:,:,0],squared_error[:,:,:,1] # split by channel
            combined = tf.keras.layers.Add()([map1,map2]) # sum error pixel-wise, across channels
            combined /= 2 # average, pixel-wise. Now we have one channel
            masked = combined * habitat_map # apply mask
            # testing that axis=(1,2) gives separate sum for each example along first dim (the batch dim)        
            #>>> a = np.ones((3,3)); b = np.ones((3,3)) +2; c=np.stack([a,b]); cost = tf.math.reduce_sum(c, axis=(1,2)); cost     
            #<tf.Tensor: shape=(2,), dtype=float64, numpy=array([ 9., 27.])>                                                      
            cost = tf.math.reduce_sum(masked, axis=(1,2)) # the rank of the tensor is reduced by 1 for each entry in axis
            cost = cost / pixels                                                                                                  
            return cost                                                                                                           
        

    # model overview and hyperparams
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model = tf.keras.Model(
        inputs = [geno_input, loc_input],                                                                                                      
        outputs = [p],
    )
    model.compile(loss=cost, optimizer=opt)
    #model.summary()
    print("\n    per layer weights:")
    for layer in model.layers: 
        if len(layer.get_weights()) > 0:
            print(layer.name, " "*(35-len(layer.name)), "weights:", layer.get_weights()[0].shape, "=", len(layer.get_weights()[0].flatten()), "\t biases:", layer.get_weights()[1].shape, "=", len(layer.get_weights()[1].flatten()))
    #print("total layers:", len(model.layers))
    print("total params:", np.sum([np.prod(v.shape) for v in model.trainable_variables]), "\n")

    # load weights
    if args.load_weights is not None:
        print("loading saved weights")
        model.load_weights(args.load_weights)
    else:
        if args.train is True and args.predict is True:
            weights = args.out + "/mapNN_" + str(args.seed) + "_model.hdf5"
            print("loading weights:", weights)
            model.load_weights(weights)
        elif args.predict is True:
            print("where is the saved model? (via --load_weights)")
            exit()

    # callbacks
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath= args.out + "/mapNN_" + str(args.seed) + "_model.hdf5",
        verbose=args.keras_verbose,
        save_best_only=True,
        saveweights_only=False,
        monitor="val_loss",
        period=1,
    )
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=args.patience
    )
    reducelr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=int(args.patience/10),
        verbose=args.keras_verbose,
        mode="auto",
        min_delta=0,
        cooldown=0,
        min_lr=0,
    )

    return model, checkpointer, earlystop, reducelr


def make_generator_params_dict(
    targets, shuffle, genos, locs, empirical_locs, map_width,
):
    params = {
        "targets": targets,
        "num_snps": args.num_snps,
        "n": args.n,
        "batch_size": args.batch_size,
        "mu": args.mu,
        "shuffle": shuffle,
        "baseseed": args.seed,
        "sampling_width": args.sampling_width,
        "phase": args.phase,
        "polarize": args.polarize,
        "genos": genos,
        "locs": locs,
        "sample_grid": args.sample_grid,
        "empirical_locs": empirical_locs,
        "map_width": map_width,
        "slim_width": args.slim_width,
        "vcf": args.vcf,
        "out": args.out,
        "simid": args.simid,
        "chroms": args.chroms,
    }
    return params






def train():

    # read targets
    print("reading input paths", flush=True)
    targets,genos,locs = dict_from_preprocessed(args.out)
    total_sims = len(targets)

    # read one map to get size
    map_width = np.load(targets[0]).shape[0]

    # split into val,train sets                                                
    sim_ids = np.arange(0, total_sims)
    train, val = train_test_split(sim_ids, test_size=args.validation_split)
    if len(val) % args.batch_size != 0 or len(train) % args.batch_size != 0:
        print(
            "\n\ntrain and val sets each need to be divisible by batch_size; otherwise some batches will have missing data\n\n"
        )
        exit()

    # organize "partitions" to hand to data generator
    partition = {}
    partition["train"] = list(train)
    partition["validation"] = list(val)

    # initialize generators
    params = make_generator_params_dict(
        targets=targets,
        shuffle=True,
        genos=genos,
        locs=locs,
        empirical_locs=None,
        map_width=map_width,
    )
    training_generator = DataGenerator(partition["train"], **params)
    validation_generator = DataGenerator(partition["validation"], **params)

    # read habitat map
    if args.habitat_map == None:
        habitat_map = None
    else:
        habitat_map = read_habitat_map(args.habitat_map, map_width)

    # train
    load_dl_modules()
    model, checkpointer, earlystop, reducelr = load_network(map_width,habitat_map)
    print("training!")
    history = model.fit(
        x=training_generator,
        epochs=args.max_epochs,
        shuffle=False, # (redundant with shuffling inside generator)
        verbose=args.keras_verbose,
        validation_data=validation_generator,
        callbacks=[checkpointer, earlystop, reducelr],
    )

    return





def predict(): 

    # load inputs                                                                               
    if args.simid is None:
        targets,genos,locs = preds_from_preprocessed(args.out)
        total_sims = len(targets)
    else:
        targets = [args.out + "/Maps/" + str(args.seed) + "/" + str(args.simid) + ".target.npy"]
        genos = [args.out + "/Genos/" + str(args.seed) +"/" + str(args.simid) +".genos.npy"]
        locs = [args.out + "/Locs/" + str(args.seed) +"/" + str(args.simid) +".locs.npy"]
        total_sims = 1

    # read one map to get size
    map_width = np.load(targets[0]).shape[0]

    # organize "partition" to hand to data generator
    partition = {}
    if args.num_pred == None:
        args.num_pred = int(total_sims)
    simids = np.random.choice(np.arange(total_sims),
                              args.num_pred, replace=False)

    # get generator ready
    params = make_generator_params_dict(
        targets=targets,
        shuffle=False,
        genos=genos,
        locs=locs,
        empirical_locs=None,
        map_width=map_width,
    )

    # predict
    print("predicting")
    os.makedirs(args.out + "/Test_" + str(args.seed), exist_ok=True)
    load_dl_modules()
    model, checkpointer, earlystop, reducelr = load_network(map_width, None)
    for b in range(int(np.ceil(args.num_pred/args.batch_size))): # loop to alleviate memory                    
        simids_batch = simids[b*args.batch_size:(b+1)*args.batch_size]
        partition["prediction"] = np.array(simids_batch)
        generator = DataGenerator(partition["prediction"], **params)
        predictions = model.predict_generator(generator)
        ###
        #np.save("temp1.npy", predictions)
        #exit()
        #predictions = np.load("temp1.npy")
        ###
        unpack_predictions(predictions, map_width, targets, locs, simids_batch, targets)
    return




def empirical():

    # read locs
    empirical_locs = read_locs(args.empirical + ".locs")
    empirical_locs = np.array(empirical_locs)
    empirical_locs = empirical_locs.T

    # read habitat map                                                          
    targets,genos,locs = dict_from_preprocessed(args.out)  # need target dims
    map_width = np.load(targets[0]).shape[0]
    if args.habitat_map == None:
        habitat_map = None
    else:
        habitat_map = read_habitat_map(args.habitat_map, map_width)

    # process locs to match data generator from training
    empirical_locs[0,:] = map_width - empirical_locs[0,:]  # flip first dimension (to match PNG)
    empirical_locs = np.flip(empirical_locs, axis=0) # swap x and y (to match PNG indices)
    
    # load modules
    load_dl_modules()

    # convert vcf to geno matrix and predict
    predictions = []
    for i in range(args.num_reps):
        print("empirical rep #", i)
        model, checkpointer, earlystop, reducelr = load_network(map_width,habitat_map) # inside loop, to get different pairs each rep.
        test_genos = vcf2genos(  # inside loop to get different snp sets
            args.empirical + ".vcf", args.n, args.num_snps, args.phase
        )
        #ibd(test_genos, locs, args.phase, args.num_snps)
        test_genos = np.reshape(
            test_genos, (1, test_genos.shape[0], test_genos.shape[1])
        )
        test_locs = np.reshape(  # (inside loop just to be clean)
            empirical_locs, (1, empirical_locs.shape[0], empirical_locs.shape[1])
        )
        prediction = model.predict([test_genos, test_locs])
        prediction = np.squeeze(prediction, axis=0)  # get rid of extra dim
        predictions.append(prediction)

    # process predictions
    ###
    #np.save("temp1.npy", predictions)
    #exit()
    #predictions = np.load("temp1.npy")
    ###
    unpack_predictions(predictions, map_width, None, test_locs, None, None)

    return





def unpack_predictions(predictions, map_width, targets, locs_dict, simids, file_name): 
    import cv2 # do I need more coffee or why can't I move this out of the fxn?

    # params
    plot_width=500
    
    # grab mean and sd from training distribution    
    mean_sd = np.load(args.out + "/mean_sd.npy")  # mean and SD from test dir.
    if args.training_params is None:
        training_mean_sd = np.array(mean_sd)
    else:  # for misspecification experiments need separate mean and sd, from training
        training_mean_sd = np.load(args.training_params)

    # read habitat map                                               
    if args.habitat_map is None:
        habitat_map_plot = None
    else:
        habitat_map = read_habitat_map(args.habitat_map, map_width)
        habitat_map_plot = read_habitat_map(args.habitat_map, plot_width)

    # simulated data
    if args.empirical is None:
        os.makedirs(os.path.join(args.out,"Test_" + str(args.seed)), exist_ok=True)
        for i in range(len(predictions)):

            # un-normalize and back-transform
            trueval = np.load(targets[simids[i]])
            prediction = predictions[i]
            for t in range(2):
                trueval[:,:,t] = (trueval[:,:,t] * mean_sd[t][1]) + mean_sd[t][0]
                prediction[:,:,t] = (prediction[:,:,t] * training_mean_sd[t][1]) + training_mean_sd[t][0]
            trueval = np.exp(trueval)
            prediction = np.exp(prediction)
            
            # apply habitat mask  (up front, since you add +1 below to avoid undefined RAE)
            if args.habitat_map is not None:
                trueval = cookie_cutter(trueval, habitat_map, fill=0.0)
                prediction = cookie_cutter(prediction, habitat_map, fill=0.0)
                relevant_pixels = np.sum(habitat_map)
            else:
                relevant_pixels = np.sum(map_width**2)
            
            # save true and pred as arrays
            simid = file_name[simids[i]].split("/")[-1].split(".")[0]
            np.save(args.out + "/Test_" + str(args.seed) + "/mapNN_" + simid + "_true.npy", trueval)
            np.save(args.out + "/Test_" + str(args.seed) + "/mapNN_" + simid + "_pred.npy", prediction)

            # calc. error
            mrae_0,mrae_1,rmse_0,rmse_1,relevant_pixels = 0,0,0,0,0
            for row in range(map_width):  # (whole-matrix operations would run into /0.0)
                for col in range(map_width):
                    if habitat_map is None:
                        mrae_0 += abs(trueval[row,col,0]-prediction[row,col,0])/trueval[row,col,0]
                        mrae_1 += abs(trueval[row,col,1]-prediction[row,col,1])/trueval[row,col,1]
                        rmse_0 += (trueval[row,col,0]-prediction[row,col,0])**2
                        rmse_1 += (trueval[row,col,1]-prediction[row,col,1])**2
                        relevant_pixels += 1
                    elif habitat_map[row,col] == 1:
                        mrae_0 += abs(trueval[row,col,0]-prediction[row,col,0])/trueval[row,col,0]
                        mrae_1 += abs(trueval[row,col,1]-prediction[row,col,1])/trueval[row,col,1]
                        rmse_0 += (trueval[row,col,0]-prediction[row,col,0])**2
                        rmse_1 += (trueval[row,col,1]-prediction[row,col,1])**2
                        relevant_pixels += 1
            # (unindent)
            mrae_0 = np.sum(mrae_0) / relevant_pixels
            mrae_1 = np.sum(mrae_1) / relevant_pixels
            rmse_0 = np.sqrt(np.sum(rmse_0) / relevant_pixels)
            rmse_1 = np.sqrt(np.sum(rmse_1) / relevant_pixels)
            with open(args.out + "/Test_" + str(args.seed) + "/mapNN_" + str(simid) + "_error.txt", "a") as out_f:
                out_f.write(str(mrae_0) + "\t" + str(mrae_1) + "\t" + str(rmse_0) + "\t" + str(rmse_1) + "\n")

            # prepare min and max values for plotting
            # (inside loop, since it sometimes get's min/max from the true map, e.g. when counting realized density)
            if args.ranges is None:                                                                                       
                min_sigma,max_sigma,min_k,max_k = get_min_max(trueval)
            else:                                                                                                         
                min_sigma,max_sigma,min_k,max_k = args.ranges
    
            # convert to (0,1) scale according to user specified ranges
            trueval[:,:,0] = (trueval[:,:,0]-min_sigma) / (max_sigma-min_sigma)
            trueval[:,:,1] = (trueval[:,:,1]-min_k) / (max_k-min_k)
            prediction[:,:,0] = (prediction[:,:,0]-min_sigma) / (max_sigma-min_sigma)
            prediction[:,:,1] = (prediction[:,:,1]-min_k) / (max_k-min_k)

            # convert to PNG format
            trueval *= 255
            trueval = np.round(trueval)
            trueval = np.clip(trueval, 0, 255)
            trueval = trueval.astype('uint8') # (II) int
            prediction *= 255
            prediction = np.round(prediction)
            prediction = np.clip(prediction, 0, 255)
            prediction = prediction.astype('uint8')
 
            # write individual dispersal and density maps (for methods comparison)
            images = []
            im = maplot(trueval[:,:,0], map_width, args.habitat_border)
            im.save(args.out + "/Test_" + str(args.seed) + "/mapNN_" + simid + "_dispersal_true.png")
            images.append(im)
            #
            im = maplot(prediction[:,:,0], map_width, args.habitat_border)
            im.save(args.out + "/Test_" + str(args.seed) + "/mapNN_" + simid + "_dispersal_pred.png")
            images.append(im)
            #
            im = maplot(trueval[:,:,1], map_width, args.habitat_border)
            im.save(args.out + "/Test_" + str(args.seed) + "/mapNN_" + simid + "_density_true.png")
            images.append(im)
            #
            im = maplot(prediction[:,:,1], map_width, args.habitat_border)
            im.save(args.out + "/Test_" + str(args.seed) + "/mapNN_" + simid + "_density_pred.png")
            images.append(im)

            # combined PNG plot
            if args.habitat_border is None:
                w = map_width
            else:
                w = np.array(im).shape[0]
            comb_im = Image.new('RGBA', (w*2,w*2), color=(0,0,0,0))
            x_offset = 0
            comb_im.paste(images[0], (0,0))
            comb_im.paste(images[1], (w,0))
            comb_im.paste(images[2], (0,w))
            comb_im.paste(images[3], (w,w))
            comb_im.save(args.out + "/Test_" + str(args.seed) + "/mapNN_" + simid + "_combined.png")

            # heat map params
            output_file = args.out + "/Test_" + str(args.seed) + "/final_" + str(simid) + ".png"
            tmpfile =  args.out + "/Test_" + str(args.seed) + "/tmp.png"

            # prep locs
            locs = np.load(locs_dict[simids[i]])
            locs[1,:] = map_width - locs[1,:]  # flip y axis
            factor = plot_width / map_width  # rescale
            locs *= factor
            locs = np.floor(locs).astype(int)  # round to nearest pixel, the circle function wants int

            # heatmaps
            cb_params = [min_sigma, max_sigma, "\u03C3"]
            disp_true = heatmap(trueval[:,:,0], plot_width, tmpfile, cb_params, habitat_map_plot, args.habitat_border, locs)
            disp_mapnn = heatmap(prediction[:,:,0], plot_width, tmpfile, None, habitat_map_plot, args.habitat_border, locs)
            cb_params = [min_k,max_k, "D"]
            dens_true = heatmap(trueval[:,:,1], plot_width, tmpfile, cb_params, habitat_map_plot, args.habitat_border, locs)
            dens_mapnn = heatmap(prediction[:,:,1], plot_width, tmpfile, None, habitat_map_plot, args.habitat_border, locs)
            all_together_0  = get_concat_h(disp_true, disp_mapnn)
            all_together_1  = get_concat_h(dens_true, dens_mapnn)
            all_together  = get_concat_v(all_together_0, all_together_1)

            # write                                                                                                     
            all_together.save(output_file)
            
    # empirical
    else:
        # params                                                                                  
        output_pref = args.out + "/Test_" + str(args.seed) + "/empirical_"
        tmpfile =  args.out + "/Test_" + str(args.seed) + "/tmp.png"
        os.makedirs(os.path.join(args.out,"Test_" + str(args.seed)), exist_ok=True)

        # prep locs                                                                               
        locs = np.reshape(np.array(locs_dict),(2,args.n))  # delete extra dim                     
        locs = np.flip(locs, axis=0)
        locs[0,:] = map_width - locs[0,:] # flip x vals
        locs[1,:] = map_width - locs[1,:] # flip y vals
        factor = plot_width / map_width  # rescale                                                
        locs *= factor
        locs = np.floor(locs).astype(int)  # round to nearest pixel, the circle function wants int

        # unnormalize (all reps at once)
        predictions = np.array(predictions)
        for t in range(2):
            predictions[:,:,:,t] = (predictions[:,:,:,t] * mean_sd[t][1]) + mean_sd[t][0]
        predictions = np.exp(predictions)

        # calc mean and var maps
        prediction = np.mean(predictions, axis=0)
        variance = np.std(predictions, axis=0)

        # plot pred and var maps separately
        maps = ["pred","var"]
        for i in range(2):
            if maps[i] == "var" and args.num_reps == 1: # no variance, then no variance map
                break # no variance, then no variance map  
            out_map = [prediction,variance][i]

            # apply mask    TODO: fill with np.nan, use nanmin() nanmax()?
            if args.habitat_map is not None:
                out_map = cookie_cutter(out_map, habitat_map, fill=0.0)
                relevant_pixels = np.sum(habitat_map)
            else:
                relevant_pixels = map_width**2

            # save output
            if maps[i] == "pred":
                np.save(str(args.out) + "/Test_" + str(args.seed) + "/mapNN_empirical_pred.npy", out_map)
                np.savetxt(str(args.out) + "/Test_" + str(args.seed) + "/mapNN_empirical_dispersal_pred.csv", out_map[:,:,0], delimiter=",", fmt='%f')
                np.savetxt(str(args.out) + "/Test_" + str(args.seed) + "/mapNN_empirical_density_pred.csv", out_map[:,:,1], delimiter=",", fmt='%f')            

            # find min and max values for plotting and empirical interpretation
            if args.ranges is None:
                min_sigma,max_sigma,min_k,max_k = get_min_max(out_map,habitat_map)
            else:
                print("maybe misleading to demand a particular range from your empirical data")
                exit()
            if maps[i] == "pred":
                print("    Predictions:")
            else:
                print("    Variance:")
            # (unindent)
            print("sigma range (SLim units):", min_sigma,max_sigma)
            print("k range (SLim units):", min_k,max_k)
            print("mean sigma (SLim units):", np.sum(out_map[:,:,0])/relevant_pixels)
            print("mean K (or density, if you counted that) (SLim units):", np.sum(out_map[:,:,1])/relevant_pixels)

            # convert to (0,1) scale
            out_map[:,:,0] = (out_map[:,:,0]-min_sigma) / (max_sigma-min_sigma)
            out_map[:,:,1] = (out_map[:,:,1]-min_k) / (max_k-min_k)

            # convert to PNG scale+format
            out_map *= 255
            out_map = np.round(out_map)
            out_map = np.clip(out_map, 0, 255)
            out_map = out_map.astype('uint8')

            # dispersal PNG
            im = maplot(out_map[:,:,0], map_width, args.habitat_border)
            im.save(str(args.out) + "/Test_" + str(args.seed) + "/mapNN_empirical_dispersal_" + maps[i] + ".png")   

            # density PNG
            im = maplot(out_map[:,:,1], map_width, args.habitat_border)
            im.save(args.out + "/Test_" + str(args.seed) + "/mapNN_empirical_density_" + maps[i] + ".png") 
            
            # dispersal heatmap
            tmpfile =  args.out + "/Test_" + str(args.seed) + "/tmp_1.png"
            cb_params = [min_sigma, max_sigma, "\u03C3"]
            disp_map = heatmap(out_map[:,:,0], plot_width, tmpfile, cb_params, habitat_map_plot, args.habitat_border, locs)

            # density heatmap
            cb_params = [min_k,max_k, "D"]
            dens_map = heatmap(out_map[:,:,1], plot_width, tmpfile, cb_params, habitat_map_plot, args.habitat_border, locs)            
            
            # merge pngs
            all_together  = get_concat_h(disp_map, dens_map)

            # write
            all_together.save(output_pref + maps[i] + ".png")

    return




def preprocess():
    trees = read_list(args.tree_list)
    maps = read_list(args.target_list)
    total_sims = len(trees)

    # empirical locations                                   
    if args.empirical != None:
        locs = read_locs(args.empirical + ".locs")
        if len(locs) != args.n:
            print("length of locs file doesn't match max_n")
            exit()
    else:
        locs = None

    # read in habitat map
    if args.habitat_map is None:
        num_relevant_pixels = args.map_width**2
    else:
        habitat_map = read_habitat_map(args.habitat_map, args.map_width)
        num_relevant_pixels = np.sum(habitat_map)


    # loop through maps to get mean and sd          
    if args.training_params is not None:
        stats = np.load(args.training_params)
    elif os.path.isfile(args.out+"/mean_sd.npy"):
        stats = np.load(args.out+"/mean_sd.npy")
    else:
        # loop through all maps to get mean
        means_summed_disp = 0
        means_summed_dens = 0
        for i in range(total_sims):
            print("getting mean from training, on sim", i)
            arr = read_map(maps[i], args.map_width)
            if args.habitat_map != None: # 
                arr = cookie_cutter(arr, habitat_map, fill=np.nan, fxn=np.log)
            else:  # this strategy avoids log(0)'s
                arr = np.log(arr)      
            # (unindent)
            means_summed_disp += np.nansum(arr[:,:,0])
            means_summed_dens += np.nansum(arr[:,:,1])
            # (unindent)
        mean_disp = means_summed_disp / num_relevant_pixels / total_sims
        mean_dens = means_summed_dens / num_relevant_pixels / total_sims

        # loop through second time to get sd
        sd_summed_disp = 0
        sd_summed_dens = 0
        for i in range(total_sims):
            print("getting sd from training, on sim", i)
            arr = read_map(maps[i], args.map_width)
            if args.habitat_map != None:
                arr = cookie_cutter(arr, habitat_map, fill=np.nan, fxn=np.log)
            else:  # this strategy avoids log(0)'s
                arr = np.log(arr)
            # (unindent)
            sd_summed_disp += np.nansum((arr[:,:,0] - mean_disp)**2)
            sd_summed_dens += np.nansum((arr[:,:,1] - mean_dens)**2)
            # (unindent)
        sd_disp = (sd_summed_disp / num_relevant_pixels / total_sims)**(0.5)
        sd_dens = (sd_summed_dens / num_relevant_pixels / total_sims)**(0.5)
        stats = []
        stats.append(np.array([mean_disp, sd_disp]))
        stats.append(np.array([mean_dens, sd_dens]))
        os.makedirs(args.out, exist_ok=True)
        np.save(args.out+"/mean_sd", stats) 
    
    # initialize generator and some things
    os.makedirs(os.path.join(args.out,"Maps",str(args.seed)), exist_ok=True)
    os.makedirs(os.path.join(args.out,"Genos",str(args.seed)), exist_ok=True)
    os.makedirs(os.path.join(args.out,"Locs",str(args.seed)), exist_ok=True)
    params = make_generator_params_dict(
        targets=None,
        shuffle=None,
        genos=None,
        locs=None,
        empirical_locs=locs,
        map_width=args.map_width,
    )
    training_generator = DataGenerator([None], **params)

    # preprocess
    if args.simid is None:
        for i in range(total_sims):
            mapfile = os.path.join(args.out,"Maps",str(args.seed),str(i)+".target")
            genofile = os.path.join(args.out,"Genos",str(args.seed),str(i)+".genos")
            locfile = os.path.join(args.out,"Locs",str(args.seed),str(i)+".locs")
            if os.path.isfile(genofile+".npy") is False or os.path.isfile(locfile+".npy") is False:
                geno_mat, locs = training_generator.sample_ts(trees[i], args.seed) 
                np.save(genofile, geno_mat)
                np.save(locfile, locs)
            if os.path.isfile(genofile+".npy") is True and os.path.isfile(locfile+".npy") is True: # only add map if inputs successful
                if os.path.isfile(mapfile+".npy") is False:
                    target = read_map(maps[i], args.map_width)
                    if args.habitat_map != None:
                        target = cookie_cutter(target, habitat_map, fill=np.nan, fxn=np.log)
                    else:  # this strategy avoids log(0)'s                                  
                        target = np.log(target)
                    for t in range(2):
                        target[:,:,t] = (target[:,:,t] - stats[t][0]) / stats[t][1]
                    if args.habitat_map != None:
                        target = cookie_cutter(target, habitat_map, fill=0)
                    np.save(mapfile, target)
    else:
        mapfile = os.path.join(args.out,"Maps",str(args.seed),str(args.simid)+".target")
        genofile = os.path.join(args.out,"Genos",str(args.seed),str(args.simid)+".genos")
        locfile = os.path.join(args.out,"Locs",str(args.seed),str(args.simid)+".locs")
        if os.path.isfile(genofile+".npy") is False or os.path.isfile(locfile+".npy") is False:
            geno_mat, locs = training_generator.sample_ts(trees[args.simid-1], args.seed) # -1 for 0-indexing
            np.save(genofile, geno_mat)
            np.save(locfile, locs)
        if os.path.isfile(genofile+".npy") is True and os.path.isfile(locfile+".npy") is True: # only add map if inputs successful
            if os.path.isfile(mapfile+".npy") is False:
                target = read_map(maps[args.simid-1], args.map_width)
                if args.habitat_map != None:
                    target = cookie_cutter(target, habitat_map, fill=np.nan, fxn=np.log)
                else:  # this strategy avoids log(0)'s                            
                    target = np.log(target)
                for t in range(2):
                    target[:,:,t] = (target[:,:,t] - stats[t][0]) / stats[t][1]   
                if args.habitat_map != None:
                    target = cookie_cutter(target, habitat_map, fill=0)
                np.save(mapfile, target)                                                             

    return



def preprocess_density_grid():
    trees = read_list(args.tree_list)
    maps = read_list(args.target_list)
    counts = read_list(args.counts_list)
    total_simids = len(trees)
    os.makedirs(os.path.join(args.out,"Maps",str(args.seed)), exist_ok=True)
    os.makedirs(os.path.join(args.out,"Genos",str(args.seed)), exist_ok=True)
    os.makedirs(os.path.join(args.out,"Locs",str(args.seed)), exist_ok=True)

    # empirical locations                                   
    if args.empirical != None:
        locs = read_locs(args.empirical + ".locs")
        if len(locs) != args.n:
            print("length of locs file doesn't match max_n")
            exit()
    else:
        locs = None

    # preprocess inputs
    genofile = os.path.join(args.out,"Genos",str(args.seed),str(args.simid)+".genos")
    locfile = os.path.join(args.out,"Locs",str(args.seed),str(args.simid)+".locs")
    if os.path.isfile(genofile+".npy") is False or os.path.isfile(locfile+".npy") is False:
        params = make_generator_params_dict(
            targets=None,
            shuffle=None,
            genos=None,
            locs=None,
            empirical_locs=locs,
            map_width=args.map_width,
        )
        training_generator = DataGenerator([None], **params)
        geno_mat, locs = training_generator.sample_ts(trees[args.simid-1], args.seed) # -1 for 0-indexing   
        np.save(genofile, geno_mat)
        np.save(locfile, locs)

    # pipeline to normalize target
    targetfile = os.path.join(args.out,"Maps",str(args.seed),str(args.simid)+".target")  # where to write the normalized target
    if os.path.isfile(targetfile+".npy") is False:

        # get mean and sd
        if args.training_params is not None:
            stats = np.load(args.training_params)
            max_disp = stats[0,2]
            max_dens = stats[1,2]
        elif os.path.isfile(args.out+"/mean_sd.npy"):
            stats = np.load(args.out+"/mean_sd.npy")
            max_disp = stats[0,2]
            max_dens = stats[1,2]
        else:  # no mean sd provided 

            # load habitat map
            if args.habitat_map is None:
                num_relevant_pixels = args.map_width**2
            else:
                habitat_map = read_habitat_map(args.habitat_map, args.map_width)
                num_relevant_pixels = np.sum(habitat_map)

            # loop through all maps to get mean                                    
            means_summed_disp = 0
            means_summed_dens = 0
            max_disp = 0 # for PNG (see below)
            max_dens = 0
            total_counted = 0
            for i in range(total_simids):
                counts_file = counts[args.simid-1]
                if os.path.isfile(counts_file):
                    total_counted += 1
                    arr = grid_density(counts_file,args.slim_width)
                    if np.max(arr[:,:,0]) > max_disp:  # (for plotting)
                        max_disp = np.max(arr[:,:,0])
                    if np.max(arr[:,:,1]) > max_dens:
                        max_dens = np.max(arr[:,:,1])
                    if args.habitat_map != None:
                        arr = cookie_cutter(arr, habitat_map, fill=0.0, fxn=np.log)
                    else:  # this strategy avoids log(0)'s
                        arr = np.log(arr)
                    # (unindent)                                                       
                    means_summed_disp += np.nansum(arr[:,:,0])
                    means_summed_dens += np.nansum(arr[:,:,1])
            # (unindent)                                                       
            mean_disp = means_summed_disp / num_relevant_pixels / total_counted
            mean_dens = means_summed_dens / num_relevant_pixels / total_counted

            # loop through second time to get sd                                   
            sd_summed_disp = 0
            sd_summed_dens = 0
            for i in range(total_simids):
                counts_file = counts[args.simid-1]
                if os.path.isfile(counts_file):
                    arr = grid_density(counts_file,args.slim_width)
                    if args.habitat_map != None:
                        arr = cookie_cutter(arr, habitat_map, fill=np.nan, fxn=np.log)
                    else:  # this strategy avoids log(0)'s
                        arr = np.log(arr)
                    # (unindent)                                                       
                    sd_summed_disp += np.nansum((arr[:,:,0] - mean_disp)**2)
                    sd_summed_dens += np.nansum((arr[:,:,1] - mean_dens)**2)
            # (unindent)                                                       
            sd_disp = (sd_summed_disp / num_relevant_pixels / total_counted)**(0.5)
            sd_dens = (sd_summed_dens / num_relevant_pixels / total_counted)**(0.5)
            stats = []
            stats.append(np.array([mean_disp, sd_disp, max_disp]))
            stats.append(np.array([mean_dens, sd_dens, max_dens]))
            os.makedirs(args.out, exist_ok=True)
            np.save(args.out+"/mean_sd", stats)

        # normalize
        counts_file = counts[args.simid-1]
        arr = grid_density(counts_file,args.slim_width)
        if args.habitat_map is not None:
            arr = cookie_cutter(arr, habitat_map, fill=np.nan, fxn=np.log)  # nan
        else:
            arr = np.log(arr)
        for t in range(2): 
            arr[:,:,t] = (arr[:,:,t] - stats[t][0]) / stats[t][1]
        if args.habitat_map is not None:
            arr = cookie_cutter(arr, habitat_map, fill=0.0)  # zero
        np.save(targetfile, arr)

        # also visualize PNG of the raw  counts, rescaled to (0,255)
        vals = arr[:,:,0]  
        vals *= stats[0][1]
        vals += stats[0][0]
        vals = np.exp(vals)
        vals /= max_disp 
        vals *= 255
        vals = np.floor(vals)
        if args.habitat_map != None: # (only relevant for the png output) 
            vals = cookie_cutter(vals, habitat_map, fill=0.0)
        im = maplot(vals, args.map_width, args.habitat_border)
        pngfile = os.path.join(args.out,"Maps",str(args.seed),str(args.simid)+".dispersal.png")
        im.save(pngfile)
        #
        vals = arr[:,:,1]
        vals *= stats[1][1]
        vals += stats[1][0]
        vals = np.exp(vals)
        vals /= max_dens
        vals *= 255
        vals = np.floor(vals)
        if args.habitat_map != None: # (only relevant for the png output)
            vals = cookie_cutter(vals, habitat_map, fill=0.0)
        im = maplot(vals, args.map_width, args.habitat_border)
        pngfile = os.path.join(args.out,"Maps",str(args.seed),str(args.simid)+".density.png")
        im.save(pngfile)

    return




def plot_history():
    loss, val_loss = [], [
        np.nan
    ]  # loss starts at iteration 0; val_loss starts at end of first epoch
    with open(args.plot_history) as infile:
        for line in infile:
            if "val_loss:" in line:
                endofline = line.strip().split(" loss:")[-1]
                loss.append(float(endofline.split()[0]))
                val_loss.append(float(endofline.split()[3]))
    loss.append(np.nan)  # make columns even-length
    epochs = np.arange(len(loss))
    fig = plt.figure(figsize=(4, 1.5), dpi=200)
    plt.rcParams.update({"font.size": 7})
    ax1 = fig.add_axes([0, 0, 0.4, 1])
    ax1.plot(epochs, val_loss, color="blue", lw=0.5, label="val_loss")
    ax1.set_xlabel("Epoch")
    ax1.plot(epochs, loss, color="red", lw=0.5, label="loss")
    ax1.legend()
    fig.savefig(args.plot_history + "_plot.pdf", bbox_inches="tight")
    return



def ci():
    map_width = 50
    plot_width = 500
    ci_file = args.out + "/Test_" + str(args.seed) + "/CIs.txt"
    if args.habitat_map is not None:
        habitat_map = read_habitat_map(args.habitat_map, map_width)
        habitat_map_plot = read_habitat_map(args.habitat_map, plot_width)

    # load inputs
    if args.simid is None:
        targets,genos,locs = preds_from_preprocessed(args.out)
        total_sims = len(targets)
    else:
        targets = [args.out + "/Maps/" + str(args.seed) + "/" + str(args.simid) + ".target.npy"]
        genos = [args.out + "/Genos/" + str(args.seed) +"/" + str(args.simid) +".genos.npy"]
        locs = [args.out + "/Locs/" + str(args.seed) +"/" + str(args.simid) +".locs.npy"]
        total_sims = 1

    # loop through preds
    R = len(targets) 
    sampling_dist = np.zeros((50,50,2,R))
    for r in range(R):
        f = "/home/chriscs/Boxes116_wolves_v4_bs/Test_1/mapNN_" + str(r+1) + "_pred.npy"
        pred = np.load(f)
        sampling_dist[:,:,:,r] = pred
    
    # pixel wise intervals
    alpha = 0.05
    interval_map = np.zeros((50,50,2))
    f = "/home/chriscs/Boxes116_wolves_v4_bs/Test_1/mapNN_1_pred.npy"  # same for all
    true = np.load(f)
    with open(ci_file,"w") as outfile:
        for i in range(50):
            for j in range(50):
                for p in range(2):
                    # confidence interval
                    dist = sampling_dist[i,j,p]
                    dist.sort()
                    thetahat_low =  dist[int(np.floor( R*   (alpha/2.0))) -1]
                    thetahat_high = dist[int(np.ceil(  R*(1-(alpha/2.0))))  ]
                    lower = 2*true[i,j,p] - thetahat_high
                    upper = 2*true[i,j,p] - thetahat_low
                    interval_map[i,j,p] = upper-lower
                    
                    # write
                    outline = [["dispersal","density"][p]]
                    outline += [i,j]
                    outline.append(true[i,j,p])
                    outline.append(lower)
                    outline.append(upper)
                    outfile.write("\t".join(map(str,outline)) + "\n")

    # find min and max values for plotting
    min_sigma,max_sigma,min_k,max_k = get_min_max(interval_map,habitat_map)
                    
    # convert to (0,1) scale 
    interval_map[:,:,0] = (interval_map[:,:,0]-min_sigma) / (max_sigma-min_sigma)
    interval_map[:,:,1] = (interval_map[:,:,1]-min_k) / (max_k-min_k)
        
    # convert to PNG format                                                                                             
    interval_map *= 255
    interval_map = np.round(interval_map)
    interval_map = np.clip(interval_map, 0, 255)
    interval_map = interval_map.astype('uint8') # (II) int

    # dispersal CIs
    tmpfile =  args.out + "/Test_" + str(args.seed) + "/tmp_1.png"
    cb_params = [min_sigma, max_sigma, "\u03C3"]
    disp_cis = heatmap(interval_map[:,:,0], plot_width, tmpfile, cb_params, habitat_map_plot, args.habitat_border)

    # density CIs
    cb_params = [min_k, max_k, "D"]
    dens_cis = heatmap(interval_map[:,:,1], plot_width, tmpfile, cb_params, habitat_map_plot, args.habitat_border)
    
    # combine
    all_together  = get_concat_h(disp_cis, dens_cis)
    
    # write                                                                                                                                                         
    output_file = args.out + "/Test_" + str(args.seed) + "/empirical_cis.png"
    all_together.save(output_file)
    
    return
    
    
### main ###
np.random.seed(args.seed)

# pre-process
if args.preprocess is True:
    print("starting pre-processing pipeline")
    preprocess()
elif args.preprocess_density_grid is True:
    preprocess_density_grid()

# train
if args.train is True:
    print("starting training pipeline")
    train()

# plot training history
if args.plot_history is not False:  # check for plot_history=file path
    plot_history()
    
# predict
if args.predict is True:
    print("starting prediction pipeline")
    if args.empirical is None:
        print("predicting on simulated data")
        predict()
    else:
        print("predicting on empirical data")
        empirical()

# get pixel-wise confidence intervals
if args.bootstrap is True:
    print("using bootstrap replicates to calculate CIs")
    ci()
