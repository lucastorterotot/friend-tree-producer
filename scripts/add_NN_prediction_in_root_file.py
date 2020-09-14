#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
times = [time.time()]

from optparse import OptionParser
usage = "usage: %prog [options] <json file for NN> <input root file>"
parser = OptionParser(usage=usage)
parser.add_option("-s", "--suffix", dest = "suffix",
                                    default="NN")
parser.add_option("-v", "--verbose", dest = "verbose",
                                    default=0)

(options,args) = parser.parse_args()

NNname = args[0].split('/')[-1].replace('.json', '').replace("-","_")

# load json and create model
times.append(time.time())

from keras.models import model_from_json

input_json = args[0]
NN_weights_path_and_file = input_json.split('/')
NN_weights_path_and_file[-1] = "NN_weights-{}".format(NN_weights_path_and_file[-1].replace('.json', '.h5'))
NN_weights_file = "/".join(NN_weights_path_and_file)

json_file = open(input_json, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(NN_weights_file)
print("Loaded model from disk:")
print("\t{}".format(input_json))

# Get infos on the trained NN
infos = NN_weights_path_and_file[-1]
infos = infos.replace('.h5', '')
infos = infos.replace('NN_weights-', '')

is_bottleneck = ("-bottleneck" == infos[-11:])

bottleneck = ""
if is_bottleneck:
    infos = infos.replace('-bottleneck', '')
    bottleneck = "-bottleneck"

Nneurons = infos.split("-")[-2]
Nlayers = infos.split("-")[-4]
channel = infos.split("-")[-5]

w_init_mode = infos.split("-")[-6]
optimizer = infos.split("-")[-7]
loss = infos.split("-")[-8]

if options.verbose > 0:
    print("Properties:")

    print(
        "\t{} channel, {} hidden layers of {} neurons with{} bottleneck".format(
            channel,
            Nlayers,
            Nneurons,
            "" if is_bottleneck else "out",
        )
    )
    print(
        "\ttrained with {} optimizer, w_init {} and {} loss.".format(
            optimizer,
            w_init_mode,
            loss,
        )
    )
    
times.append(time.time())

# load root file
import uproot

root_file_input = args[1]
#root_file_input = "/data2/htt/trees/fakes/190819%HiggsSUSYGG450%mt_mssm_nominal/NtupleProducer/tree.root" # for debugging without having to provide args
root_file_output = root_file_input.replace('.root', '-{}.root'.format(options.suffix))

root_file_in = uproot.open(root_file_input)
df = root_file_in['events'].pandas.df()

import numpy as np
df["mt_tt"] = (2*df["l1_pt"]*df["l2_pt"]*(1-np.cos(df["l1_phi"]-df["l2_phi"])))**.5

inputs = [
    "tau1_pt",
    "tau1_eta",
    "tau1_phi",
    "tau2_pt",
    "tau2_eta",
    "tau2_phi",
    # "jet1_pt",
    # "jet1_eta",
    # "jet1_phi",
    # "jet2_pt",
    # "jet2_eta",
    # "jet2_phi",
    # "recoil_pt",
    # "recoil_eta",
    # "recoil_phi",
    "MET_pt",
    "MET_phi",
    "MET_covXX",
    "MET_covXY",
    "MET_covYY",
    # "MET_significance",
    "mT1",
    "mT2",
    "mTtt",
    "mTtot",
    ]

times.append(time.time())
df["predictions_{}".format(NNname)] = loaded_model.predict(df[inputs])
times.append(time.time())

tree_dtype = {}
tree_data = {}
for b in df.keys():
    tree_dtype[b] = df[b].dtype.name
    if tree_dtype[b] == 'uint64':
        tree_dtype[b] = 'int64'
    tree_data[b] = np.array(df[b])

root_file_out = uproot.recreate(root_file_output)
print("Opened new file")
root_file_out.newtree('events', tree_dtype)
print("Created new tree")
root_file_out['events'].extend(tree_data)
print("New tree filled")


times.append(time.time())

print("Time sumary:")
print("\t- Loading of the NN: {} s;".format(np.round(times[-4]-times[-5], 3)))
print("\t- Loading of the root file: {} s;".format(np.round(times[-3]-times[-4], 3)))
print("\t- Computing predictions on {} events: {} s <=> {} s for 10k events.".format(df.shape[0], np.round(times[-2]-times[-3],3), np.round((times[-2]-times[-3])/df.shape[0]*10000, 3)))
print("\t- Creating new root file: {} s <=> {} s for 10k events.".format(np.round(times[-1]-times[-2],3), np.round((times[-1]-times[-2])/df.shape[0]*10000,3)))
