#!/usr/bin/env python

import ROOT as r
import glob
import argparse
import json
import os
import numpy as np
import stat
import re
import copy
import time
from multiprocessing import Pool


r.gROOT.ProcessLine( "gErrorIgnoreLevel = 2001;")

shellscript_template = '''#!/bin/sh
ulimit -s unlimited
set -e

cd {TASKDIR}

{COMMANDS}

'''

command_template = '''
if [ $1 -eq {JOBNUMBER} ]; then
    {COMMAND}
fi
'''

def write_trees_to_files(info):
    nick = info[0]
    collection_path = info[1]
    db = info[2]
    print "Copying trees for %s"%nick
    nick_path = os.path.join(collection_path,nick)
    if not os.path.exists(nick_path):
        os.mkdir(nick_path)
    outputfile = r.TFile.Open(os.path.join(nick_path,nick+".root"),"recreate")
    for p in db[nick]["pipelines"]:
        if db[nick]["pipelines"][p] > 0:
            outputfile.mkdir(p)
            outputfile.cd(p)
            tree = db[nick][p].CloneTree()
            tree.Write("",r.TObject.kOverwrite)
            db[nick][p].Reset()
    outputfile.Close()

def check_output_files(f):
    valid_file = True
    if not os.path.exists(f):
        valid_file = False
        print "File not there:",f
    else:
        F = r.TFile.Open(f, "read")
        if F:
            valid_file = not F.IsZombie() and not F.TestBit(r.TFile.kRecovered)
            F.Close()
        else:
            F.Close()
            valid_file = False
        if not valid_file:
            print "File is corrupt: ",f
            os.remove(f)
    return valid_file

def prepare_jobs(input_ntuples_list, inputs_base_folder, inputs_friends_folders, events_per_job, batch_cluster, executable, walltime, max_jobs_per_batch, custom_workdir_path, restrict_to_channels, restrict_to_shifts):
    ntuple_database = {}
    for f in input_ntuples_list:
        restrict_to_channels_file = copy.deepcopy(restrict_to_channels)
        nick = f.split("/")[-1].replace(".root","")
        print "Saving inputs for %s"%nick
        if "SingleMuon_Run" in nick or "MuTauFinalState" in nick:
            restrict_to_channels_file = list(set(['mt']).intersection(restrict_to_channels_file)) if len(restrict_to_channels_file) > 0 else ['mt']
            print "\tWarning: restrict %s to '%s' channel(s)"%(nick,restrict_to_channels_file)
        if "SingleElectron_Run" in nick or "EGamma_Run" in nick or "ElTauFinalState" in nick:
            restrict_to_channels_file = list(set(['et']).intersection(restrict_to_channels_file)) if len(restrict_to_channels_file) > 0 else ['et']
            print "\tWarning: restrict %s to '%s' channel(s)"%(nick,restrict_to_channels_file)
        if "Tau_Run" in nick or "TauTauFinalState" in nick:
            restrict_to_channels_file = list(set(['tt']).intersection(restrict_to_channels_file)) if len(restrict_to_channels_file) > 0 else ['tt']
            print "\tWarning: restrict %s to '%s' channel(s)"%(nick,restrict_to_channels_file)
        if "MuonEG_Run" in nick or "ElMuFinalState" in nick:
            restrict_to_channels_file = list(set(['em']).intersection(restrict_to_channels_file)) if len(restrict_to_channels_file) > 0 else ['em']
            print "\tWarning: restrict %s to '%s' channel(s)"%(nick,restrict_to_channels_file)
        ntuple_database[nick] = {}
        ntuple_database[nick]["path"] = f
        F = r.TFile.Open(f,"read")
        pipelines = [k.GetName() for k in F.GetListOfKeys()]
        if len(restrict_to_channels_file) > 0 or (len(restrict_to_channels_file) == 0 and len(restrict_to_channels) > 0):
            pipelines = [p for p in pipelines if p.split("_")[0] in restrict_to_channels_file]
        if len(restrict_to_shifts) > 0:
            pipelines = [p for p in pipelines if p.split("_")[1] in restrict_to_shifts]
        ntuple_database[nick]["pipelines"] = {}
        for p in pipelines:
            ntuple_database[nick]["pipelines"][p] = F.Get(p).Get("ntuple").GetEntries()
    job_database = {}
    job_number = 0
    for nick in ntuple_database:
        for p in ntuple_database[nick]["pipelines"]:
            n_entries = ntuple_database[nick]["pipelines"][p]
            if n_entries > 0:
                entry_list = np.append(np.arange(0,n_entries,events_per_job),[n_entries])
                first_entries = entry_list[:-1]
                last_entries = entry_list[1:] -1
                for first,last in zip(first_entries, last_entries):
                    job_database[job_number] = {}
                    job_database[job_number]["input"] = ntuple_database[nick]["path"]
                    job_database[job_number]["folder"] = p
                    job_database[job_number]["tree"] = "ntuple"
                    job_database[job_number]["first_entry"] = first
                    job_database[job_number]["last_entry"] = last
                    channel = p.split("_")[0]
                    if channel in inputs_friends_folders.keys() and len(inputs_friends_folders[channel])>0:
                        job_database[job_number]["input_friends"] = " ".join([job_database[job_number]["input"].replace(inputs_base_folder, friend_folder) for friend_folder in inputs_friends_folders[channel]])
                    job_number +=1
            else:
                print "Warning: %s has no entries in pipeline %s"%(nick,p)
    if custom_workdir_path:
        workdir_path = os.path.join(custom_workdir_path,executable+"_workdir")
    else:
        workdir_path = os.path.join(os.environ["CMSSW_BASE"],"src",executable+"_workdir")
    if not os.path.exists(workdir_path):
        os.mkdir(workdir_path)
    if not os.path.exists(os.path.join(workdir_path,"logging")):
        os.mkdir(os.path.join(workdir_path,"logging"))
    commandlist = []
    for jobnumber in job_database:
        options = " ".join(["--"+k+" "+str(v) for (k,v) in job_database[jobnumber].items()])
        commandline = "{EXEC} {OPTIONS}".format(EXEC=executable, OPTIONS=options)
        command = command_template.format(JOBNUMBER=str(jobnumber), COMMAND=commandline)
        commandlist.append(command)
    commands = "\n".join(commandlist)
    shellscript_content = shellscript_template.format(COMMANDS=commands,TASKDIR=workdir_path)
    executable_path = os.path.join(workdir_path,"condor_"+executable+".sh")
    jobdb_path = os.path.join(workdir_path,"condor_"+executable+".json")
    datasetdb_path = os.path.join(workdir_path,"dataset.json")
    with open(executable_path,"w") as shellscript:
        shellscript.write(shellscript_content)
        os.chmod(executable_path, os.stat(executable_path).st_mode | stat.S_IEXEC)
        shellscript.close()
    condorjdl_template_path = os.path.join(os.environ["CMSSW_BASE"],"src/HiggsAnalysis/friend-tree-producer/data/submit_condor_%s.jdl"%batch_cluster)
    condorjdl_template_file = open(condorjdl_template_path,"r")
    condorjdl_template = condorjdl_template_file.read()

    gc_template_path = os.path.join(os.environ["CMSSW_BASE"],"src/HiggsAnalysis/friend-tree-producer/data/grid-control_%s.conf"%batch_cluster)
    gc_template_file = open(gc_template_path, "r")
    gc_template = gc_template_file.read()
    if walltime > 0:
        if walltime < 86399:
            gc_content = gc_template.format(TASKDIR=workdir_path,EXECUTABLE=executable_path,WALLTIME=time.strftime("%H:%M:%S", time.gmtime(walltime)),NJOBS=job_number)
        else: 
            print "Warning: Please set walltimes greater than 24 hours manually in gc config."
            gc_content = gc_template.format(TASKDIR=workdir_path,EXECUTABLE=executable_path,WALLTIME="24:00:00",NJOBS=job_number)
    else:
        print "Warning: walltime for %s cluster not set. Setting it to 1h."%batch_cluster
        gc_content = gc_template.format(TASKDIR=workdir_path,EXECUTABLE=executable_path,WALLTIME="1:00:00",NJOBS=job_number)
    gc_path =  os.path.join(workdir_path,"grid_control_{}_{}.conf".format(executable,batch_cluster))
    with open(gc_path, "w") as gc:
        gc.write(gc_content)
        gc.close()


    argument_borders = np.append(np.arange(0,job_number,max_jobs_per_batch),[job_number])
    first_borders = argument_borders[:-1]
    last_borders = argument_borders[1:] -1
    printout_list = []
    for index, (first,last) in enumerate(zip(first_borders,last_borders)):
        condorjdl_path = os.path.join(workdir_path,"condor_"+executable+"_%d.jdl"%index)
        argument_list = np.arange(first,last+1)
        if not os.path.exists(os.path.join(workdir_path,"logging", str(index))):
            os.mkdir(os.path.join(workdir_path,"logging", str(index)))
        arguments_path = os.path.join(workdir_path,"arguments_%d.txt"%(index))
        with open(arguments_path, "w") as arguments_file:
            arguments_file.write("\n".join([str(arg) for arg in argument_list]))
            arguments_file.close()
        njobs = "arguments from arguments_%d.txt"%(index)
        if batch_cluster in  ["etp6","etp7","lxplus6","lxplus7"]:
            if walltime > 0:
                condorjdl_content = condorjdl_template.format(TASKDIR=workdir_path,TASKNUMBER=str(index),EXECUTABLE=executable_path,NJOBS=njobs,WALLTIME=str(walltime))
            else:
                print "Warning: walltime for %s cluster not set. Setting it to 1h."%batch_cluster
                condorjdl_content = condorjdl_template.format(TASKDIR=workdir_path,TASKNUMBER=str(index),EXECUTABLE=executable_path,NJOBS=njobs,WALLTIME=str(3600))
        else:
            condorjdl_content = condorjdl_template.format(TASKDIR=workdir_path,TASKNUMBER=str(index),EXECUTABLE=executable_path,NJOBS=njobs)
        with open(condorjdl_path,"w") as condorjdl:
            condorjdl.write(condorjdl_content)
            condorjdl.close()
        printout_list.append("cd {TASKDIR}; condor_submit {CONDORJDL}".format(TASKDIR=workdir_path, CONDORJDL=condorjdl_path))
    print
    print "To run the condor submission, execute the following:"
    print
    print "\n".join(printout_list)
    print
    print "Or with grid-control:"
    print "go.py {} -Gc".format(gc_path)
    print
    with open(jobdb_path,"w") as db:
        db.write(json.dumps(job_database, sort_keys=True, indent=2))
        db.close()
    with open(datasetdb_path,"w") as datasets:
        datasets.write(json.dumps(ntuple_database, sort_keys=True, indent=2))
        datasets.close()

def collect_outputs(executable,cores,custom_workdir_path):
    if custom_workdir_path:
        workdir_path = os.path.join(custom_workdir_path,executable+"_workdir")
    else:
        workdir_path = os.path.join(os.environ["CMSSW_BASE"],"src",executable+"_workdir")
    jobdb_path = os.path.join(workdir_path,"condor_"+executable+".json")
    datasetdb_path = os.path.join(workdir_path,"dataset.json")
    jobdb_file = open(jobdb_path,"r")
    jobdb = json.loads(jobdb_file.read())
    datasetdb_file = open(datasetdb_path,"r")
    datasetdb = json.loads(datasetdb_file.read())
    collection_path = os.path.join(workdir_path,executable+"_collected")
    if not os.path.exists(collection_path):
        os.mkdir(collection_path)
    for jobnumber in sorted([int(k) for k in jobdb]):
        nick = jobdb[str(jobnumber)]["input"].split("/")[-1].replace(".root","")
        pipeline = jobdb[str(jobnumber)]["folder"]
        tree = jobdb[str(jobnumber)]["tree"]
        first = jobdb[str(jobnumber)]["first_entry"]
        last = jobdb[str(jobnumber)]["last_entry"]
        filename = "_".join([nick,pipeline,str(first),str(last)])+".root"
        filepath = os.path.join(workdir_path,nick,filename)
        datasetdb[nick].setdefault(pipeline,r.TChain("/".join([pipeline,tree]))).Add(filepath)

    nicks = sorted(datasetdb)
    pool = Pool(cores)
    pool.map(write_trees_to_files, zip(nicks,[collection_path]*len(nicks), [datasetdb]*len(nicks)))

def check_and_resubmit(executable,custom_workdir_path):
    if custom_workdir_path:
        workdir_path = os.path.join(custom_workdir_path,executable+"_workdir")
    else:
        workdir_path = os.path.join(os.environ["CMSSW_BASE"],"src",executable+"_workdir")
    jobdb_path = os.path.join(workdir_path,"condor_"+executable+".json")
    datasetdb_path = os.path.join(workdir_path,"dataset.json")
    jobdb_file = open(jobdb_path,"r")
    jobdb = json.loads(jobdb_file.read())
    datasetdb_file = open(datasetdb_path,"r")
    datasetdb = json.loads(datasetdb_file.read())
    arguments_path = os.path.join(workdir_path,"arguments_resubmit.txt")
    job_to_resubmit = []
    for jobnumber in sorted([int(k) for k in jobdb]):
        nick = jobdb[str(jobnumber)]["input"].split("/")[-1].replace(".root","")
        pipeline = jobdb[str(jobnumber)]["folder"]
        tree = jobdb[str(jobnumber)]["tree"]
        first = jobdb[str(jobnumber)]["first_entry"]
        last = jobdb[str(jobnumber)]["last_entry"]
        filename = "_".join([nick,pipeline,str(first),str(last)])+".root"
        filepath = os.path.join(workdir_path,nick,filename)
        if not check_output_files(filepath):
            job_to_resubmit.append(jobnumber)
    with open(arguments_path, "w") as arguments_file:
        arguments_file.write("\n".join([str(arg) for arg in job_to_resubmit]))
        arguments_file.close()
    if not os.path.exists(os.path.join(workdir_path,"logging", "remaining")):
        os.mkdir(os.path.join(workdir_path,"logging", "remaining"))
    condor_jdl_path = os.path.join(workdir_path,"condor_"+executable+"_0.jdl")
    with open(condor_jdl_path, "r") as file:
        condor_jdl_resubmit = file.read()
    condor_jdl_resubmit_path = os.path.join(workdir_path,"condor_"+executable+"_resubmit.jdl")
    condor_jdl_resubmit = re.sub("\_0.txt","_resubmit.txt",condor_jdl_resubmit).replace("/0/","/remaining/")
    with open(condor_jdl_resubmit_path, "w") as file:
        file.write(condor_jdl_resubmit)
        file.close
    print
    print "To run the resubmission, check {} first".format(condor_jdl_resubmit_path)
    print "Command:"
    print "cd {TASKDIR}; condor_submit {CONDORJDL}".format(TASKDIR=workdir_path, CONDORJDL=condor_jdl_resubmit_path)


def extract_friend_paths(packed_paths):
    extracted_paths = {
        "em" : [],
        "et" : [],
        "mt" : [],
        "tt" : []
        }
    for pathname in packed_paths:
        splitpath=pathname.split('{')
        et_path, mt_path, tt_path = splitpath[0], splitpath[0], splitpath[0]
        path_per_ch = {}
        for channel in extracted_paths.keys():
            path_per_ch[channel] = splitpath[0]
        first = True
        for split in splitpath:
            if first:
                first = False
            else:
                subsplit = split.split('}')
                chdict=json.loads('{"' + subsplit[0].replace(':', '":"').replace(',', '","') + '"}')
                for channel in extracted_paths.keys():
                    if channel in chdict.keys() and path_per_ch[channel]:
                        path_per_ch[channel] += chdict[channel] + subsplit[1]
                    elif len(chdict.keys())>0: # don't take channels into account if not provided by user unless there is no channel dependence at all
                        path_per_ch[channel] = None
        for channel in extracted_paths.keys():
            if path_per_ch[channel]:
                extracted_paths[channel].append(path_per_ch[channel])
    return extracted_paths

def main():
    parser = argparse.ArgumentParser(description='Script to manage condor batch system jobs for the executables and their outputs.')
    parser.add_argument('--executable',required=True, choices=['SVFit', 'MELA', 'NNScore', 'NNMass', 'NNrecoil', 'FakeFactors', 'ZPtMReweighting'], help='Executable to be used for friend tree creation ob the batch system.')
    parser.add_argument('--batch_cluster',required=True, choices=['naf','etp6','etp7','lxplus6','lxplus7'], help='Batch system cluster to be used.')
    parser.add_argument('--command',required=True, choices=['submit','collect','check'], help='Command to be done by the job manager.')
    parser.add_argument('--input_ntuples_directory',required=True, help='Directory where the input files can be found. The file structure in the directory should match */*.root wildcard.')
    parser.add_argument('--friend_ntuples_directories', nargs='+', default=[], help='Directory where the friend files can be found. The file structure in the directory should match the one of the base ntuples. Channel dependent parts of the path can be inserted like /commonpath/{et:et_folder,mt:mt_folder,tt:tt_folder}/commonpath.')
    parser.add_argument('--events_per_job',required=True, type=int, help='Event to be processed by each job')
    parser.add_argument('--walltime',default=-1, type=int, help='Walltime to be set for the job (in seconds). If negative, then it will not be set. [Default: %(default)s]')
    parser.add_argument('--cores',default=5, type=int, help='Number of cores to be used for the collect command. [Default: %(default)s]')
    parser.add_argument('--max_jobs_per_batch',default=10000, type=int, help='Maximal number of job per batch. [Default: %(default)s]')
    parser.add_argument('--extended_file_access',default=None, type=str, help='Additional prefix for the file access, e.g. via xrootd.')
    parser.add_argument('--custom_workdir_path',default=None, type=str, help='Absolute path to a workdir directory different from $CMSSW_BASE/src.')
    parser.add_argument('--restrict_to_channels', nargs='+', default=[], help='Produce friends only for certain channels')
    parser.add_argument('--restrict_to_shifts', nargs='+', default=[], help='Produce friends only for certain shifts')
    parser.add_argument('--restrict_to_samples_wildcards', nargs='+', default=[], help='Produce friends only for samples matching the path wildcard')

    args = parser.parse_args()
    if len(args.restrict_to_samples_wildcards) == 0:
        args.restrict_to_samples_wildcards.append("*")
    input_ntuples_list = []
    for wildcard in args.restrict_to_samples_wildcards:
        input_ntuples_list += glob.glob(os.path.join(args.input_ntuples_directory,wildcard,"*.root"))
    extracted_friend_paths = extract_friend_paths(args.friend_ntuples_directories)
    if args.extended_file_access:
        input_ntuples_list = ["/".join([args.extended_file_access,f]) for f in input_ntuples_list]
    if args.command == "submit":
        prepare_jobs(input_ntuples_list, args.input_ntuples_directory, extracted_friend_paths, args.events_per_job, args.batch_cluster, args.executable, args.walltime, args.max_jobs_per_batch, args.custom_workdir_path, args.restrict_to_channels, args.restrict_to_shifts)
    elif args.command == "collect":
        collect_outputs(args.executable, args.cores, args.custom_workdir_path)
    elif args.command == "check":
        check_and_resubmit(args.executable, args.custom_workdir_path)
if __name__ == "__main__":
    main()
