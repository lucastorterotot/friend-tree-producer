#!/usr/bin/env python

import ROOT as r
import glob
import argparse
import json
import os
import numpy as np
import stat
from multiprocessing import Pool

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
        outputfile.mkdir(p)
        outputfile.cd(p)
        tree = db[nick][p].CopyTree("")
        tree.Write("",r.TObject.kOverwrite)
    outputfile.Close()

def prepare_jobs(input_ntuples_list, events_per_job, batch_cluster, executable, walltime):
    ntuple_database = {}
    for f in input_ntuples_list:
        nick = os.path.basename(f).strip(".root")
        ntuple_database[nick] = {}
        ntuple_database[nick]["path"] = f
        F = r.TFile.Open(f,"read")
        pipelines = [k.GetName() for k in F.GetListOfKeys()]
        ntuple_database[nick]["pipelines"] = {}
        for p in pipelines:
            ntuple_database[nick]["pipelines"][p] = F.Get(p).Get("ntuple").GetEntries()
    job_database = {}
    job_number = 0
    for nick in ntuple_database:
        for p in ntuple_database[nick]["pipelines"]:
            n_entries = ntuple_database[nick]["pipelines"][p]
            entry_list = np.append(np.arange(0,n_entries,events_per_job),[n_entries -1])
            first_entries = entry_list[:-1]
            last_entries = entry_list[1:] -1
            last_entries[-1] += 1
            for first,last in zip(first_entries, last_entries):
                job_database[job_number] = {}
                job_database[job_number]["input"] = ntuple_database[nick]["path"]
                job_database[job_number]["folder"] = p
                job_database[job_number]["tree"] = "ntuple"
                job_database[job_number]["first_entry"] = first
                job_database[job_number]["last_entry"] = last
                job_number +=1
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
    condorjdl_path = os.path.join(workdir_path,"condor_"+executable+".jdl")
    jobdb_path = os.path.join(workdir_path,"condor_"+executable+".json")
    datasetdb_path = os.path.join(workdir_path,"dataset.json")
    with open(executable_path,"w") as shellscript:
        shellscript.write(shellscript_content)
        os.chmod(executable_path, os.stat(executable_path).st_mode | stat.S_IEXEC)
        shellscript.close()
    condorjdl_template_path = os.path.join(os.environ["CMSSW_BASE"],"src/HiggsAnalysis/friend-tree-producer/data/submit_condor_%s.jdl"%batch_cluster)
    condorjdl_template_file = open(condorjdl_template_path,"r")
    condorjdl_template = condorjdl_template_file.read()
    njobs = str(job_number)
    if batch_cluster == "etp":
        if walltime > 0:
            condorjdl_content = condorjdl_template.format(TASKDIR=workdir_path,EXECUTABLE=executable_path,NJOBS=njobs,WALLTIME=str(walltime))
        else:
            print "Warning: walltime for % cluster not set. Setting it to 1h."%batch_cluster
            condorjdl_content = condorjdl_template.format(TASKDIR=workdir_path,EXECUTABLE=executable_path,NJOBS=njobs,WALLTIME=str(3600))
    elif batch_cluster == "lxplus":
        if walltime > 0:
            condorjdl_content = condorjdl_template.format(TASKDIR=workdir_path,EXECUTABLE=executable_path,NJOBS=njobs,WALLTIME=str(walltime))
        else:
            print "Warning: walltime for % cluster not set. Setting it to 1h."%batch_cluster
            condorjdl_content = condorjdl_template.format(TASKDIR=workdir_path,EXECUTABLE=executable_path,NJOBS=njobs,WALLTIME=str(3600))
    else:
        condorjdl_content = condorjdl_template.format(TASKDIR=workdir_path,EXECUTABLE=executable_path,NJOBS=njobs)
    with open(condorjdl_path,"w") as condorjdl:
        condorjdl.write(condorjdl_content)
        condorjdl.close()
    print "To run the condor submission, execute the following:"
    print
    print "cd {TASKDIR}; condor_submit {CONDORJDL}".format(TASKDIR=workdir_path, CONDORJDL=condorjdl_path)
    with open(jobdb_path,"w") as db:
        db.write(json.dumps(job_database, sort_keys=True, indent=2))
        db.close()
    with open(datasetdb_path,"w") as datasets:
        datasets.write(json.dumps(ntuple_database, sort_keys=True, indent=2))
        datasets.close()

def collect_outputs(executable,cores):
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
        nick = os.path.basename(jobdb[str(jobnumber)]["input"]).strip(".root")
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

def main():
    parser = argparse.ArgumentParser(description='Script to manage condor batch system jobs for the executables and their outputs.')
    parser.add_argument('--executable',required=True, choices=['SVFit', 'MELA'], help='Executable to be used for friend tree creation ob the batch system.')
    parser.add_argument('--batch_cluster',required=True, choices=['naf','etp', 'lxplus'], help='Batch system cluster to be used.')
    parser.add_argument('--command',required=True, choices=['submit','collect'], help='Command to be done by the job manager.')
    parser.add_argument('--input_ntuples_directory',required=True, help='Directory where the input files can be found. The file structure in the directory should match */*.root wildcard.')
    parser.add_argument('--events_per_job',required=True, type=int, help='Event to be processed by each job')
    parser.add_argument('--walltime',default=-1, type=int, help='Walltime to be set for the job (in seconds). If negative, then it will not be set. [Default: %(default)s]')
    parser.add_argument('--cores',default=5, type=int, help='Number of cores to be used for the collect command. [Default: %(default)s]')
    args = parser.parse_args()

    input_ntuples_list = glob.glob(os.path.join(args.input_ntuples_directory,"*","*.root"))
    if args.command == "submit":
        prepare_jobs(input_ntuples_list, args.events_per_job, args.batch_cluster, args.executable, args.walltime)
    elif args.command == "collect":
        collect_outputs(args.executable, args.cores)

if __name__ == "__main__":
    main()
