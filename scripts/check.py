import itertools
from multiprocessing import Pool
import re
import os, json, logging

logger = logging.getLogger("job_managment")
from streampaths import *
import ROOT as r

r.PyConfig.IgnoreCommandLineOptions = True
r.gROOT.ProcessLine("gErrorIgnoreLevel = 2001;")


def check_output_files(f, mode, t, n):
    valid_file = True
    if os.environ["USER"] == "mscham":
        print("Checking: ", f)
    if mode == "local":
        if not os.path.exists(f):
            valid_file = False
            print("File not there:", f)
        else:
            F = r.TFile.Open(f, "read")
            if F:
                valid_file = not F.IsZombie() and not F.TestBit(r.TFile.kRecovered)
            else:
                valid_file = False
            if not valid_file:
                F.Close()
                print("File is corrupt: ", f)
                os.remove(f)
            else:
                input_tree = F.Get(str(t))
                if not input_tree:
                    print("Tree not found:", f, t)
                    valid_file = False
                else:
                    n_tr = input_tree.GetEntries()
                    if n_tr != n:
                        valid_file = False
                        print("WRONG number of events:", f, t, n_tr, "!=", n)
                F.Close()
    elif mode == "xrootd":
        myclient = client.FileSystem(server_xrootd["GridKA"])
        status, info = myclient.stat(f)
        if not info:
            print("File not there:", f)
            valid_file = False
    return valid_file


def check_output_files_wrap(args):
    return check_output_files(*args)


def check_and_resubmit(executable, custom_workdir_path, mode, check_all, cores):
    if custom_workdir_path:
        workdir_path = os.path.join(custom_workdir_path, executable.replace('.py', '') + "_workdir")
    else:
        workdir_path = os.path.join(
            os.environ["CMSSW_BASE"], "src", executable.replace('.py', '') + "_workdir"
        )

    # Read job-database (same for HTCondor and GC)
    jobdb_path = os.path.join(workdir_path, "condor_" + executable.replace('.py', '') + ".json")
    jobdb_file = open(jobdb_path, "r")
    jobdb = json.loads(jobdb_file.read())

    # Check which of the incomplete files are there and can be read
    job_to_resubmit = set()
    if cores >= 1:
        pool = Pool(cores)
        shared_filepath = []
        shared_tree_name = []
        shared_n = []
        shared_job_to_resubmit = []
        shared_jobnumber = []
        shared_subjobnumber = []
    for jobnumber in sorted([int(k) for k in jobdb.keys()]):
        for subjobnumber in range(len(jobdb[str(jobnumber)])):

            nick = (
                jobdb[str(jobnumber)][subjobnumber]["input"]
                .split("/")[-1]
                .replace(".root", "")
            )
            pipeline = jobdb[str(jobnumber)][subjobnumber]["folder"]
            tree = jobdb[str(jobnumber)][subjobnumber]["tree"]
            first = jobdb[str(jobnumber)][subjobnumber]["first_entry"]
            last = jobdb[str(jobnumber)][subjobnumber]["last_entry"]
            status = jobdb[str(jobnumber)][subjobnumber]["status"]
            n = last - first + 1

            # Get single file path
            filename = "_".join([nick, pipeline, str(first), str(last)]) + ".root"
            if mode == "local":
                filepath = os.path.join(workdir_path, nick, filename)
            elif mode == "xrootd":
                gc_path = os.path.join(
                    workdir_path, "grid_control_{}.conf".format(executable.replace('.py', ''))
                )
                with open(gc_path, "r") as gc_file:
                    for line in gc_file.readlines():
                        if "se path" in line:
                            filepath = (
                                "/store/"
                                + line.split("/store/")[1].strip("\n")
                                + "/"
                                + filename
                            )
                            break

            if cores > 1:
                shared_filepath.append(filepath)
                shared_tree_name.append("/".join([pipeline, tree]))
                shared_n.append(n)
                shared_jobnumber.append(jobnumber)
                shared_subjobnumber.append(subjobnumber)
            else:
                # Check the file if incomplete
                if status != "complete" or check_all:
                    if not check_output_files(
                        filepath, mode, "/".join([pipeline, tree]), n
                    ):
                        job_to_resubmit.add(jobnumber)
                    else:
                        jobdb[str(jobnumber)][subjobnumber]["status"] = "complete"
    if cores > 1:
        logger.debug("starting pool.map")

        # print(zip(shared_filepath, [mode] * len(shared_filepath), shared_tree_name, shared_n)[0])
        x = pool.map(
            check_output_files_wrap,
            itertools.izip(
                shared_filepath, itertools.repeat(mode), shared_tree_name, shared_n
            ),
        )
        resubmit_jobid = []
        for i, xi in enumerate(x):
            if not xi:
                # if not check_output_files(filepath, mode, '/'.join([pipeline, tree]), n):
                job_to_resubmit.add(shared_jobnumber[i])
                resubmit_jobid.append(shared_jobnumber[i])
                print("resubmit:", shared_jobnumber[i], shared_subjobnumber[
                    i
                ], shared_filepath[i])
            else:
                jobdb[str(shared_jobnumber[i])][shared_subjobnumber[i]][
                    "status"
                ] = "complete"

    # Save list of jobs to resubmit
    arguments_path = os.path.join(workdir_path, "arguments_resubmit.txt")
    with open(arguments_path, "w") as arguments_file:
        arguments_file.write("\n".join([str(arg) for arg in sorted(job_to_resubmit)]))
        arguments_file.close()

    # prepare resubmit logging path
    if not os.path.exists(os.path.join(workdir_path, "logging", "remaining")):
        os.makedirs(os.path.join(workdir_path, "logging", "remaining"))

    # Save resubmittion script
    condor_jdl_path = os.path.join(workdir_path, "condor_" + executable.replace('.py', '') + "_0.jdl")
    with open(condor_jdl_path, "r") as file:
        condor_jdl_resubmit = file.read()
    condor_jdl_resubmit_file = "condor_" + executable.replace('.py', '') + "_resubmit.jdl"
    condor_jdl_resubmit_path = os.path.join(workdir_path, condor_jdl_resubmit_file)
    condor_jdl_resubmit = re.sub(
        "\_0.txt", "_resubmit.txt", condor_jdl_resubmit
    ).replace("/0/", "/remaining/")
    with open(condor_jdl_resubmit_path, "w") as file:
        file.write(condor_jdl_resubmit)
        file.close

    # Save updated job-database
    with open(jobdb_path, "w") as db:
        db.write(json.dumps(jobdb, sort_keys=True, indent=2))
        db.close()

    # Final screen message
    if len(job_to_resubmit) > 0:
        print("")
        print("To run the resubmission, check {} first".format(condor_jdl_resubmit_path))
        print("Command:")
        print("condor_submit {CONDORJDL}".format(
            TASKDIR=os.path.abspath(workdir_path), CONDORJDL=os.path.abspath(condor_jdl_resubmit_file)
        ))
        print("")
    else:
        print("\nNothing to resubmit.\n")
