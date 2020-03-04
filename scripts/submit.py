from multiprocessing import Pool, Manager
import stat
import copy
import numpy as np
import time
import os, json, logging

logger = logging.getLogger("job_managment")
from streampaths import *
import uproot

from threading import Lock
s_print_lock = Lock()

def extract_friend_paths(packed_paths):
    extracted_paths = {"em": [], "et": [], "mt": [], "tt": []}
    for pathname in packed_paths:
        splitpath = pathname.split("{")
        et_path, mt_path, tt_path = splitpath[0], splitpath[0], splitpath[0]
        path_per_ch = {}
        for channel in extracted_paths.keys():
            path_per_ch[channel] = splitpath[0]
        first = True
        for split in splitpath:
            if first:
                first = False
            else:
                subsplit = split.split("}")
                chdict = json.loads(
                    '{"' + subsplit[0].replace(":", '":"').replace(",", '", "') + '"}'
                )
                for channel in extracted_paths.keys():
                    if channel in chdict.keys() and path_per_ch[channel]:
                        path_per_ch[channel] += chdict[channel] + subsplit[1]
                    elif (
                        len(chdict.keys()) > 0
                    ):  # don't take channels into account if not provided by user unless there is no channel dependence at all
                        path_per_ch[channel] = None
        for channel in extracted_paths.keys():
            if path_per_ch[channel]:
                extracted_paths[channel].append(path_per_ch[channel])
    return extracted_paths


def get_entries(*args):
    f = args[0]["f"]
    restrict_to_channels = args[0]["restrict_to_channels"]
    restrict_to_shifts = args[0]["restrict_to_shifts"]
    Global = args[0]["Global"]

    restrict_to_channels_file = copy.deepcopy(restrict_to_channels)
    nick = f.split("/")[-1].replace(".root", "")

    warnings = []
    if "SingleMuon_Run" in nick or "MuTauFinalState" in nick:
        restrict_to_channels_file = (
            list(set(["mt", "mm"]).intersection(restrict_to_channels_file))
            if len(restrict_to_channels_file) > 0
            else ["mt", "mm"]
        )
        warnings.append(
            "\tWarning: restrict %s to '%s' channel(s)"
            % (nick, restrict_to_channels_file)
        )

    elif (
        "SingleElectron_Run" in nick
        or "EGamma_Run" in nick
        or "ElTauFinalState" in nick
    ):
        restrict_to_channels_file = (
            list(set(["et"]).intersection(restrict_to_channels_file))
            if len(restrict_to_channels_file) > 0
            else ["et"]
        )
        warnings.append(
            "\tWarning: restrict %s to '%s' channel(s)"
            % (nick, restrict_to_channels_file)
        )

    elif "Tau_Run" in nick or "TauTauFinalState" in nick:
        restrict_to_channels_file = (
            list(set(["tt"]).intersection(restrict_to_channels_file))
            if len(restrict_to_channels_file) > 0
            else ["tt"]
        )
        warnings.append(
            "\tWarning: restrict %s to '%s' channel(s)"
            % (nick, restrict_to_channels_file)
        )

    elif "MuonEG_Run" in nick or "ElMuFinalState" in nick:
        restrict_to_channels_file = (
            list(set(["em"]).intersection(restrict_to_channels_file))
            if len(restrict_to_channels_file) > 0
            else ["em"]
        )
        warnings.append(
            "\tWarning: restrict %s to '%s' channel(s)"
            % (nick, restrict_to_channels_file)
        )

    F = uproot.open(f)
    pipelines = [x.strip(";1") for x in F.keys() ]
    if len(restrict_to_channels_file) > 0 or len(restrict_to_channels) > 0:
        pipelines = [
            p for p in pipelines if p.split("_")[0] in restrict_to_channels_file
        ]
    if len(restrict_to_shifts) > 0:
        pipelines = [p for p in pipelines if p.split("_")[1] in restrict_to_shifts]
    pipelieness = {}
    for p in pipelines:
        try:
            pipelieness[p] = F[p+"/ntuple"].numentries
        except:
            import sys

            with s_print_lock:
                print "Unexpected error:", sys.exc_info()[0]
                logger.critical("problem in file: %s pipeline: %s" % (f, p))
            raise
    del F
    with s_print_lock:
        Global.counter += 1
        logger.info(
            "done: %s [%d/%d]" % (nick, Global.counter, Global.ninput_ntuples_list)
        )
        logger.debug("\t pipelines: \n\t\t%s" % "\n\t\t".join(pipelines))
        for w in warnings:
            logger.warning(w)
    return [nick, {"path": f, "pipelines": pipelieness,}]


def prepare_jobs(
    input_ntuples_list,
    inputs_base_folder,
    inputs_friends_folders,
    events_per_job,
    batch_cluster,
    executable,
    walltime,
    max_jobs_per_batch,
    custom_workdir_path,
    restrict_to_channels,
    restrict_to_shifts,
    mode,
    output_server_srm,
    cores,
    dry,
    se_path=None,
    shadow_input_ntuples_directory=None,
    input_server=None,
    conditional=False,
    extra_parameters="",
):
    logger.debug("starting prepare_jobs")
    cmsswbase = os.environ["CMSSW_BASE"]
    inputs_friends_folders = extract_friend_paths(inputs_friends_folders)
    manager = Manager()
    ntuple_database = manager.dict()
    Global = manager.Namespace()
    Global.counter = 0
    Global.ninput_ntuples_list = len(input_ntuples_list)
    toDo = map(
        lambda e: {
            "f": e,
            "restrict_to_channels": restrict_to_channels,
            "restrict_to_shifts": restrict_to_shifts,
            "Global": Global,
        },
        input_ntuples_list,
    )
    if cores > 1:
        pool = Pool(cores)

        logger.debug("starting pool.map")
        res = pool.map(get_entries, toDo)
    else:
        res = []
        for e in toDo:
            res.append(get_entries(e))
    for r in res:
        ntuple_database[r[0]] = r[1]

    job_database = {0: []}
    job_number = 0
    ### var for check if last file was completly processed
    lastTouchedEntry = -99
    ####variable for checking if the are still some event of this pipeline that need distributing
    curNofEventsInJob = 0
    ## iterate over the files
    logger.debug("starting wd creation")
    sorted_nicks = ntuple_database.keys()
    sorted_nicks.sort()
    for nick in sorted_nicks:
        sorted_pipelines = ntuple_database[nick]["pipelines"].keys()
        sorted_pipelines.sort()
        for p in sorted_pipelines:
            n_entries = ntuple_database[nick]["pipelines"][p]
            if n_entries > 0:
                ####variable for checking if there are still some event of this pipeline that need distributing
                logger.debug(
                    str(
                        "Processing:"
                        + nick.split("_")[1]
                        + "with "
                        + str(n_entries)
                        + " events"
                    )
                )
                entriesRemainingCurPipe = n_entries
                while entriesRemainingCurPipe > 0:
                    ### check if we need to move on the the next job
                    if curNofEventsInJob == events_per_job:
                        curNofEventsInJob = 0
                        job_number += 1
                        job_database[job_number] = []
                    else:
                        pass
                    job_database[job_number].append({})

                    if lastTouchedEntry == -99:
                        first = 0
                    else:
                        first = lastTouchedEntry + 1
                    ## check if the events of this pipeline can be packed in this job
                    if entriesRemainingCurPipe <= events_per_job - curNofEventsInJob:
                        last = first + entriesRemainingCurPipe - 1
                        lastTouchedEntry = -99
                    else:
                        last = first + (events_per_job - curNofEventsInJob) - 1
                        lastTouchedEntry = last
                    curNofEventsInJob += last - first + 1
                    entriesRemainingCurPipe = entriesRemainingCurPipe - (
                        last - first + 1
                    )
                    job_database[job_number][-1]["input"] = ntuple_database[nick][
                        "path"
                    ]
                    if "FakeFactor" in executable:
                        job_database[job_number][-1]["cmsswbase"] = cmsswbase
                    job_database[job_number][-1]["folder"] = p
                    job_database[job_number][-1]["tree"] = "ntuple"
                    job_database[job_number][-1]["first_entry"] = first
                    job_database[job_number][-1]["last_entry"] = last
                    job_database[job_number][-1]["status"] = "submitted"
                    if "NNScore" in executable:
                        job_database[job_number][-1]["conditional"] = int(conditional)
                    channel = p.split("_")[0]
                    if (
                        channel in inputs_friends_folders.keys()
                        and len(inputs_friends_folders[channel]) > 0
                    ):
                        job_database[job_number][-1]["input_friends"] = " ".join(
                            [
                                job_database[job_number][-1]["input"].replace(
                                    inputs_base_folder, friend_folder
                                )
                                for friend_folder in inputs_friends_folders[channel]
                            ]
                        )
                    logger.debug(
                        str(
                            {
                                "nick": nick,
                                "job_number": job_number,
                                "l/f": (first, last),
                                "lastTouchedEntry": lastTouchedEntry,
                                "curNofEventsInJob": curNofEventsInJob,
                                "entriesRemCurPipe": entriesRemainingCurPipe,
                            }
                        )
                    )
            else:
                print "Warning: %s has no entries in pipeline %s" % (nick, p)
    logger.debug(str("Done creating the job_database"))
    for j in job_database.keys():
        logger.debug("Job {}:".format(j))
        for sj in job_database[j]:
            logger.debug(str(sj))
    job_database_copy = copy.deepcopy(job_database)
    #### add 1 to the job number, to fix the last job not being exexuted:
    job_number += 1

    if dry:
        return
    if custom_workdir_path:
        if not os.path.exists(custom_workdir_path):
            os.makedirs(custom_workdir_path)
        workdir_path = os.path.join(custom_workdir_path, executable.replace('.py', '') + "_workdir")
    else:
        workdir_path = os.path.join(
            os.environ["CMSSW_BASE"], "src", executable.replace('.py', '') + "_workdir"
        )
    if not os.path.exists(workdir_path):
        os.makedirs(workdir_path)
    if not os.path.exists(os.path.join(workdir_path, "logging")):
        os.makedirs(os.path.join(workdir_path, "logging"))

    command_template = """
    if [ $1 -eq {JOBNUMBER} ]; then
        {COMMAND}
    fi
    """

    commandlist = []
    for jobnumber in job_database.keys():
        for subjobnumber in range(len(job_database[jobnumber])):
            options = " ".join(
                [
                    "--" + k + " " + str(v)
                    for (k, v) in job_database[jobnumber][subjobnumber].items()
                    if k != "status"
                ]
                + [extra_parameters]
            )
            if mode == "xrootd":
                options += " --organize_outputs=false"
            commandline = "{EXEC} {OPTIONS}".format(EXEC=executable, OPTIONS=options)
            command = command_template.format(
                JOBNUMBER=str(jobnumber), COMMAND=commandline
            )
            commandlist.append(command)
    commands = "\n".join(commandlist)

    shellscript_template = """#!/bin/sh
        ulimit -s unlimited
        set -e

        cd {TASKDIR}

        {COMMANDS}

        echo 0
    """
    shellscript_content = shellscript_template.format(
        COMMANDS=commands, TASKDIR=os.path.abspath(workdir_path)
    )
    executable_path = os.path.join(workdir_path, "condor_" + executable.replace('.py', '') + ".sh")
    gc_executable_path = os.path.join(
        workdir_path, "condor_" + executable.replace('.py', '') + "_forGC.sh"
    )
    jobdb_path = os.path.join(workdir_path, "condor_" + executable.replace('.py', '') + ".json")
    datasetdb_path = os.path.join(workdir_path, "dataset.json")
    with open(executable_path, "w") as shellscript:
        shellscript.write(shellscript_content)
        os.chmod(executable_path, os.stat(executable_path).st_mode | stat.S_IEXEC)
        shellscript.close()
    with open(gc_executable_path, "w") as shellscript:
        shellscript.write(shellscript_content.replace("$1", "$GC_JOB_ID"))
        os.chmod(executable_path, os.stat(executable_path).st_mode | stat.S_IEXEC)
        shellscript.close()
    if mode == "xrootd":
        global gc_date_tag
        gc_date_tag = "{}_{}".format(executable.replace('.py', ''), time.strftime("%Y-%m-%d_%H-%M-%S"))

        os.makedirs(os.path.join(os.environ["CMSSW_BASE"], "src", gc_date_tag))
        with open(
            os.path.join(
                os.environ["CMSSW_BASE"],
                "src",
                gc_date_tag,
                "condor_{}_forGC.sh".format(executable.replace('.py', '')),
            ),
            "w",
        ) as shellscript:
            shellscript.write(
                shellscript_content.replace("$1", "$GC_JOB_ID").replace(
                    "cd {TASKDIR}\n".format(TASKDIR=os.path.abspath(workdir_path)), ""
                )
            )
            os.chmod(executable_path, os.stat(executable_path).st_mode | stat.S_IEXEC)
            shellscript.close()
        gc_executable_path = "$CMSSW_BASE/src/{}/condor_{}_forGC.sh".format(
            gc_date_tag, executable.replace('.py', '')
        )
    condorjdl_template_path = os.path.join(
        os.environ["CMSSW_BASE"],
        "src/HiggsAnalysis/friend-tree-producer/data/submit_condor_%s.jdl"
        % batch_cluster,
    )
    condorjdl_template_file = open(condorjdl_template_path, "r")
    condorjdl_template = condorjdl_template_file.read()
    if mode == "local":
        gc_storage_dir = workdir_path
        extra_se_info = ""
    elif mode == "xrootd":
        gc_storage_dir = (
            server_srm[output_server_srm]
            + "store/user/{}/gc_storage/{}".format(os.environ["USER"], gc_date_tag)
            if se_path is None
            else se_path
        )
        extra_se_info = "se output files = *.root\nse output pattern = @XBASE@.@XEXT@"
    gc_template_path = os.path.join(
        os.environ["CMSSW_BASE"],
        "src/HiggsAnalysis/friend-tree-producer/data/grid-control_%s.conf"
        % batch_cluster,
    )
    gc_template_file = open(gc_template_path, "r")
    gc_template = gc_template_file.read()
    if walltime > 0:
        if walltime < 86399:
            gc_content = gc_template.format(
                STORAGE_DIR=gc_storage_dir,
                EXTRA_SE_INFO=extra_se_info,
                TASKDIR=workdir_path,
                EXECUTABLE=gc_executable_path,
                WALLTIME=time.strftime("%H:%M:%S", time.gmtime(walltime)),
                NJOBS=job_number,
            )
        else:
            print "Warning: Please set walltimes greater than 24 hours manually in gc config."
            gc_content = gc_template.format(
                STORAGE_DIR=gc_storage_dir,
                EXTRA_SE_INFO=extra_se_info,
                TASKDIR=workdir_path,
                EXECUTABLE=gc_executable_path,
                WALLTIME="24:00:00",
                NJOBS=job_number,
            )
    else:
        print "Warning: walltime for %s cluster not set. Setting it to 1h." % batch_cluster
        gc_content = gc_template.format(
            STORAGE_DIR=gc_storage_dir,
            EXTRA_SE_INFO=extra_se_info,
            TASKDIR=workdir_path,
            EXECUTABLE=gc_executable_path,
            WALLTIME="1:00:00",
            NJOBS=job_number,
        )
    if mode == "xrootd":
        gc_content = gc_content.replace("&&(TARGET.ProvidesEKPResources==True)", "")
    gc_path = os.path.join(workdir_path, "grid_control_{}.conf".format(executable.replace('.py', '')))
    with open(gc_path, "w") as gc:
        gc.write(gc_content)
        gc.close()

    argument_borders = np.append(
        np.arange(0, job_number, max_jobs_per_batch), [job_number]
    )
    first_borders = argument_borders[:-1]
    last_borders = argument_borders[1:] - 1
    printout_list = []
    for index, (first, last) in enumerate(zip(first_borders, last_borders)):
        condorjdl_path = os.path.join(
            workdir_path, "condor_" + executable.replace('.py', '') + "_%d.jdl" % index
        )
        argument_list = np.arange(first, last + 1)
        if not os.path.exists(os.path.join(workdir_path, "logging", str(index))):
            os.makedirs(os.path.join(workdir_path, "logging", str(index)))
        arguments_path = os.path.join(workdir_path, "arguments_%d.txt" % (index))
        with open(arguments_path, "w") as arguments_file:
            arguments_file.write("\n".join([str(arg) for arg in argument_list]))
            arguments_file.close()
        njobs = "arguments from %s/arguments_%d.txt" % (os.path.abspath(workdir_path), index)
        if batch_cluster in ["etp6", "etp7", "lxplus6", "lxplus7", "rwth"]:
            if walltime > 0:
                condorjdl_content = condorjdl_template.format(
                    TASKDIR=workdir_path,
                    TASKNUMBER=str(index),
                    EXECUTABLE=executable_path,
                    NJOBS=njobs,
                    WALLTIME=str(walltime),
                )
            else:
                print "Warning: walltime for %s cluster not set. Setting it to 1h." % batch_cluster
                condorjdl_content = condorjdl_template.format(
                    TASKDIR=workdir_path,
                    TASKNUMBER=str(index),
                    EXECUTABLE=executable_path,
                    NJOBS=njobs,
                    WALLTIME=str(3600),
                )
        else:
            condorjdl_content = condorjdl_template.format(
                TASKDIR=os.path.abspath(workdir_path),
                TASKNUMBER=str(index),
                EXECUTABLE=os.path.abspath(executable_path),
                NJOBS=njobs,
            )
        with open(condorjdl_path, "w") as condorjdl:
            condorjdl.write(condorjdl_content)
            condorjdl.close()
        printout_list.append(
            "condor_submit {CONDORJDL}".format(
                TASKDIR=os.path.abspath(workdir_path), CONDORJDL=os.path.abspath(condorjdl_path)
            )
        )
    print
    print "To run the condor submission, execute the following:"
    print
    print "\n".join(printout_list)
    print
    print "Or with grid-control:"
    print "go.py {} -Gc".format(gc_path)
    print
    with open(jobdb_path, "w") as db:
        db.write(json.dumps(job_database_copy, sort_keys=True, indent=2))
        db.close()
    with open(datasetdb_path, "w") as datasets:
        datasets.write(json.dumps(ntuple_database.copy(), sort_keys=True, indent=2))
        datasets.close()
