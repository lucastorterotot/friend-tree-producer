#!/usr/bin/env python
import os, glob
import fnmatch

from XRootD import client
from XRootD.client.flags import DirListFlags, OpenFlags, MkDirFlags, QueryCode
import logging

logger = logging.getLogger("job_managment")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

import parse_args
import submit
import check
import collect


gc_date_tag = None

server_xrootd = {
    "GridKA": "root://cmsxrootd-kit.gridka.de:1094/",
    "DESY": "root://cmsxrootd-kit.gridka.de:1094/",
    "RWTH": "root://grid-se004.physik.rwth-aachen.de:1094/",
    "EOS": "root://eosuser.cern.ch:1094/",
}

server_srm = {
    "GridKA": "srm://cmssrm-kit.gridka.de:8443/srm/managerv2?SFN=/pnfs/gridka.de/cms/disk-only/",
    "DESY": "srm://dcache-se-cms.desy.de:8443/srm/managerv2?SFN=/pnfs/desy.de/cms/tier2/",
    "RWTH": "srm://grid-srm.physik.rwth-aachen.de:8443/srm/managerv2?SFN=/pnfs/physik.rwth-aachen.de/cms/",
}


def doReplace(file_name, old, new):

    with open(file_name, "r") as f:
        data = f.read()

    data = data.replace(old, new)

    with open(file_name, "w") as f:
        f.write(data)


def main():
    parser = parse_args.setup_parser()
    args = parser.parse_args()
    input_ntuples_list = []
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # shadow the remote with local when creating jobs
    if args.shadow_input_ntuples_directory is not None:
        a = args.input_ntuples_directory
        args.input_ntuples_directory = args.shadow_input_ntuples_directory
        args.shadow_input_ntuples_directory = a

    # TODO: Expand functionalities for remote access
    if args.mode == "xrootd":

        if args.executable != "SVFit":
            print "Remote access currently only works for SVFit computation."
            exit()

        if args.input_server not in ["GridKA", "RWTH"]:
            print "Remote access currently only works for input files on Gridka and RWTH."
            exit()

    if args.command == "submit":

        if args.mode == "local" or args.shadow_input_ntuples_directory is not None:
            if len(args.restrict_to_samples_wildcards) == 0:
                args.restrict_to_samples_wildcards.append("*")
            for wildcard in args.restrict_to_samples_wildcards:
                input_ntuples_list += glob.glob(
                    os.path.join(args.input_ntuples_directory, wildcard, "*.root")
                )
            if args.extended_file_access:
                input_ntuples_list = [
                    "/".join([args.extended_file_access, f]) for f in input_ntuples_list
                ]

        elif args.mode == "xrootd":
            myclient = client.FileSystem(server_xrootd[args.input_server])
            status, listing = myclient.dirlist(
                args.input_ntuples_directory, DirListFlags.STAT
            )
            dataset_dirs = [entry.name for entry in listing]
            all_files = []

            for sd in dataset_dirs:
                dataset_dir = os.path.join(args.input_ntuples_directory, sd)
                s, dataset_listing = myclient.dirlist(dataset_dir, DirListFlags.STAT)
                if dataset_listing is not None:
                    all_files += [
                        "root://"
                        + f.hostaddr
                        + "/"
                        + os.path.join(dataset_listing.parent, f.name)
                        for f in dataset_listing
                    ]

            if len(args.restrict_to_samples_wildcards) == 0:
                args.restrict_to_samples_wildcards.append("*")
            for wildcard in args.restrict_to_samples_wildcards:
                input_ntuples_list += fnmatch.filter(all_files, wildcard)

        logger.debug(input_ntuples_list)

        submit.prepare_jobs(
            input_ntuples_list,
            args.input_ntuples_directory,
            args.friend_ntuples_directories,
            args.events_per_job,
            args.batch_cluster,
            args.executable,
            args.walltime,
            args.max_jobs_per_batch,
            args.custom_workdir_path,
            args.restrict_to_channels,
            args.restrict_to_shifts,
            args.mode,
            args.output_server_xrootd,
            args.cores,
            args.dry,
            args.se_path,
            args.shadow_input_ntuples_directory,
            args.input_server,
            conditional=args.conditional,
            extra_parameters=args.extra_parameters,
        )

        if args.shadow_input_ntuples_directory:
            if args.dry:
                return
            if args.custom_workdir_path:
                workdir_path = os.path.join(
                    args.custom_workdir_path, args.executable + "_workdir"
                )
            else:
                workdir_path = os.path.join(
                    os.environ["CMSSW_BASE"], "src", args.executable + "_workdir"
                )

            datasetdb_path = os.path.join(workdir_path, "dataset.json")
            executable_path = os.path.join(
                workdir_path, "condor_" + args.executable + ".sh"
            )
            gc_executable_path = os.path.join(
                workdir_path, "condor_" + args.executable + "_forGC.sh"
            )
            jobdb_path = os.path.join(
                workdir_path, "condor_" + args.executable + ".json"
            )

            doReplace(
                datasetdb_path,
                old=args.input_ntuples_directory,
                new=server_xrootd[args.input_server]
                + args.shadow_input_ntuples_directory,
            )
            doReplace(
                executable_path,
                old=args.input_ntuples_directory,
                new=server_xrootd[args.input_server]
                + args.shadow_input_ntuples_directory,
            )
            doReplace(
                gc_executable_path,
                old=args.input_ntuples_directory,
                new=server_xrootd[args.input_server]
                + args.shadow_input_ntuples_directory,
            )
            doReplace(
                jobdb_path,
                old=args.input_ntuples_directory,
                new=server_xrootd[args.input_server]
                + args.shadow_input_ntuples_directory,
            )

            if args.mode == "xrootd":
                print "condor_{}_forGC.sh".format(args.executable)
                shellscript = os.path.join(
                    os.environ["CMSSW_BASE"],
                    "src",
                    gc_date_tag,
                    "condor_{}_forGC.sh".format(args.executable),
                )
                doReplace(
                    shellscript,
                    old=args.input_ntuples_directory,
                    new=server_xrootd[args.input_server]
                    + args.shadow_input_ntuples_directory,
                )

    elif args.command == "collect":
        collect.collect_outputs(
            args.executable, args.cores, args.custom_workdir_path, args.mode
        )
    elif args.command == "check":
        check.check_and_resubmit(
            args.executable, args.custom_workdir_path, args.mode, args.all, args.cores
        )


if __name__ == "__main__":
    main()
