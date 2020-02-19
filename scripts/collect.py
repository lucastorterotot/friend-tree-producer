import os, json, logging

logger = logging.getLogger("job_managment")
import ROOT as r
from multiprocessing import Pool

r.PyConfig.IgnoreCommandLineOptions = True
r.gROOT.ProcessLine("gErrorIgnoreLevel = 2001;")


def write_trees_to_files(info):
    nick = info[0]
    collection_path = info[1]
    db = info[2]
    print "Copying trees for %s" % nick
    nick_path = os.path.join(collection_path, nick)
    if not os.path.exists(nick_path):
        os.mkdir(nick_path)
    outputfile = r.TFile.Open(os.path.join(nick_path, nick + ".root"), "recreate")
    for p in db[nick]["pipelines"]:
        if db[nick]["pipelines"][p] > 0:
            outputfile.mkdir(p)
            outputfile.cd(p)
            tree = db[nick][p].CloneTree()
            tree.Write("", r.TObject.kOverwrite)
            db[nick][p].Reset()
    outputfile.Close()


def collect_outputs(executable, cores, custom_workdir_path, mode):
    if custom_workdir_path:
        workdir_path = os.path.join(custom_workdir_path, executable + "_workdir")
    else:
        workdir_path = os.path.join(
            os.environ["CMSSW_BASE"], "src", executable + "_workdir"
        )
    jobdb_path = os.path.join(workdir_path, "condor_" + executable + ".json")
    datasetdb_path = os.path.join(workdir_path, "dataset.json")
    gc_path = os.path.join(workdir_path, "grid_control_{}.conf".format(executable))
    jobdb_file = open(jobdb_path, "r")
    jobdb = json.loads(jobdb_file.read())
    datasetdb_file = open(datasetdb_path, "r")
    datasetdb = json.loads(datasetdb_file.read())
    collection_path = os.path.join(workdir_path, executable + "_collected")
    if not os.path.exists(collection_path):
        os.makedirs(collection_path)
    # print jobdb
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
            filename = "_".join([nick, pipeline, str(first), str(last)]) + ".root"
            if mode == "local":
                filepath = os.path.join(workdir_path, nick, filename)
            elif mode == "xrootd":
                with open(gc_path, "r") as gc_file:
                    for line in gc_file.readlines():
                        if "se path" in line:
                            filepath = (
                                server_xrootd["GridKA"]
                                + "/store/"
                                + line.split("/store/")[1].strip("\n")
                                + "/"
                                + filename
                            )
                            break
            datasetdb[nick].setdefault(
                pipeline, r.TChain("/".join([pipeline, tree]))
            ).Add(filepath)

    nicks = sorted(datasetdb)
    if mode == "local":
        pool = Pool(cores)
        pool.map(
            write_trees_to_files,
            zip(nicks, [collection_path] * len(nicks), [datasetdb] * len(nicks)),
        )
    elif mode == "xrootd":  # it did not complete when using Pool in xrootd mode
        for nick in nicks:
            write_trees_to_files([nick, collection_path, datasetdb])
