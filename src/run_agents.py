import os
import multiprocessing
from multiprocessing import Process, JoinableQueue
import sys
import time
import argparse
from glob import glob
from os import path
from run import main, add_common_parser_opts, override_json_params
import json

q = JoinableQueue()
NUM_THREADS = multiprocessing.cpu_count() // 2

parser = argparse.ArgumentParser(description='Run all .json config files')
parser.add_argument('config_path', type=str, nargs='+',
                    help='json path to run all json files inside')
parser.add_argument('--out-dir-prefix', type=str, default="", required=False,
                    help='prefix for output log path')
parser = add_common_parser_opts(parser)
args = parser.parse_args()
agent_configs = args.config_path
args_params = vars(args)

def run_single_config(queue, index):
    # Set CPU affinity
    ncpus = multiprocessing.cpu_count() // 2
    os.sched_setaffinity(0, [index % ncpus, index % ncpus + ncpus])
    while True:
        conf_path = queue.get()
        json_params = json.load(open(conf_path))
        params = override_json_params(args_params, json_params, ['config_path', 'out_dir_prefix'])
        # Append a prefix for output path.
        if args.out_dir_prefix:
            params['out_dir'] = path.join(args.out_dir_prefix, params['out_dir'])
            print(f"setting output dir to {params['out_dir']}")
        try:
            print("Running config:", params)
            main(params)
        except Exception as e:
            print("ERROR", e)
            raise e
        queue.task_done()

for i in range(NUM_THREADS):
    worker = Process(target=run_single_config, args=(q,i))
    worker.daemon = True
    worker.start()

filelist = []
for c in agent_configs:
    filelist.extend(glob(path.join(c, "**/*.json"), recursive=True))
# De-duplicate.
filelist=list(set(filelist))

sorted_filelist = sorted(filelist, key=lambda x: 0 if 'humanoid' in x else 1)

for fname in sorted_filelist:
    print("python run.py --config-path {}".format(fname))
time.sleep(3)

for fname in sorted_filelist:
    q.put(fname)

q.join()
