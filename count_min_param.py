import os
import sys
import time
import argparse
import numpy as np
import itertools

from multiprocessing import Pool

from utils.conll_utils import get_data_conll_query
from utils.utils import get_stat, git_log, get_data_str_with_ports_list
from utils.aol_utils import get_data_aol_query
from sketch import count_min, count_sketch


def myfunc(y, n_buckets, n_hash, name):
    start_t = time.time()
    if name == "count_min":
        loss = count_min(y, n_buckets, int(n_hash))
    else:
        loss = count_sketch(y, n_buckets, int(n_hash))

    print(
        "%s: # hashes %d, # buckets %d - loss %.2f\t time: %.2f sec"
        % (name, n_hash, n_buckets, loss, time.time() - start_t)
    )
    return loss

class CountMinEvaluation:
    def __init__(self, data, save, seed, n_hashes_list, space_list, n_workers, data_name, count_sketch):
        self.data = data
        self.save = save
        self.seed = seed
        self.n_hashes_list = n_hashes_list
        self.space_list = space_list
        self.n_workers = n_workers
        self.data_name = data_name
        self.count_sketch = count_sketch

    @classmethod
    def from_args(cls, a):
        return cls(a.data, a.save, a.seed, a.n_hashes_list, a.space_list, a.n_workers, a.data_name, a.count_sketch)

    def run(self):
        args = self
        command = " ".join(sys.argv) + "\n"
        log_str = command
        # log_str += git_log() + "\n"
        # print(log_str)
        np.random.seed(args.seed)

        if args.data_name == "aol":
            assert len(args.data) == 1
            x, y = get_data_aol_query(args.data[0])
        elif args.data_name in ["conll", "wikicorpus"]:
            if isinstance(args.data, list) and len(args.data)==1:
                args.data = args.data[0]
            x, y = get_data_conll_query(args.data)
        else:
            x, y = get_data_str_with_ports_list(args.data)
        get_stat(args.data, x, y)

        if args.count_sketch:
            name = "count_sketch"
        else:
            name = "count_min"
        folder = os.path.join("param_results", name, "")
        if not os.path.exists(folder):
            os.makedirs(folder)

        nb_all = []
        nh_all = []
        for n_hash in args.n_hashes_list:
            for space in args.space_list:
                n_buckets = int(space * 1e6 / (n_hash * 4))
                nh_all.append(n_hash)
                nb_all.append(n_buckets)

        # myfunc(y, nb_all[0], nh_all[0], name)

        rshape = (len(args.n_hashes_list), len(args.space_list))

        start_t = time.time()
        pool = Pool(args.n_workers)
        results = pool.starmap(
            myfunc,
            zip(itertools.repeat(y), nb_all, nh_all, itertools.repeat(name))
        )
        pool.close()
        pool.join()

        results = np.reshape(results, rshape)
        nb_all = np.reshape(nb_all, rshape)
        nh_all = np.reshape(nh_all, rshape)

        log_str += "==== results ====\n"
        for i in range(len(results)):
            for j in range(len(results[i])):
                space = nh_all[i, j] * nb_all[i, j] * 4 / 1e6
                log_str += "%s: # hashes %d, # buckets %d, space %.2f MB - loss %.2f\n" % (
                    name,
                    nh_all[i, j],
                    nb_all[i, j],
                    space,
                    results[i, j],
                )
        log_str += "total time: %.2f sec\n" % (time.time() - start_t)

        log_str += "==== best parameters ====\n"
        best_param_idx = np.argmin(results, axis=0)
        best_n_buckets = nb_all[best_param_idx, np.arange(len(nb_all[0]))]
        best_n_hashes = nh_all[best_param_idx, np.arange(len(nb_all[0]))]
        best_loss = results[best_param_idx, np.arange(len(nb_all[0]))]
        for i in range(len(best_loss)):
            log_str += "space: %.2f, n_buckets %d, n_hashes %d - \tloss %.2f\n" % (
                args.space_list[i],
                best_n_buckets[i],
                best_n_hashes[i],
                best_loss[i],
            )
        log_str += "total time: %.2f sec\n" % (time.time() - start_t)

        print(log_str)
        with open(os.path.join(folder, args.save + ".log"), "w") as f:
            f.write(log_str)

        np.savez(
            os.path.join(folder, args.save),
            command=command,
            loss_all=results,
            n_hashes=nh_all,
            n_buckets=nb_all,
            space_list=args.space_list,
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument(
        "--data", type=str, nargs="*", help="list of input .npy data", required=True
    )
    argparser.add_argument(
        "--save", type=str, help="prefix to save the results", required=True
    )
    argparser.add_argument(
        "--seed", type=int, help="random state for sklearn", default=69
    )
    argparser.add_argument(
        "--n_hashes_list", type=int, nargs="*", help="number of hashes", required=True
    )
    argparser.add_argument(
        "--space_list", type=float, nargs="*", help="space in MB", required=True
    )
    argparser.add_argument(
        "--n_workers", type=int, help="number of workers", default=10
    )
    argparser.add_argument("--data_name", default="aol", choices=["aol", "ip", "conll", "wikicorpus"])
    argparser.add_argument("--count_sketch", action="store_true", default=False)
    args = argparser.parse_args()
    cme = CountMinEvaluation.from_args(args)
    cme.run()

