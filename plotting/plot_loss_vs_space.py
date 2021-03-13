import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")


def get_best_loss_space(data):
    rshape = (len(data["space_list"]), -1)
    best_param_idx = np.argmin(data["loss_all"].reshape(rshape), axis=1)
    loss = data["loss_all"].reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    space_actual = (
        data["space_actual"].reshape(rshape)[np.arange(rshape[0]), best_param_idx] / 1e6
    )
    return loss, space_actual

def smooth_out(data, model_size):
    indices = np.argsort(data["space_actual"])
    space_actual = (data["space_actual"][indices] / 1e6 + model_size).tolist()
    loss_all = data["loss_all"][indices].tolist()
    while True:
        j = 0
        while j + 1 < len(space_actual) - 1:
            j += 1
            if loss_all[j + 1] > loss_all[j]:
                del loss_all[j + 1]
                del space_actual[j + 1]
                break

        if j == len(space_actual) - 1:
            break
    return loss_all, space_actual


class PlotLossVsSpace:
    def __init__(
        self,
        count_min,
        learned_cmin,
        model_names,
        perfect_ccm,
        lookup_table_ccm,
        model_sizes,
        lookup_size,
        x_lim,
        y_lim,
        title,
        algo,
    ):
        self.count_min = count_min
        self.learned_cmin = learned_cmin
        self.model_names = model_names
        self.perfect_ccm = perfect_ccm
        self.lookup_table_ccm = lookup_table_ccm
        self.model_sizes = model_sizes
        self.lookup_size = lookup_size
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.title = title
        self.algo = algo

    @classmethod
    def from_args(cls, a):
        return cls(
            a.count_min,
            a.learned_cmin,
            a.model_names,
            a.perfect_ccm,
            a.lookup_table_ccm,
            a.model_sizes,
            a.lookup_size,
            a.x_lim,
            a.y_lim,
            a.title,
            a.algo,
        )

    def run(self):
        args = self
        if args.learned_cmin:
            if not args.model_sizes:
                args.model_sizes = np.zeros(len(args.learned_cmin))
            assert len(args.learned_cmin) == len(
                args.model_names
            ), "provide names for the learned_cmin results"
            assert len(args.learned_cmin) == len(
                args.model_sizes
            ), "provide model sizes for the learned_cmin results"

        ax = plt.figure().gca()

        if args.count_min:
            data = np.load(args.count_min)
            space_cmin = data["space_list"]
            loss_cmin = np.amin(data["loss_all"], axis=0)
            ax.plot(space_cmin, loss_cmin, label=args.algo)

        if args.lookup_table_ccm:
            data = np.load(args.lookup_table_ccm)
            if len(data["loss_all"].shape) == 1:
                print("plot testing results for lookup table")
                ax.plot(
                    data["space_actual"] / 1e6 + args.lookup_size,
                    data["loss_all"],
                    linestyle="-.",
                    label="Table lookup " + args.algo,
                )
            else:
                loss_lookup, space_actual = get_best_loss_space(data)
                ax.plot(
                    space_actual,
                    loss_lookup + args.lookup_size,
                    linestyle="-.",
                    label="Table lookup " + args.algo,
                )

        if args.perfect_ccm:
            data = np.load(args.perfect_ccm)
            if len(data["loss_all"].shape) == 1:
                print("plot testing results for perfect CCM")
                ax.plot(
                    data["space_actual"] / 1e6,
                    data["loss_all"],
                    linestyle="-.",
                    label="Learned " + args.algo + " (Ideal)",
                )
            else:
                loss_cutoff_pf, space_actual = get_best_loss_space(data)
                ax.plot(
                    space_cmin,
                    loss_cutoff_pf,
                    linestyle="--",
                    label="Learned " + args.algo + " (Ideal)",
                )

        if args.learned_cmin:
            for i, cmin_result in enumerate(args.learned_cmin):
                data = np.load(cmin_result)
                if len(data["loss_all"].shape) == 1:
                    loss_all, space_actual = smooth_out(data, model_size=args.model_sizes[i])
                    ax.plot(
                        space_actual,
                        loss_all,
                        label=args.model_names[i],
                    )
                else:
                    loss_cutoff, space_actual = get_best_loss_space(data)
                    ax.plot(
                        space_actual + args.model_sizes[i],
                        loss_cutoff,
                        label=args.model_names[i],
                    )

        ax.set_ylabel("loss")
        ax.set_xlabel("space (MB)")
        if args.y_lim:
            ax.set_ylim(args.y_lim)
        if args.x_lim:
            ax.set_xlim(args.x_lim)

        title = "loss vs space - %s" % args.title
        ax.set_title(title)
        plt.legend(loc="upper right")
        plt.show()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--count_min", type=str, default="")
    argparser.add_argument("--learned_cmin", type=str, nargs="*", default=[])
    argparser.add_argument(
        "--model_names", type=str, nargs="*", default=["Learned Count Min"]
    )
    argparser.add_argument("--perfect_ccm", type=str, default="")
    argparser.add_argument("--lookup_table_ccm", type=str, default="")
    argparser.add_argument("--model_sizes", type=float, nargs="*", default=[])
    argparser.add_argument("--lookup_size", type=float, default=0.0)
    argparser.add_argument("--x_lim", type=float, nargs="*", default=[])
    argparser.add_argument("--y_lim", type=float, nargs="*", default=[])
    argparser.add_argument("--title", type=str, default="")
    argparser.add_argument("--algo", type=str, default="Alg")
    args = argparser.parse_args()

    plvs= PlotLossVsSpace.from_args(args)
    plvs.run()
