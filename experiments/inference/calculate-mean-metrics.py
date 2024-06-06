# basic
import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import yaml
from loguru import logger
from tqdm import tqdm

# read all json files under metrics folder
base = Path(__file__).parent.resolve()
json_files = list(base.glob("**/*.json"))


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Filter out problematic AbM numbered file identifiers.",
        epilog=textwrap.dedent(
            """
        Example usage:
            python calculate-mean-mcc.py /path/to/json_dir
        """
        ),
    )
    parser.add_argument(
        "json_dir", type=Path, help="folder containing json files from evaluate output"
    )
    parser.add_argument(
        "-json",
        "--json_path",
        type=Path,
        default=None,
        help="output file to save the mean mcc and std",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not print the mean mcc and std to stdout",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = cli()
    assert args.json_dir.is_dir(), f"{args.json_dir} is not a directory"
    json_files = list(args.json_dir.glob("**/*.json"))
    assert json_files, f"No json files found in {args.json_dir}"
    metrics = ["mcc", "roc_auc", "precision", "recall", "f1", "tp", "tn", "fp", "fn"]
    # find the value mcc from the key metrics_epitope_pred
    avg_node_dict = {}
    avg_link_dict = {}
    for metric in metrics:
        logger.info(f"Calculating mean and std for {metric}")
        metric_dict = {"node": {}, "link": {}}
        for file in tqdm(json_files):
            with open(file, "r") as f:
                name = file.stem
                data = json.load(f)
                metric_dict["node"][name] = None
                metric_dict["link"][name] = None
                try:
                    metric_dict["node"][name] = data["metrics_epitope_pred"][metric]
                    metric_dict["link"][name] = data["metrics_bi_adj"][metric]
                except Exception as e:
                    logger.error(f"Error in {file}: {e}")
                    continue

        node_arr = np.array(list(metric_dict["node"].values()))
        link_arr = np.array(list(metric_dict["link"].values()))
        metric_node_dict = {
            "mean": node_arr.mean(),
            "standard deviation": node_arr.std(),
            "standard error": node_arr.std() / np.sqrt(len(node_arr)),
            "metadata": {"n": len(node_arr), metric: metric_dict["node"]},
        }
        metric_link_dict = {
            "mean": link_arr.mean(),
            "standard deviation": link_arr.std(),
            "standard error": link_arr.std() / np.sqrt(len(link_arr)),
            "metadata": {"n": len(link_arr), metric: metric_dict["link"]},
        }
        avg_node_dict[metric] = metric_node_dict
        avg_link_dict[metric] = metric_link_dict
    d = {
        "node": avg_node_dict,
        "link": avg_link_dict,
    }

    # print json
    if not args.quiet:
        for metric in metrics:
            print(
                f"Node {metric} mean: {d['node'][metric]['mean']:.5f}; stddev: {d['node'][metric]['standard deviation']:.5f}; stderr: {d['node'][metric]['standard error']:.5f}"
            )
            print(
                f"Link {metric} mean: {d['link'][metric]['mean']:.5f}; stddev: {d['link'][metric]['standard deviation']:.5f}; stderr: {d['link'][metric]['standard error']:.5f}"
            )
    # save file if needed
    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump(d, f, indent=2)
