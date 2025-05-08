"""
Change userconf.yml a bit and train lots of models

"""

import os
import argparse
import subprocess

from fishjaw.util import util


def main():
    """
    Write stuff to userconf.yml, train a new model in a subprocess, and repeat

    """
    alphas = [0.22, 0.23, 0.28, 0.30, 0.32, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    betas = [1 - alpha for alpha in alphas]

    names = [
        f"alpha_{alpha:.2f}_beta_{beta:.2f}".replace(".", "_") + ".pkl"
        for alpha, beta in zip(alphas, betas)
    ]

    conf_path = util.rootdir() / "userconf.yml"
    assert conf_path.exists(), f"Config file {conf_path} does not exist"

    my_env = os.environ.copy()
    my_env["PYTHONPATH"] = str(util.rootdir())

    for alpha, beta, name in zip(alphas, betas, names):
        # Instead of reading the yaml, read the lines directly to preserve formatting and comments
        with open(conf_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith("model_path:"):
                lines[lines.index(line)] = f"model_path: {name}\n"

            if line.startswith('  "alpha":'):
                lines[lines.index(line)] = f'  "alpha": {alpha},\n'
            if line.startswith('  "beta":'):
                lines[lines.index(line)] = f'  "beta": {beta},\n'

        with open(conf_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        with open(f"log_{name}.txt", "w", encoding="utf-8") as f:
            subprocess.run(
                ["python", "scripts/train_model.py"],
                check=True,
                env=my_env,
                stdout=f,
                stderr=subprocess.PIPE,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Train lots of models with different alphas and betas"
    )
    main(**vars(parser.parse_args()))
