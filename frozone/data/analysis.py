""" Standalone script for analyzing the raw float-zone data. """
import os
import random
from collections import defaultdict, Counter
from glob import glob as glob  # glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pelutils.ds.plots as plots
from pelutils import log, split_path, thousands_seperators
from pelutils.parser import Parser, Option
from tqdm import tqdm

from frozone.data import RAW_SUBDIR, PHASES


_plot_folder = "analysis-plots"

def plot_line_distributions(path: str, num_lines: list[int], num_lines_by_machine: defaultdict[str, list[int]]):
    num_lines = np.array(num_lines)
    num_lines = num_lines[num_lines < 15000]
    with plots.Figure(os.path.join(path, _plot_folder, "line-distributions.png"), figsize=(23, 10)):
        plt.subplot(121)
        plt.hist(num_lines, bins=50, density=True)
        plt.title("Number of lines")
        plt.xlabel("Number of data lines")
        plt.ylabel("Probability density")
        plt.xlim(left=-1000, right=16000)
        plt.grid()

        plt.subplot(122)
        for machine, lines in num_lines_by_machine.items():
            lines = np.array(lines)
            lines = lines[lines < 15000]
            plt.plot(*plots.histogram(lines, bins=15), "-o")
        plt.title("Number of lines by machine")
        plt.xlabel("Number of data lines")
        plt.ylabel("Probability density")
        plt.xlim(left=-1000, right=16000)
        plt.grid()

def plot_phase_distribution(path: str, num_lines_by_phase: defaultdict[int, list[int]]):
    with plots.Figure(os.path.join(path, _plot_folder, "phase-distributions.png"), legend_fontsize=0.75):
        for i, phase_number in enumerate(sorted(num_lines_by_phase.keys())):
            num_lines = np.array(num_lines_by_phase[phase_number])
            plt.axvline(num_lines.mean(), color=plots.colours[i], lw=2)
            num_lines = num_lines[num_lines<10000]
            plt.plot(
                *plots.histogram(num_lines, bins=25, density=True, ignore_zeros=True),
                "--o",
                color=plots.colours[i],
                label=PHASES[phase_number],
            )
        plt.title("Number of data lines by phase")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Number of lines")
        plt.ylabel("Frequency")
        plt.legend(loc=3)
        plt.grid()

if __name__ == "__main__":
    parser = Parser(Option("max-files", type=int, default=None))
    job = parser.parse_args()

    log.configure(os.path.join(job.location, "analysis.log"), print_level=None)

    with log.log_errors:
        raw_files = glob(os.path.join(job.location, RAW_SUBDIR, "**", "*.txt"), recursive=True)
        random.shuffle(raw_files)
        raw_files = raw_files[:job.max_files]

        num_lines = list()
        num_lines_by_machine = defaultdict(list)
        num_lines_by_phase = defaultdict(list)

        log.section("Analyzing %s raw data files" % thousands_seperators(len(raw_files)))
        for raw_file in tqdm(raw_files):

            try:
                df = pd.read_csv(raw_file, quoting=3, delim_whitespace=True)
            except Exception as e:
                log.warning("Failed to load %s with the error" % raw_file, e)

            path_components = split_path(raw_file)
            machine = path_components[2]
            num_lines.append(len(df))
            num_lines_by_machine[machine].append(len(df))

            try:
                lines_by_phase = Counter(df.Growth_State_Act.values)
            except AttributeError:
                log.warning("No Growth_State_Act in %s" % raw_file)
                continue

            for phase_number, count in lines_by_phase.items():
                if phase_number not in PHASES:
                    log.warning("Phase number %i encountered in %s" % (phase_number, raw_file))
                    continue
                num_lines_by_phase[phase_number].append(count)

        log.section("Plotting")
        plot_line_distributions(job.location, num_lines, num_lines_by_machine)
        plot_phase_distribution(job.location, num_lines_by_phase)
