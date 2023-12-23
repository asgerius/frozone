""" Standalone script for analyzing environment data. """
import math
import os
import shutil
from datetime import datetime
from glob import glob as glob  # glob
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots
from pelutils import log
from pelutils.parser import Parser, Option, JobDescription

import frozone.environments as environments
from frozone.data import PHASE_TO_INDEX, PHASES, PROCESSED_SUBDIR, TRAIN_SUBDIR, Dataset, list_processed_data_files
from frozone.data.dataloader import load_data_files
from frozone.plot import get_figure_args


_plot_folder = "analysis-plots"

def analyse_full_floatzone_data(job: JobDescription, dataset: Dataset):

    env = environments.FloatZone

    with plots.Figure(os.path.join(job.location, _plot_folder, f"phase-dist.pdf"), **get_figure_args(1, 1)):
        for phase, phase_index in PHASE_TO_INDEX.items():
            counts = list()
            for metadata, (X, U, S, R, Z) in dataset:
                is_phase = env.is_phase(phase, S)
                if c := is_phase.sum():
                    counts.append(c)

            plt.subplot(2, 3, phase_index + 1)

            plt.hist(env.dt * np.array(counts) / 3600, bins=30)
            plt.xlabel("Time [h]")
            plt.ylabel("Count")
            plt.title(PHASES[phase])

            plt.grid()

def analyse_processed_data(job: JobDescription, env: Type[environments.Environment], dataset: Dataset):

    columns = 4
    is_floatzone = env is environments.FloatZone
    rows = math.ceil((len(env.XLabels) + len(env.ULabels) + len(env.ZLabels) + is_floatzone) / columns)

    log("Plotting %i samples" % len(dataset))
    for i, (metadata, (X, U, S, R, Z)) in enumerate(dataset):
        with plots.Figure(os.path.join(job.location, _plot_folder, f"sample_{i}.pdf"), **get_figure_args(rows, columns)):
            subplot_no = 0

            if is_floatzone:
                for xlabel in env.XLabels:
                    if xlabel is environments.FloatZone.XLabels.FullPolyDia:
                        continue
                    subplot_no += 1
                    plt.subplot(rows, columns, subplot_no)
                    for phase_num, phase_index in PHASE_TO_INDEX.items():
                        phase_name = PHASES[phase_num]
                        is_phase = S[:, phase_index] == 1
                        plt.plot(
                            env.dt * np.where(is_phase)[0] / 3600,
                            X[is_phase, xlabel],
                            color=plots.colours[phase_index],
                        )
                        if xlabel in env.reference_variables:
                            plt.plot(
                                env.dt * np.where(is_phase)[0] / 3600,
                                R[is_phase, env.reference_variables.index(xlabel)],
                                color="red",
                                lw=3,
                                label="Target" if phase_index == 0 else None,
                            )

                    if xlabel in env.reference_variables:
                        plt.legend()
                    plt.xlabel("Time [h]")
                    plt.title(env.format_label(xlabel))
                    plt.grid()

                for ulabel in env.ULabels:
                    subplot_no += 1
                    plt.subplot(rows, columns, subplot_no)
                    for phase_num, phase_index in PHASE_TO_INDEX.items():
                        phase_name = PHASES[phase_num]
                        is_phase = S[:, phase_index] == 1
                        plt.plot(env.dt * np.where(is_phase)[0] / 3600, U[is_phase, ulabel], plots.colours[phase_index])
                    plt.xlabel("Time [h]")
                    plt.title(env.format_label(ulabel))
                    plt.grid()

            else:
                for xlabel in env.XLabels:
                    if xlabel is environments.Steuermann.XLabels.FullPolyDia:
                        continue
                    subplot_no += 1
                    plt.subplot(rows, columns, subplot_no)
                    plt.plot(env.dt * np.arange(len(X)), X[:, xlabel], label="Observed")
                    if xlabel in env.reference_variables:
                        plt.plot(env.dt * np.arange(len(X)) / 3600, R[:, env.reference_variables.index(xlabel)], color="red", label="Target")
                    plt.xlabel("Time [h]")
                    plt.title(env.format_label(xlabel))
                    plt.legend()
                    plt.grid()

                for ulabel in env.ULabels:
                    subplot_no += 1
                    plt.subplot(rows, columns, subplot_no)
                    plt.plot(env.dt * np.arange(len(U)) / 3600, U[:, ulabel])
                    plt.xlabel("Time [h]")
                    plt.title(env.format_label(ulabel))
                    plt.grid()

                for zlabel in env.ZLabels:
                    subplot_no += 1
                    plt.subplot(rows, columns, subplot_no)
                    plt.plot(env.dt * np.arange(len(Z)) / 3600, Z[:, zlabel])
                    plt.xlabel("Time [h]")
                    plt.title(env.format_label(zlabel))
                    plt.grid()

            if is_floatzone:
                subplot_no += 1
                plt.subplot(rows, columns, subplot_no)
                for phase_num, phase_index in PHASE_TO_INDEX.items():
                    phase_name = PHASES[phase_num]
                    is_phase = S[:, phase_index] == 1
                    plt.plot(env.dt * np.where(is_phase)[0] / 3600, np.full(is_phase.sum(), phase_index), plots.colours[phase_index], lw=5, label=phase_name)
                plt.xlabel("Time [h]")
                plt.legend(ncol=3)
                plt.grid()

            # plt.suptitle(os.path.split(metadata.raw_file or "Simulation")[-1] + f" ({metadata.date.year})", fontsize="xx-large")

def analyse_processed_data_floatzone(job: JobDescription, dataset: Dataset):

    env = environments.FloatZone

    columns = 4
    rows = math.ceil((len(env.XLabels) + len(env.ULabels)) / columns)

    log("Plotting %i samples for each phase" % len(dataset))

    for phase in PHASE_TO_INDEX:

        phase_name = PHASES[phase]
        log("Plotting phase %s" % phase_name)

        plot_folder = os.path.join(job.location, _plot_folder + " " + phase_name)
        shutil.rmtree(plot_folder, ignore_errors=True)

        for i, (metadata, (X_full, U_full, S_full, R_full, *_)) in enumerate(dataset):
            is_phase = env.is_phase(phase, S_full)
            if not is_phase.sum():
                continue
            X = X_full[is_phase]
            U = U_full[is_phase]
            R = R_full[is_phase]

            with plots.Figure(os.path.join(plot_folder, f"sample_{i}.pdf"), **get_figure_args(rows, columns)):
                subplot_no = 0

                for xlabel in env.XLabels:
                    if xlabel is env.XLabels.FullPolyDia:
                        continue
                    subplot_no += 1
                    plt.subplot(rows, columns, subplot_no)
                    plt.plot(env.dt * np.arange(len(X)) / 3600, X[:, xlabel], label="Observed")
                    if xlabel in env.reference_variables:
                        plt.plot(env.dt * np.arange(len(R)) / 3600, R[:, env.reference_variables.index(xlabel)], lw=3, c="red", label="Target")
                        plt.legend()
                    plt.xlabel("Time [h]")
                    plt.title(env.format_label(xlabel))
                    plt.grid()

                for ulabel in env.ULabels:
                    subplot_no += 1
                    plt.subplot(rows, columns, subplot_no)
                    plt.plot(env.dt * np.arange(len(U)) / 3600, U[:, ulabel])
                    plt.xlabel("Time [h]")
                    plt.title(env.format_label(ulabel))
                    plt.grid()

                # plt.suptitle(f"{phase_name} from {os.path.split(metadata.raw_file)[-1]} ({metadata.date.year})", fontsize="xx-large")

def analyse_simulated_data_floatzone(job: JobDescription, sim_dataset: Dataset, true_dataset: Dataset):

    columns = 4
    rows = math.ceil((len(env.XLabels) + len(env.ULabels)) / columns)

    log("Plotting %i simulation samples" % len(dataset))
    for i, ((metadata_sim, (X_sim, U_sim, S_sim, R_sim)), (metadata_true, (X_true, U_true, S_true, R_true))) in enumerate(zip(sim_dataset, true_dataset)):
        with plots.Figure(os.path.join(job.location, _plot_folder, f"sample_sim_{i}.pdf"), **get_figure_args(rows, columns)):
            subplot_no = 0
            is_phase = environments.FloatZone.is_phase("Cone", S_true)
            phase_index = PHASE_TO_INDEX[512]

            for xlabel in env.XLabels:
                subplot_no += 1
                plt.subplot(rows, columns, subplot_no)
                plt.plot(
                    env.dt * np.where(is_phase)[0] / 3600,
                    X_true[is_phase, xlabel],
                    label="Observed",
                )
                plt.plot(
                    env.dt * np.where(is_phase)[0] / 3600,
                    X_sim[:, xlabel],
                    label="Simulated",
                )
                if xlabel in env.reference_variables:
                    plt.plot(
                        env.dt * np.where(is_phase)[0] / 3600,
                        R_true[is_phase, env.reference_variables.index(xlabel)],
                        color="red",
                        lw=3,
                        label="Target",
                    )

                plt.xlabel("Time [h]")
                plt.title(env.format_label(xlabel))
                plt.grid()
                plt.legend()

            for ulabel in env.ULabels:
                subplot_no += 1
                plt.subplot(rows, columns, subplot_no)
                plt.plot(env.dt * np.where(is_phase)[0], U_true[is_phase, ulabel] / 3600, label="Observed")
                plt.plot(env.dt * np.where(is_phase)[0], U_sim[:, ulabel] / 3600, lw=3, label="Simulated")
                plt.xlabel("Time [h]")
                plt.title(env.format_label(ulabel))
                plt.grid()
                plt.legend()

            # plt.suptitle(os.path.split(metadata_true.raw_file or "Simulation")[-1] + f" ({metadata_true.date.year})", fontsize="xx-large")

if __name__ == "__main__":
    parser = Parser(
        Option("env", default="FloatZone"),
    )
    job = parser.parse_args()
    env = getattr(environments, job.env)

    log.configure(os.path.join(job.location, "analysis.log"))

    with log.log_errors:
        log.section("Loading data")
        train_data_files = list_processed_data_files(job.location, TRAIN_SUBDIR)
        dataset, _ = load_data_files(train_data_files, None, max_num_files=5, year=datetime.now().year, with_tqdm=True)
        full_dataset, _ = load_data_files(train_data_files, None, max_num_files=len(train_data_files), with_tqdm=True)

        shutil.rmtree(os.path.join(job.location, _plot_folder), ignore_errors=True)

        log.section("Analysing processed data")
        analyse_processed_data(job, env, dataset)
        if env is environments.FloatZone:
            analyse_processed_data_floatzone(job, dataset)
            log.section("Analysing full dataset")
            analyse_full_floatzone_data(job, full_dataset)
