""" Standalone script for analyzing environment data. """
import math
import os
import random
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
from frozone.data import PHASE_TO_INDEX, PHASES, PROCESSED_SUBDIR, SIMULATED_SUBDIR, TEST_SUBDIR, TRAIN_SUBDIR, Dataset, list_processed_data_files
from frozone.data.dataloader import load_data_files


_plot_folder = "analysis-plots"

def analyse_full_floatzone_data(job: JobDescription, dataset: Dataset):

    env = environments.FloatZone

    with plots.Figure(os.path.join(job.location, _plot_folder, f"phase-dist.png"), figsize=(25, 18)):
        for phase, phase_index in PHASE_TO_INDEX.items():
            counts = list()
            for metadata, (X, U, S, R) in dataset:
                is_phase = env.is_phase(phase, S)
                if c := is_phase.sum():
                    counts.append(c)

            plt.subplot(2, 3, phase_index + 1)

            plt.hist(env.dt * np.array(counts), bins=30)
            plt.xlabel("Time [s]")
            plt.ylabel("Count")
            plt.title(PHASES[phase])

            plt.grid()

def analyse_processed_data(job: JobDescription, env: Type[environments.Environment], dataset: Dataset):

    columns = 4
    is_floatzone = env is environments.FloatZone
    rows = math.ceil((len(env.XLabels) + len(env.ULabels) + is_floatzone) / columns)

    log("Plotting %i samples" % len(dataset))
    for i, (metadata, (X, U, S, R)) in enumerate(dataset):
        with plots.Figure(os.path.join(job.location, _plot_folder, f"sample_{i}.png"), figsize=(15 * columns, 10 * rows)):
            subplot_no = 0

            if is_floatzone:
                for xlabel in env.XLabels:
                    subplot_no += 1
                    plt.subplot(rows, columns, subplot_no)
                    for phase_num, phase_index in PHASE_TO_INDEX.items():
                        phase_name = PHASES[phase_num]
                        is_phase = S[:, phase_index] == 1
                        plt.plot(
                            env.dt * np.where(is_phase)[0],
                            X[is_phase, xlabel],
                            color=plots.colours[phase_index],
                            lw=1.5,
                        )
                        if xlabel in env.reference_variables:
                            plt.plot(
                                env.dt * np.where(is_phase)[0],
                                R[is_phase, env.reference_variables.index(xlabel)],
                                color="black",
                                lw=1.2,
                                label="Reference" if phase_index == 0 else None,
                            )

                    if xlabel in env.reference_variables:
                        plt.legend()
                    plt.xlabel("Time [s]")
                    plt.title(env.format_label(xlabel))
                    plt.grid()

                for ulabel in env.ULabels:
                    subplot_no += 1
                    plt.subplot(rows, columns, subplot_no)
                    for phase_num, phase_index in PHASE_TO_INDEX.items():
                        phase_name = PHASES[phase_num]
                        is_phase = S[:, phase_index] == 1
                        plt.plot(env.dt * np.where(is_phase)[0], U[is_phase, ulabel], plots.colours[phase_index], lw=1.5)
                    plt.xlabel("Time [s]")
                    plt.title(env.format_label(ulabel))
                    plt.grid()

            else:
                for xlabel in env.XLabels:
                    subplot_no += 1
                    plt.subplot(rows, columns, subplot_no)
                    plt.plot(env.dt * np.arange(len(X)), X[:, xlabel], lw=1.5)
                    plt.xlabel("Time [s]")
                    plt.title(env.format_label(xlabel))
                    plt.grid()

                for ulabel in env.ULabels:
                    subplot_no += 1
                    plt.subplot(rows, columns, subplot_no)
                    plt.plot(env.dt * np.arange(len(U)), U[:, ulabel], lw=1.5)
                    plt.xlabel("Time [s]")
                    plt.title(env.format_label(ulabel))
                    plt.grid()

            if is_floatzone:
                subplot_no += 1
                plt.subplot(rows, columns, subplot_no)
                for phase_num, phase_index in PHASE_TO_INDEX.items():
                    phase_name = PHASES[phase_num]
                    is_phase = S[:, phase_index] == 1
                    plt.plot(env.dt * np.where(is_phase)[0], np.full(is_phase.sum(), phase_index), plots.colours[phase_index], lw=4, label=phase_name)
                plt.xlabel("Time [s]")
                plt.title("Phase")
                plt.legend(loc="lower right")
                plt.grid()

            plt.suptitle(os.path.split(metadata.raw_file or "Simulation")[-1] + f" ({metadata.date.year})", fontsize="xx-large")

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

        for i, (metadata, (X_full, U_full, S_full, R_full)) in enumerate(dataset):
            is_phase = env.is_phase(phase, S_full)
            if not is_phase.sum():
                continue
            X = X_full[is_phase]
            U = U_full[is_phase]
            R = R_full[is_phase]

            with plots.Figure(os.path.join(plot_folder, f"sample_{i}.png"), figsize=(15 * columns, 10 * rows)):
                subplot_no = 0

                for xlabel in env.XLabels:
                    subplot_no += 1
                    plt.subplot(rows, columns, subplot_no)
                    plt.plot(np.arange(len(X)) * env.dt, X[:, xlabel], lw=1.5, label="Observed")
                    if xlabel in env.reference_variables:
                        plt.plot(env.dt * np.arange(len(R)), R[:, env.reference_variables.index(xlabel)], lw=1.5, label="Reference")
                        plt.legend()
                    plt.xlabel("Time [s]")
                    plt.title(env.format_label(xlabel))
                    plt.grid()

                for ulabel in env.ULabels:
                    subplot_no += 1
                    plt.subplot(rows, columns, subplot_no)
                    plt.plot(np.arange(len(U)) * env.dt, U[:, ulabel], lw=1.5)
                    plt.xlabel("Time [s]")
                    plt.title(env.format_label(ulabel))
                    plt.grid()

                plt.suptitle(f"{phase_name} from {os.path.split(metadata.raw_file)[-1]} ({metadata.date.year})", fontsize="xx-large")

def analyse_simulated_data_floatzone(job: JobDescription, sim_dataset: Dataset, true_dataset: Dataset):

    columns = 4
    rows = math.ceil((len(env.XLabels) + len(env.ULabels)) / columns)

    log("Plotting %i simulation samples" % len(dataset))
    for i, ((metadata_sim, (X_sim, U_sim, S_sim, R_sim)), (metadata_true, (X_true, U_true, S_true, R_true))) in enumerate(zip(sim_dataset, true_dataset)):
        with plots.Figure(os.path.join(job.location, _plot_folder, f"sample_sim_{i}.png"), figsize=(15 * columns, 10 * rows)):
            subplot_no = 0
            is_phase = environments.FloatZone.is_phase("Cone", S_true)
            phase_index = PHASE_TO_INDEX[512]

            for xlabel in env.XLabels:
                subplot_no += 1
                plt.subplot(rows, columns, subplot_no)
                plt.plot(
                    env.dt * np.where(is_phase)[0],
                    X_true[is_phase, xlabel],
                    lw=1.5,
                    label="Observed",
                )
                plt.plot(
                    env.dt * np.where(is_phase)[0],
                    X_sim[:, xlabel],
                    lw=1.5,
                    label="Simulated",
                )
                if xlabel in env.reference_variables:
                    plt.plot(
                        env.dt * np.where(is_phase)[0],
                        R_true[is_phase, env.reference_variables.index(xlabel)],
                        color="black",
                        lw=1.2,
                        label="Reference",
                    )

                plt.xlabel("Time [s]")
                plt.title(env.format_label(xlabel))
                plt.grid()
                plt.legend()

            for ulabel in env.ULabels:
                subplot_no += 1
                plt.subplot(rows, columns, subplot_no)
                plt.plot(env.dt * np.where(is_phase)[0], U_true[is_phase, ulabel], lw=1.5, label="Observed")
                plt.plot(env.dt * np.where(is_phase)[0], U_sim[:, ulabel], lw=1, label="Simulated")
                plt.xlabel("Time [s]")
                plt.title(env.format_label(ulabel))
                plt.grid()
                plt.legend()

            plt.suptitle(os.path.split(metadata_true.raw_file or "Simulation")[-1] + f" ({metadata_true.date.year})", fontsize="xx-large")

if __name__ == "__main__":
    parser = Parser(
        Option("env", default="FloatZone"),
    )
    job = parser.parse_args()
    env = getattr(environments, job.env)

    log.configure(os.path.join(job.location, "analysis.log"))

    with log.log_errors:
        log.section("Loading data")
        test_data_files = list_processed_data_files(job.location, TEST_SUBDIR)
        dataset, _ = load_data_files(test_data_files, None, max_num_files=5, year=datetime.now().year)
        full_dataset, _ = load_data_files(test_data_files, None, max_num_files=len(test_data_files))

        shutil.rmtree(os.path.join(job.location, _plot_folder), ignore_errors=True)

        log.section("Analysing processed data")
        # analyse_processed_data(job, env, dataset)
        if env is environments.FloatZone:
            # analyse_processed_data_floatzone(job, dataset)
            log.section("Analysing full dataset")
            # analyse_full_floatzone_data(job, full_dataset)
            if SIMULATED_SUBDIR in os.listdir(job.location):
                log.section("Analysing simulated data")
                sim_data_files = list_processed_data_files(job.location, TRAIN_SUBDIR, SIMULATED_SUBDIR)
                random.shuffle(sim_data_files)
                sim_data_files = sim_data_files[:5]
                true_data_files = [s.replace(f"/{SIMULATED_SUBDIR}/", f"/{PROCESSED_SUBDIR}/") for s in sim_data_files]
                sim_dataset, _ = load_data_files(sim_data_files, None, year=0, shuffle=False)
                true_dataset, _ = load_data_files(true_data_files, None, year=0, shuffle=False)
                for (metadata_sim, _), (metadata_true, _) in zip(sim_dataset, true_dataset, strict=True):
                    assert metadata_sim.raw_file == metadata_true.raw_file, f"{metadata_sim.raw_file} == {metadata_true.raw_file}"
                analyse_simulated_data_floatzone(job, sim_dataset, true_dataset)
