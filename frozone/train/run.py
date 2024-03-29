import gc
import os
import time

import torch
from pelutils import log, TT
from pelutils.parser import Parser, Argument, Option, Flag

import frozone.train
from frozone.train.train import train


options = (
    Argument("env"),
    Argument("data-path"),
    Option("phase", type=str, default=None),

    # Training parameters
    Option("history-window", type=float, default=30),
    Option("prediction-window", type=float, default=20),
    Option("history-interp", type=float, default=1),
    Option("prediction-interp", type=float, default=1),
    Option("batches", type=int, default=10000),
    Option("batch-size", type=int, default=200),
    Option("lr", type=float, default=1e-5),
    Option("num-models", type=int, default=1),
    Option("max-num-data-files", type=int, default=0),
    Option("eval-size", type=int, default=4000),
    Option("loss-fn", default="l1", choices=["l1", "l2", "huber"]),
    Option("huber-delta", type=float, default=0.1),
    Option("alpha", type=float, default=1, help="Exponential smoothing factor"),

    # Model parameters
    Option("dz", type=int, default=120),
    Option("encoder-name", default="FullyConnected"),
    Option("decoder-name", default="FullyConnected"),
    Option("fc-layer-num", type=int, default=4),
    Option("fc-layer-size", type=int, default=756),
    Option("t-layer-num", type=int, default=4),
    Option("t-nhead", type=int, default=4),
    Option("t-d-feedforward", type=int, default=384),
    Option("dropout", type=float, default=0),
    Option("activation-fn", default="GELU"),
)

if __name__ == "__main__":

    parser = Parser(*options, multiple_jobs=True)
    jobs = parser.parse_args()

    for job in jobs:
        job.prepare_directory()
        log.configure(os.path.join(job.location, "train.log"))
        with log.log_errors:
            log.section("Starting training of job %s" % job.name)
            log.log_repo()
            try:
                frozone.train.is_doing_training = True
                train(job)
            finally:
                frozone.train.is_doing_training = False

            # Wait a little between jobs to make sure data loader thread has time to stop
            time.sleep(5)

            TT.reset()

            # Based on sad experiences in the past, the following may or may not be necessary
            # to prevent a memory leak when running multiple jobs.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
