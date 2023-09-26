import gc
import os
import time
import warnings

import torch
from pelutils import log, TT
from pelutils.parser import Parser, Argument, Option

import frozone.train
from frozone.train.train import train


warnings.filterwarnings("always")

options = (
    Argument("env"),
    Argument("data-path"),
    Option("phase", type=str, default=None),

    # Training parameters
    Option("history-window", type=float, default=30),
    Option("prediction-window", type=float, default=20),
    Option("batches", type=int, default=10000),
    Option("batch-size", type=int, default=200),
    Option("lr", type=float, default=1e-5),
    Option("num-models", type=int, default=1),
    Option("max-num-data-files", type=int, default=0),
    Option("eval-size", type=int, default=4000),
    Option("loss-fn", default="l1", choices=["l1", "l2", "huber"]),
    Option("huber-delta", type=float, default=0.02),
    Option("alpha", type=float, default=0.5),
    Option("epsilon", type=float, default=0.0),
    Option("augment-prob", type=float, default=0.0),

    # Model parameters
    Option("dz", type=int, default=256),
    Option("encoder-name", default="FullyConnected"),
    Option("decoder-name", default="FullyConnected"),
    Option("fc-layer-num", type=int, default=3),
    Option("fc-layer-size", type=int, default=256),
    Option("t-layer-num", type=int, default=3),
    Option("t-nhead", type=int, default=6),
    Option("t-d-feedforward", type=int, default=1024),
    Option("dropout", type=float, default=0),
    Option("activation-fn", default="ReLU"),
)

if __name__ == "__main__":

    parser = Parser(*options, multiple_jobs=True)
    jobs = parser.parse_args()

    for job in jobs:
        job.prepare_directory()
        log.configure(os.path.join(job.location, "train.log"))
        with log.log_errors:
            log.section("Starting training of job %s" % job.name)
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
