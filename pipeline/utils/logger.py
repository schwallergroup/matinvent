import os
import sys
import logging
from typing import Any
import wandb
import pandas as pd


class SeverityLevelBetween(logging.Filter):
    def __init__(self, min_level: int, max_level: int) -> None:
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level

    def filter(self, record) -> bool:
        return self.min_level <= record.levelno < self.max_level


def setup_logging() -> None:
    root = logging.getLogger()
    # Perform setup only if logging has not been configured
    target_logging_level = getattr(logging, os.environ.get("LOGLEVEL", "INFO").upper())
    root.setLevel(target_logging_level)
    if not root.hasHandlers():
        log_formatter = logging.Formatter(
            "%(asctime)s (%(levelname)s): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Send INFO (or target) to stdout
        handler_out = logging.StreamHandler(sys.stdout)
        handler_out.addFilter(
            SeverityLevelBetween(target_logging_level, logging.WARNING)
        )
        handler_out.setFormatter(log_formatter)
        root.addHandler(handler_out)

        # Send WARNING (and higher) to stderr
        handler_err = logging.StreamHandler(sys.stderr)
        handler_err.setLevel(logging.WARNING)
        handler_err.setFormatter(log_formatter)
        root.addHandler(handler_err)


class Logger():
    """Generic class to interface with various logging modules, e.g. wandb,
    tensorboard, etc.
    """

    def __init__(self, config=None) -> None:
        self.config = config

    # @abstractmethod
    # def watch(self, model, log_freq: int = 1000):
    #     """
    #     Monitor parameters and gradients.
    #     """

    def log(self, update_dict, step: int, split: str = ""):
        """
        Log some values.
        """
        assert step is not None
        if split != "":
            new_dict = {}
            for key in update_dict:
                new_dict[f"{split}/{key}"] = update_dict[key]
            update_dict = new_dict
        return update_dict

    # @abstractmethod
    # def log_plots(self, plots) -> None:
    #     pass

    # @abstractmethod
    # def mark_preempting(self) -> None:
    #     pass

    # @abstractmethod
    # def log_summary(self, summary_dict: dict[str, Any]) -> None:
    #     pass

    # @abstractmethod
    # def log_artifact(self, name: str, type: str, file_location: str) -> None:
    #     pass


class WandBLogger(Logger):
    def __init__(
        self,
        name,
        project,
        entity=None,
        group=None,
        mode='online',
        config=None,
        **kwargs,
    ) -> None:
        super().__init__(config)

        wandb.init(
            name=name,
            project=project,
            entity=entity,
            group=group,
            mode=mode,
            config=config,
            **kwargs,
        )

    def watch(self, model, log="all", log_freq: int = 1000) -> None:
        wandb.watch(model, log=log, log_freq=log_freq)

    def log(self, update_dict, step: int, split: str = "") -> None:
        update_dict = super().log(update_dict, step, split)
        wandb.log(update_dict, step=int(step))

    def log_plots(self, plots, caption: str = "") -> None:
        assert isinstance(plots, list)
        plots = [wandb.Image(x, caption=caption) for x in plots]
        wandb.log({"data": plots})

    def log_table(
        self, name: str, cols: list, data: list, step: int = None, commit=False
    ) -> None:
        # cols are 1D list of N elements, data must be NxK where the number of cols must match cols
        # see https://docs.wandb.ai/guides/tables
        table = wandb.Table(columns=cols, data=data)
        wandb.log({name: table}, step=step, commit=commit)

    def log_summary(self, summary_dict: dict[str, Any]):
        for k, v in summary_dict.items():
            wandb.run.summary[k] = v

    def mark_preempting(self) -> None:
        wandb.mark_preempting()

    def log_artifact(self, name: str, type: str, file_location: str) -> None:
        art = wandb.Artifact(name=name, type=type)
        art.add_file(file_location)
        art.save()


class CSVLogger(Logger):
    def __init__(
        self,
        save_dir: str,
        fname: str = 'metrics',
        config=None,
        **kwargs,
    ) -> None:
        super().__init__(config)
        self.save_dir = save_dir
        self.fname = fname
        self.df = None

    def log(self, update_dict, step: int, split: str = "") -> None:
        update_dict = super().log(update_dict, step, split)
        update_dict['step'] = step
        if self.df is None:
            self.df = pd.DataFrame([update_dict])
        else:
            self.df = pd.concat([self.df, pd.DataFrame([update_dict])])
        fpath = os.path.join(self.save_dir, f'{self.fname}.csv')
        self.df.to_csv(fpath, index=False)
