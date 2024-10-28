from lightning.pytorch.callbacks import Callback
import numpy as np
import nni
import copy
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ReportMetrics(Callback):

    def __init__(self, nni_report=True):
        self.training_results = []  # (train_loss, train_accuracy)
        self.validation_results = []  # (val_loss, val_accuracy)
        self.nni_report = nni_report
        self.save_metrics = False

    def on_fit_start(self, trainer, pl_module):
        self.save_metrics = True

    def on_fit_end(self, trainer, pl_module):
        self.save_metrics = False

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.save_metrics:
            return
        epoc_loss_val = trainer.logged_metrics['val_loss'].item()
        epoc_accuracy_val = trainer.logged_metrics['val_accuracy'].item()
        self.validation_results.append([copy.copy(epoc_loss_val), copy.copy(epoc_accuracy_val)])
        epoc_loss_train = trainer.logged_metrics['train_loss'].item()
        epoc_accuracy_train = trainer.logged_metrics['train_accuracy'].item()
        self.training_results.append([copy.copy(epoc_loss_train), copy.copy(epoc_accuracy_train)])

        if self.nni_report:
            nni.report_intermediate_result({"default": np.round(epoc_accuracy_val * 100, 4),
                                            "training acc.": np.round(epoc_accuracy_train * 100,
                                                                      4),
                                            "val. loss": np.round(epoc_loss_val, 4),
                                            "train. loss": np.round(epoc_loss_train, 4)
                                            })

        logger.debug(
            f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}: \n\t"
            f"training loss: {self.training_results[-1][0]} \n\t"
            f"validation loss: {self.validation_results[-1][0]} \n\t"
            f"training accuracy: {np.round(self.training_results[-1][1] * 100, 4)}% \n\t"
            f"validation accuracy: {np.round(self.validation_results[-1][1] * 100, 4)}%\n"
        )

    def state_dict(self):
        return {
            "train_results": self.training_results,
            "validation_results": self.validation_results,
        }

    def load_state_dict(self, state_dict):
        self.training_results = state_dict['train_results']
        self.validation_results = state_dict['validation_results']


class ValidationDeltaStopping(Callback):
    def __init__(self):
        super().__init__()
        self.EarlyStop_delta_val_loss = 0.5  # intended as the percentage of change in validation loss
        self.counter_small_delta_loss = 0
        self.stop_small_delta_loss = 5  # how many times the condition must be met to induce early stopping
        self.EarlyStop_delta_val_loss_up = 0.5  # intended as the percentage of increase in validation loss
        self.counter_delta_loss_up = 0
        self.stop_delta_loss_up = 5  # how many times the condition must be met to induce early stopping
        self.EarlyStop_delta_val_acc_low = 0.1  # intended as the percentage of change in validation accuracy
        self.counter_small_delta_acc = 0
        self.stop_small_delta_acc = 5  # how many times the condition must be met to induce early stopping
        self.EarlyStop_delta_val_acc_high = 2  # intended as the percentage of decrease in validation accuracy
        self.counter_large_delta_acc = 0
        self.stop_large_delta_acc = 5  # how many times the condition must be met to induce early stopping
        self.stopped_epoch = 0
        self.verbose = True
        self.validation_results = []  # (val_loss, val_accuracy)

    def on_validation_end(self, trainer, pl_module) -> None:
        epoc_loss_val = trainer.logged_metrics['val_loss'].item()
        epoc_accuracy_val = trainer.logged_metrics['val_accuracy'].item()
        self.validation_results.append([copy.copy(epoc_loss_val), copy.copy(epoc_accuracy_val)])
        self._run_early_stopping_check(trainer)

    def _run_early_stopping_check(self, trainer):

        if trainer.current_epoch >= 2:
            # Check loss variations (on validation data) during training: count number of epoch with SMALL (<
            # EarlyStop_delta_val_loss %) CHANGES
            if np.abs(self.validation_results[-1][0] - self.validation_results[-2][0]) / \
                    self.validation_results[-2][0] * 100 < self.EarlyStop_delta_val_loss:
                self.counter_small_delta_loss += 1
            else:
                self.counter_small_delta_loss = 0
            # Check loss variations (on validation data) during training: count number of epoch with LARGE (>
            # EarlyStop_delta_val_loss_up %) INCREASE
            if (self.validation_results[-1][0] - self.validation_results[-2][0]) / \
                    self.validation_results[-2][0] * 100 > self.EarlyStop_delta_val_loss_up:
                self.counter_delta_loss_up += 1
            else:
                self.counter_delta_loss_up = 0
            # check accuracy variations (on validation data) during training: count number of epoch with SMALL (>
            # EarlyStop_delta_val_acc %) CHANGES
            if np.abs(self.validation_results[-1][1] - self.validation_results[-2][1]) / \
                    self.validation_results[-2][1] * 100 < self.EarlyStop_delta_val_acc_low:
                self.counter_small_delta_acc += 1
            else:
                self.counter_small_delta_acc = 0
            # check accuracy variations (on validation data) during training: count number of epoch with LARGE (>
            # EarlyStop_delta_val_acc %) DECREASE
            if (self.validation_results[-2][1] - self.validation_results[-1][1]) / \
                    self.validation_results[-2][1] * 100 > self.EarlyStop_delta_val_acc_high:
                self.counter_large_delta_acc += 1
            else:
                self.counter_large_delta_acc = 0

            should_stop, reasons = self._evaluate_stopping_criteria(trainer)

            should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
            trainer.should_stop = trainer.should_stop or should_stop
            if should_stop:
                self.stopped_epoch = trainer.current_epoch
            if len(reasons) > 0 and self.verbose:
                self._log_info(reasons)

    def _evaluate_stopping_criteria(self, trainer):
        should_stop = False
        reasons = []

        if self.counter_small_delta_loss >= self.stop_small_delta_loss:
            should_stop = True
            reason = (
                f"Training stopped after {trainer.current_epoch}/{trainer.max_epochs} epochs: "
                "stop condition for small validation loss changes met."
            )

            reasons.append(reason)
        if self.counter_delta_loss_up >= self.stop_delta_loss_up:
            should_stop = True
            reason = (
                f'Training stopped after {trainer.current_epoch}/{trainer.max_epochs} epochs: '
                f'stop condition for validation loss increase met.'
            )
            reasons.append(reason)
        if self.counter_small_delta_acc >= self.stop_small_delta_acc:
            should_stop = True
            reason = (
                f'Training stopped after {trainer.current_epoch}/{trainer.max_epochs} epochs: '
                f'stop condition for small validation accuracy changes met.'
            )
            reasons.append(reason)
        if self.counter_large_delta_acc >= self.stop_large_delta_acc:
            should_stop = True
            reason = (
                f'Training stopped after {trainer.current_epoch}/{trainer.max_epochs} epochs: '
                f'stop condition for validation accuracy decrease met.'
            )
            reasons.append(reason)
        return should_stop, reasons

    @staticmethod
    def _log_info(messages):
        for message in messages:
            logger.debug(f'{message}\n')

    def state_dict(self):
        return {
            "counter_small_delta_loss": self.counter_small_delta_loss,
            "counter_delta_loss_up": self.counter_delta_loss_up,
            "counter_small_delta_acc": self.counter_small_delta_acc,
            "counter_large_delta_acc": self.counter_large_delta_acc,
            "validation_results": self.validation_results,
            "stopped_epoch": self.stopped_epoch,
        }

    def load_state_dict(self, state_dict):
        self.counter_small_delta_loss = state_dict["counter_small_delta_loss"]
        self.counter_delta_loss_up = state_dict["counter_delta_loss_up"]
        self.counter_small_delta_acc = state_dict["counter_small_delta_acc"]
        self.counter_large_delta_acc = state_dict["counter_large_delta_acc"]
        self.validation_results = state_dict["validation_results"]
        self.stopped_epoch = state_dict["stopped_epoch"]
