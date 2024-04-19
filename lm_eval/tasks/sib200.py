import transformers.data.metrics.squad_metrics as squad_metrics
from lm_eval.api.metric import mean
from lm_eval.api.task import PromptSourceTask


class Sib200(PromptSourceTask):
    VERSION = 1
    DATASET_PATH = "sib200"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["test"]
        # return self.dataset["validation"]

