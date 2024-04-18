import logging
import sys
from dataclasses import dataclass
from typing import List, Union

import numpy as np
from datasets import load_dataset, Dataset, DatasetDict

from templates import *
from utils import temp_seed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_task(task_name):
    aa = task_name.split("__")
    if len(aa) == 2:
        task_group, subtask = aa
    else:
        task_group = aa[0]
        subtask = None
    class_ = getattr(sys.modules[__name__], f"{task_group}Dataset")
    instance = class_(subtask)
    return instance


@dataclass
class Sample:
    id: int = None
    data: dict = None
    correct_candidate: Union[str, List[str]] = None
    candidates: List[str] = None


class Dataset_(Dataset):
    mixed_set = False
    train_sep = "\n\n"
    generation = False  # whether this is a generation task

    def __init__(self, dataset=None, subtask=None, **kwargs) -> None:
        self.samples = None
        self.subtask = subtask
        
    def load_dataset(self, dataset, path, **kwargs):
        return dataset
        
    def get_task_name(self):
        return self.subtask
    
    @staticmethod
    def get_template(template_version=0):
        templates = {0: Template}
        return templates[template_version]

    def build_sample(self, example):
        return

    def sample_train_sets(self, num_train=32, num_dev=None, num_eval=None, num_train_sets=None, seed=None):
        if seed is not None:
            # one train/demo set using the designated seed
            seeds = [seed]
        elif num_train_sets is not None:
            # num_train_sets train/demo sets
            seeds = list(range(num_train_sets))
        else:
            # one train/demo set per evaluation sample
            assert num_dev is None  # not supported
            len_valid_samples = len(self.samples["valid"]) if num_eval is None else num_eval
            with temp_seed(0):
                seeds = np.random.randint(0, 10000, len_valid_samples)

        train_samples = []
        for i, set_seed in enumerate(seeds):
            if self.mixed_set:  # This is always False for now
                raise NotImplementedError
            else:
                if num_dev is not None:
                    train_samples.append(self.sample_subset(data_split="train", seed=set_seed,
                                                            num=num_train + num_dev))  # dev set is included at the end of train set
                    if num_train + num_dev > len(self.samples["train"]):
                        num_train = 0.7 * len(self.samples["train"])
                        num_train = int(num_train) + 1
                        num_dev = len(self.samples["train"]) - num_train
        
                else:
                    train_samples.append(self.sample_subset(data_split="train", seed=set_seed, num=num_train))
                if num_dev is not None:
                    logger.info(f"Sample train set {len(train_samples[-1])}/{len(self.samples['train'])}")
                    logger.info(f"... including dev set {num_dev} samples")
        return train_samples

    def sample_subset(self, data_split="train", seed=0, num=100, exclude=None):
        with temp_seed(seed):
            samples = self.samples[data_split]
            lens = len(samples)
            logger.info(f"Sampling {num} samples from {data_split} set of size {lens}")
            index = np.random.permutation(lens).tolist()[:num if exclude is None else num + 1]
            if exclude is not None and exclude in index:
                index.remove(exclude)
            else:
                index = index[:num]
            return [samples[i] for i in index]

    @property
    def valid_samples(self):
        return self.samples["valid"]


class SST2Dataset(Dataset_):
    train_sep = "\n\n"

    def __init__(self,dataset=None, subtask=None, **kwargs) -> None:

        self.load_dataset(dataset, subtask, **kwargs)

    def load_dataset(self, dataset, path, **kwargs):
        d = super().load_dataset(dataset, path, **kwargs)
        if d is None:
            d = load_dataset('glue', 'sst2')
     
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        label = int(example["label"])
        return Sample(id=example["idx"], data=example, correct_candidate=label, candidates=[0, 1])

    @staticmethod
    def get_template(template_version=0):
        return {0: SST2Template, 1: SST2TemplateEmpty}[template_version]()
    


class CopaDataset(Dataset_):
    train_sep = "\n\n"
    mixed_set = False

    def __init__(self,dataset=None, subtask=None, **kwargs) -> None:
        self.load_dataset(dataset, subtask, **kwargs)

    def load_dataset(self, dataset, path, **kwargs):
        d = super().load_dataset(dataset, path, **kwargs)
        if d is None:
            d = load_dataset('super_glue', 'copa')
        train_examples = d["train"]
        valid_examples = d["validation"]

        train_samples = [self.build_sample(example) for example in train_examples]
        valid_samples = [self.build_sample(example) for example in valid_examples]
        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example):
        sample = \
            Sample(
                id=example["idx"],
                data=example,
                candidates=[example["choice1"], example["choice2"]],
                correct_candidate=example[f"choice{example['label'] + 1}"],
            )

        return sample

    @staticmethod
    def get_template(template_version=0):
        return {0: CopaTemplate, 1: CopaTemplateEmpty}[template_version]()


class BoolQDataset(Dataset_):
    def __init__(self, dataset=None, subtask=None, **kwargs) -> None:
        self.load_dataset(dataset, subtask, **kwargs)

    def load_dataset(self, dataset, path, **kwargs):
        d = super().load_dataset(dataset, path, **kwargs)
        if d is None:
            d = load_dataset("boolq")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=["Yes", "No"],
                correct_candidate="Yes" if example["answer"] else "No",
            )

        return sample
    @staticmethod
    def get_template(template_version=2):
        return {0: BoolQTemplate, 1: BoolQTemplateV2, 2: BoolQTemplateV3}[template_version]()


class MultiRCDataset(Dataset_):

    def __init__(self, dataset=None, subtask=None, **kwargs) -> None:
        self.load_dataset(dataset, subtask, **kwargs)

    def load_dataset(self, dataset, path, **kwargs):
        d = super().load_dataset(dataset, path, **kwargs)
        if d is None:
            d = load_dataset("super_glue", "multirc")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )

        return sample

    @staticmethod
    def get_template(template_version=0):
        return {0: MultiRCTemplate}[template_version]()


class CBDataset(Dataset_):

    def __init__(self, dataset=None, subtask=None, **kwargs) -> None:
        self.load_dataset(dataset, subtask, **kwargs)

    def load_dataset(self,dataset, path, **kwargs):
        d = super().load_dataset(dataset, path, **kwargs)
        if d is None:
            d = load_dataset("super_glue", "cb")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1, 2],
                correct_candidate=example['label']
            )

        return sample

    @staticmethod
    def get_template(template_version=0):
        return {0: CBTemplate}[template_version]()


class WICDataset(Dataset_):

    def __init__(self,dataset=None, subtask=None, **kwargs) -> None:
        self.load_dataset(dataset, subtask, **kwargs)

    def load_dataset(self, dataset, path, **kwargs):
        d = super().load_dataset(dataset, path, **kwargs)
        if d is None:
            d = load_dataset("super_glue", "wic")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )

        return sample

    @staticmethod
    def get_template(template_version=0):
        return {0: WICTemplate}[template_version]()


class WSCDataset(Dataset_):

    def __init__(self, dataset=None, subtask=None, **kwargs) -> None:
        self.load_dataset(dataset, subtask, **kwargs)

    def load_dataset(self,dataset, path, **kwargs):
        d = super().load_dataset(dataset, path, **kwargs)
        if d is None:
            d = load_dataset("super_glue", "wsc.fixed")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )

        return sample
    @staticmethod
    def get_template(template_version=0):
        return {0: WSCTemplate}[template_version]()


class ReCoRDDataset(Dataset_):

    def __init__(self, dataset=None, subtask=None, **kwargs) -> None:
        self.load_dataset(dataset, subtask, **kwargs)

    def load_dataset(self,dataset, path, **kwargs):
        d = super().load_dataset(dataset, path, **kwargs)
        if d is None:
            d = load_dataset("super_glue", "record")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=example['entities'],
                correct_candidate=example['answers']
            )

        return sample

    @staticmethod
    def get_template(template_version=0):
        return {0: ReCoRDTemplateGPT3}[template_version]()


class RTEDataset(Dataset_):

    def __init__(self, dataset=None, subtask=None, **kwargs) -> None:
        self.load_dataset(dataset, subtask, **kwargs)

    def load_dataset(self,dataset, path, **kwargs):
        d = super().load_dataset(dataset, path, **kwargs)
        if d is None:
            d = load_dataset("super_glue", "rte")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )

        return sample

    @staticmethod
    def get_template(template_version=0):
        return {0: RTETemplate, 1: RTETemplateEmpty}[template_version]()


class SQuADDataset(Dataset_):
    metric_name = "f1"
    generation = True

    def __init__(self, dataset=None, subtask=None, **kwargs) -> None:
        self.load_dataset(dataset)

    def load_dataset(self, dataset):
        d = super().load_dataset(dataset, None)
        if d is None:
            d = load_dataset("squad")
        train_examples = d["train"]
        valid_examples = d["validation"]

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)]
        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        answers = example['answers']['text']
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "title": example['title'],
                "context": example['context'],
                "question": example['question'],
                "answers": answers
            },
            candidates=None,
            correct_candidate=answers
        )

    @staticmethod
    def get_template(template_version=0):
        return {0: SQuADv2Template}[template_version]()


class DROPDataset(Dataset_):
    metric_name = "f1"
    generation = True

    def __init__(self, dataset=None, subtask=None, **kwargs) -> None:
        self.load_dataset(dataset)

    def load_dataset(self, dataset):
        d = super().load_dataset(dataset, None)
        if d is None:
            d = load_dataset("drop")
        train_examples = d["train"]
        valid_examples = d["validation"]

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)]
        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        answers = example['answers_spans']['spans']
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "context": example['passage'],
                "question": example['question'],
                "answers": answers
            },
            candidates=None,
            correct_candidate=answers
        )

    @staticmethod
    def get_template(template_version=0):
        return {0: DROPTemplate}[template_version]()


class WinoGrandeDataset(Dataset_):
    def __init__(self, dataset=None, subtask=None, **kwargs) -> None:
        super().__init__(dataset, subtask, **kwargs)
        self.load_dataset(dataset, subtask, **kwargs)

    def load_dataset(self,dataset, path, **kwargs):
        d = super().load_dataset(dataset, path, **kwargs)
        if d is None:
            d = load_dataset('winogrande', 'winogrande_m')
        train_set = d['train']
        valid_set = d['validation']

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        """
        Prompt adapted from https://arxiv.org/pdf/2110.08207.pdf
        """
        sentence = example["sentence"]
        context, target = sentence.split("_")
        sample = Sample(
            data=example,
            candidates=[example['option1'] + target, example['option2'] + target],
            correct_candidate=example[f'option{example["answer"]}'] + target,
        )
        return sample

    @staticmethod
    def get_template(template_version=0):
        if template_version == 0:
            return WinoGrandeTemplate()
        else:
            raise NotImplementedError(f"Template version {template_version} not implemented for WinoGrande")


