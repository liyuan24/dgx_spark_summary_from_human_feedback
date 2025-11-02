# coding=utf-8
# Copyright 2022 the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import datasets
import json


_CITATION = """\
@inproceedings{stienon2020learning,
  author = {Nisan Stiennon and Long Ouyang and Jeff Wu and Daniel M. Ziegler and Ryan Lowe and Chelsea Voss and Alec Radford and Dario Amodei and Paul Christiano},
  title = {Learning to summarize from human feedback},
  booktitle = {NeurIPS},
  year = 2020,
}
"""

_URL = "https://openaipublic.blob.core.windows.net/summarize-from-feedback/dataset"

_DESCRIPTION = """\
Summarize from Feedback contains the human feedback data released by the "Learning to summarize from human feedback" paper.
"""


class SummarizeFromFeedbackConfig(datasets.BuilderConfig):
    """BuilderConfig for Summarize from Feedback."""

    def __init__(self, features, **kwargs):
        """BuilderConfig for Summarize from Feedback.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SummarizeFromFeedbackConfig, self).__init__(**kwargs)
        self.features = features


class SummarizeFromFeedback(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = SummarizeFromFeedbackConfig

    BUILDER_CONFIGS = [
        SummarizeFromFeedbackConfig(
            name="comparisons",
            features=datasets.Features(
                {
                    "info": {
                        "id": datasets.Value("string"),
                        "post": datasets.Value("string"),
                        "title": datasets.Value("string"),
                        "subreddit": datasets.Value("string"),
                        "site": datasets.Value("string"),
                        "article": datasets.Value("string"),
                    },
                    "summaries": [
                        {
                            "text": datasets.Value("string"),
                            "policy": datasets.Value("string"),
                            "note": datasets.Value("string"),
                        },
                    ],
                    "choice": datasets.Value("int32"),
                    "worker": datasets.Value("string"),
                    "batch": datasets.Value("string"),
                    "split": datasets.Value("string"),
                    "extra": {"confidence": datasets.Value("int32")},
                }
            ),
        ),
        SummarizeFromFeedbackConfig(
            name="axis",
            features=datasets.Features(
                {
                    "info": {
                        "id": datasets.Value("string"),
                        "post": datasets.Value("string"),
                        "title": datasets.Value("string"),
                        "subreddit": datasets.Value("string"),
                        "site": datasets.Value("string"),
                        "article": datasets.Value("string"),
                    },
                    "summary": {
                        "text": datasets.Value("string"),
                        "policy": datasets.Value("string"),
                        "note": datasets.Value("string"),
                        "axes": {
                            "overall": datasets.Value("int32"),
                            "accuracy": datasets.Value("int32"),
                            "coverage": datasets.Value("int32"),
                            "coherence": datasets.Value("int32"),
                            "compatible": datasets.Value("bool"),
                        },
                    },
                    "worker": datasets.Value("string"),
                    "batch": datasets.Value("string"),
                    "split": datasets.Value("string"),
                }
            ),
        ),
    ]

    IMAGE_EXTENSION = ".png"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=self.config.features,
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        comparison_batch_files = [
            "batch3.json",
            "batch4.json",
            "batch5.json",
            "batch10.json",
            "batch11.json",
            "batch12.json",
            "batch13.json",
            "batch14.json",
            "batch15.json",
            "batch16.json",
            "batch17.json",
            "batch18.json",
            "batch19.json",
            "batch20.json",
            "batch22.json",
            "batch6.json",
            "batch7.json",
            "batch8.json",
            "batch9.json",
            "batch0_cnndm.json",
            "cnndm0.json",
            "cnndm2.json",
            "edit_b2_eval_test.json",
        ]

        axis_batch_files = [
            "cnndm1.json",
            "cnndm3.json",
            "cnndm4.json",
            "tldraxis1.json",
            "tldraxis2.json",
        ]

        if self.config.name == "axis":
            downloaded_files = dl_manager.download_and_extract(
                [
                    os.path.join(_URL, "axis_evals", batch_file)
                    for batch_file in axis_batch_files
                ]
            )

            examples = []
            for file in downloaded_files:
                examples += [
                    json.loads(comparisons_json)
                    for comparisons_json in open(file).readlines()
                ]

            test_examples = []
            valid_examples = []
            for example in examples:
                if example["split"] == "test":
                    test_examples.append(example)
                elif example["split"] in ("valid1", "valid2"):
                    valid_examples.append(example)
                else:
                    raise ValueError(
                        f"{example['split']} is an unrecognized dataset split."
                    )

            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST, gen_kwargs={"raw_examples": test_examples}
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"raw_examples": valid_examples},
                ),
            ]

        elif self.config.name == "comparisons":
            downloaded_files = dl_manager.download_and_extract(
                [
                    os.path.join(_URL, "comparisons", batch_file)
                    for batch_file in comparison_batch_files
                ]
            )

            examples = []
            for file in downloaded_files:
                examples += [
                    json.loads(comparisons_json)
                    for comparisons_json in open(file).readlines()
                ]

            train_examples = []
            valid_examples = []
            for example in examples:
                if example["split"] == "train":
                    train_examples.append(example)
                elif example["split"] in ("valid1", "valid2"):
                    valid_examples.append(example)
                else:
                    raise ValueError(
                        f"{example['split']} is an unrecognized dataset split."
                    )

            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"raw_examples": train_examples},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"raw_examples": valid_examples},
                ),
            ]

        else:
            raise ValueError(
                "Unrecognized config name. Options are axis and comparisons"
            )

    def _generate_examples(self, raw_examples, no_labels=False):
        """Yields examples."""
        id_ = 0
        for example in raw_examples:

            if self.config.name == "comparisons":
                if "confidence" not in example["extra"]:
                    example["extra"]["confidence"] = None

                if "id" not in example["info"]:
                    example["info"]["id"] = None

            elif self.config.name == "axis":
                if "overall" not in example["summary"]["axes"]:
                    example["summary"]["axes"]["overall"] = None

                if "accuracy" not in example["summary"]["axes"]:
                    example["summary"]["axes"]["accuracy"] = None

                if "coherence" not in example["summary"]["axes"]:
                    example["summary"]["axes"]["coherence"] = None

                if "coverage" not in example["summary"]["axes"]:
                    example["summary"]["axes"]["coverage"] = None

                if "compatible" not in example["summary"]["axes"]:
                    example["summary"]["axes"]["compatible"] = None
            else:
                raise ValueError(
                    "Unrecognized config name. Options are axis and comparisons"
                )

            if "article" not in example["info"]:
                example["info"]["article"] = None

            if "site" not in example["info"]:
                example["info"]["site"] = None

            if "subreddit" not in example["info"]:
                example["info"]["subreddit"] = None

            if "post" not in example["info"]:
                example["info"]["post"] = None

            id_ += 1
            yield id_, example


if __name__ == "__main__":
    summary_from_human_feedback = SummarizeFromFeedback(config_name="comparisons")
    summary_from_human_feedback.download_and_prepare()
    train_dataset = summary_from_human_feedback.as_dataset(split=datasets.Split.TRAIN)
    validation_dataset = summary_from_human_feedback.as_dataset(
        split=datasets.Split.VALIDATION
    )
    print(type(train_dataset))
    print(len(train_dataset))
    print(len(validation_dataset))
