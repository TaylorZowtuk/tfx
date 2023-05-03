# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Define LocalDagRunner to run the pipeline locally."""

import os
import logging

from tfx import v1 as tfx
import pipeline

# Pipeline name will be used to identify this pipeline.
PIPELINE_NAME = "recommender"

# TFX pipeline produces many output files and metadata. All output data will be
# stored under this OUTPUT_DIR.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# TFX produces two types of outputs, files and metadata.
# - Files will be created under PIPELINE_ROOT directory.
# - Metadata will be written to SQLite database in METADATA_PATH.
PIPELINE_ROOT = os.path.join(OUTPUT_DIR, "tfx_pipeline_output", PIPELINE_NAME)
METADATA_PATH = os.path.join(OUTPUT_DIR, "tfx_metadata", PIPELINE_NAME, "metadata.db")

# The last component of the pipeline, "Pusher" will produce serving model under
# SERVING_MODEL_DIR.
SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, "serving_model")

# Specifies the model code directory.
MODELS_PATH = os.path.dirname(os.path.abspath(__file__))


def run():
    """Define a local pipeline."""

    tfx.orchestration.LocalDagRunner().run(
        pipeline.create_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            module_file=os.path.join(MODELS_PATH, "trainer_module.py"),
            serving_model_dir=SERVING_MODEL_DIR,
            metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(
                METADATA_PATH
            ),
        )
    )


def config_logger(level):
    logging.getLogger().setLevel(level)
    logging.basicConfig(
        format="[%(asctime)s %(filename)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %I:%M:%S",
    )


if __name__ == "__main__":
    config_logger(logging.INFO)
    run()
