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
"""
This file defines TFX pipeline and various components in the pipeline.
"""

import os
import json

from typing import Any, Dict, Optional

import tensorflow as tf
import tensorflow_datasets as tfds
import apache_beam as beam

from tfx.components.example_gen.base_example_gen_executor import BaseExampleGenExecutor
from tfx.components.example_gen.component import FileBasedExampleGen
from tfx.components.example_gen import utils
from tfx.dsl.components.base import executor_spec
from ml_metadata.proto import metadata_store_pb2

from tfx import v1 as tfx


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _TFDatasetToExample(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline,
    exec_properties: Dict[str, Any],
    split_pattern: str
    ) -> beam.pvalue.PCollection:
    """Read a TensorFlow Dataset and create tf.Examples"""
    custom_config = json.loads(exec_properties['custom_config'])
    dataset_name = custom_config['dataset']
    split_name = custom_config['split']

    builder = tfds.builder(dataset_name)
    builder.download_and_prepare()

    return (pipeline
            | 'MakeExamples' >> tfds.beam.ReadFromTFDS(builder, split=split_name)
            | 'AsNumpy' >> beam.Map(tfds.as_numpy)
            | 'ToDict' >> beam.Map(dict)
            | 'ToTFExample' >> beam.Map(utils.dict_to_example)
            )

class TFDSExecutor(BaseExampleGenExecutor):
  def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
    """Returns PTransform for TF Dataset to TF examples."""
    return _TFDatasetToExample

def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    module_file: str,
    serving_model_dir: str,
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
) -> tfx.dsl.Pipeline:
    """Implements a product recommender pipeline with TFX."""

    components = []

    # Ratings data.
    ratings_example_gen = FileBasedExampleGen(
        input_base='dummy',
        custom_config={'dataset':'movielens/100k-ratings', 'split':'train'},
        custom_executor_spec=executor_spec.BeamExecutorSpec(TFDSExecutor)).with_id('ratings_example_gen')
    components.append(ratings_example_gen)
    # Features of all the available movies.
    movies_example_gen = FileBasedExampleGen(
        input_base='dummy',
        custom_config={'dataset':'movielens/100k-movies', 'split':'train'},
        custom_executor_spec=executor_spec.BeamExecutorSpec(TFDSExecutor)).with_id('movies_example_gen')
    components.append(movies_example_gen)

    movies_stats_gen = tfx.components.StatisticsGen(
        examples=movies_example_gen.outputs['examples']).with_id('movies_stats_gen')
    components.append(movies_stats_gen)
    ratings_stats_gen = tfx.components.StatisticsGen(
        examples=ratings_example_gen.outputs['examples']).with_id('ratings_stats_gen')
    components.append(ratings_stats_gen)

    movies_schema_gen = tfx.components.SchemaGen(
        statistics=movies_stats_gen.outputs['statistics'],
        infer_feature_shape=False).with_id('movies_schema_gen')
    ratings_schema_gen = tfx.components.SchemaGen(
        statistics=ratings_stats_gen.outputs['statistics'],
        infer_feature_shape=False).with_id('ratings_schema_gen')
    components.append(movies_schema_gen)
    components.append(ratings_schema_gen)

    movies_transform = tfx.components.Transform(
        examples=movies_example_gen.outputs['examples'],
        schema=movies_schema_gen.outputs['schema'],
        module_file=os.path.abspath('movies_transform_module.py')).with_id('movies_transform')
    components.append(movies_transform)
    ratings_transform = tfx.components.Transform(
        examples=ratings_example_gen.outputs['examples'],
        schema=ratings_schema_gen.outputs['schema'],
        module_file=os.path.abspath('ratings_transform_module.py')).with_id('ratings_transform')
    components.append(ratings_transform)

    trainer = tfx.components.Trainer(
        module_file=os.path.abspath(module_file),
        examples=ratings_transform.outputs['transformed_examples'],
        transform_graph=ratings_transform.outputs['transform_graph'],
        schema=ratings_transform.outputs['post_transform_schema'],
        train_args=tfx.proto.TrainArgs(num_steps=500),
        eval_args=tfx.proto.EvalArgs(num_steps=10),
        custom_config={
            'epochs':3,
            'movies':movies_transform.outputs['transformed_examples'],
            'movie_schema':movies_transform.outputs['post_transform_schema'],
            'ratings':ratings_transform.outputs['transformed_examples'],
            'ratings_schema':ratings_transform.outputs['post_transform_schema']
            }).with_id('trainer')
    components.append(trainer)

    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir))).with_id('pusher')
    components.append(pusher)

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        metadata_connection_config=metadata_connection_config,
    )
