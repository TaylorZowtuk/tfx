
from typing import Dict, List, Text

import pdb

import os
import absl
import datetime
import glob
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_recommenders as tfrs

from absl import logging
from tfx.types import artifact_utils

from tfx import v1 as tfx
from tfx_bsl.coders import example_coder
from tfx_bsl.public import tfxio

absl.logging.set_verbosity(absl.logging.INFO)

EMBEDDING_DIMENSION = 32
INPUT_FN_BATCH_SIZE = 1


def extract_str_feature(dataset, feature_name):
  np_dataset = []
  for example in dataset:
    np_example = example_coder.ExampleToNumpyDict(example.numpy())
    np_dataset.append(np_example[feature_name][0].decode())
  return tf.data.Dataset.from_tensor_slices(np_dataset)


class MovielensModel(tfrs.Model):

  def __init__(self, user_model, movie_model, tf_transform_output, movies_uri):
    super().__init__()
    self.movie_model: tf.keras.Model = movie_model
    self.user_model: tf.keras.Model = user_model

    movies_artifact = movies_uri.get()[0]
    input_dir = artifact_utils.get_split_uri([movies_artifact], 'train')
    movie_files = glob.glob(os.path.join(input_dir, '*'))
    movies = tf.data.TFRecordDataset(movie_files, compression_type="GZIP")
    movies_dataset = extract_str_feature(movies, 'movie_title')

    loss_metrics = tfrs.metrics.FactorizedTopK(
        candidates=movies_dataset.batch(128).map(movie_model)
        )

    self.task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
        metrics=loss_metrics
        )


  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    try:
      user_embeddings = tf.squeeze(self.user_model(features['user_id']), axis=1)
      # And pick out the movie features and pass them into the movie model,
      # getting embeddings back.
      positive_movie_embeddings = self.movie_model(features['movie_title'])

      # The task computes the loss and the metrics.
      _task = self.task(user_embeddings, positive_movie_embeddings)
    except BaseException as err:
      logging.error('######## ERROR IN compute_loss:\n{}\n###############'.format(err))

    return _task


# This function will apply the same transform operation to training data
# and serving requests.
def _apply_preprocessing(raw_features, tft_layer):
  try:
    transformed_features = tft_layer(raw_features)
  except BaseException as err:
    logging.error('######## ERROR IN _apply_preprocessing:\n{}\n###############'.format(err))

  return transformed_features


def _input_fn(file_pattern: List[Text],
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  try:
    return data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(
          batch_size=batch_size),
      tf_transform_output.transformed_metadata.schema)
  except BaseException as err:
    logging.error('######## ERROR IN _input_fn:\n{}\n###############'.format(err))

  return None


def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example and applies TFT."""
  try:
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
      """Returns the output to be used in the serving signature."""
      try:
        feature_spec = tf_transform_output.raw_feature_spec()
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        result = model(transformed_features)
      except BaseException as err:
        logging.error('######## ERROR IN serve_tf_examples_fn:\n{}\n###############'.format(err))
      return result
  except BaseException as err:
      logging.error('######## ERROR IN _get_serve_tf_examples_fn:\n{}\n###############'.format(err))

  return serve_tf_examples_fn


def _build_user_model(
    tf_transform_output: tft.TFTransformOutput, # Specific to ratings
    embedding_dimension: int = 32) -> tf.keras.Model:
  """Creates a Keras model for the query tower.

  Args:
    tf_transform_output: [tft.TFTransformOutput], the results of Transform
    embedding_dimension: [int], the dimensionality of the embedding space

  Returns:
    A keras Model.
  """
  try:
    unique_user_ids = tf_transform_output.vocabulary_by_name('user_id_vocab')
    users_vocab_str = [b.decode() for b in unique_user_ids]

    model = tf.keras.Sequential(
        [
         tf.keras.layers.StringLookup(
             vocabulary=users_vocab_str, mask_token=None),
         # We add an additional embedding to account for unknown tokens.
         tf.keras.layers.Embedding(len(users_vocab_str) + 1, embedding_dimension)
         ])
  except BaseException as err:
    logging.error('######## ERROR IN _build_user_model:\n{}\n###############'.format(err))

  return model


def _build_movie_model(
    tf_transform_output: tft.TFTransformOutput, # Specific to movies
    embedding_dimension: int = 32) -> tf.keras.Model:
  """Creates a Keras model for the candidate tower.

  Args:
    tf_transform_output: [tft.TFTransformOutput], the results of Transform
    embedding_dimension: [int], the dimensionality of the embedding space

  Returns:
    A keras Model.
  """
  try:
    unique_movie_titles = tf_transform_output.vocabulary_by_name('movie_title_vocab')
    titles_vocab_str = [b.decode() for b in unique_movie_titles]

    model = tf.keras.Sequential(
        [
         tf.keras.layers.StringLookup(
             vocabulary=titles_vocab_str, mask_token=None),
         # We add an additional embedding to account for unknown tokens.
         tf.keras.layers.Embedding(len(titles_vocab_str) + 1, embedding_dimension)
        ])
  except BaseException as err:
      logging.error('######## ERROR IN _build_movie_model:\n{}\n###############'.format(err))
  return model


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  try:
    logging.info(f"Custom config: {fn_args.custom_config}")
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor,
                              tf_transform_output, INPUT_FN_BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor,
                            tf_transform_output, INPUT_FN_BATCH_SIZE)

    model = MovielensModel(
        _build_user_model(tf_transform_output, EMBEDDING_DIMENSION),
        _build_movie_model(tf_transform_output, EMBEDDING_DIMENSION),
        tf_transform_output,
        fn_args.custom_config['movies']
        )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq='batch')

    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
  except BaseException as err:
    logging.error('######## ERROR IN run_fn before fit:\n{}\n###############'.format(err))

  try:
    model.fit(
        train_dataset,
        epochs=fn_args.custom_config['epochs'],
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback])
  except BaseException as err:
      logging.error('######## ERROR IN run_fn during fit:\n{}\n###############'.format(err))

  try:
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)

    movies_artifact = fn_args.custom_config['movies'].get()[0]
    input_dir = artifact_utils.get_split_uri([movies_artifact], 'eval')
    movie_files = glob.glob(os.path.join(input_dir, '*'))
    movies = tf.data.TFRecordDataset(movie_files, compression_type="GZIP")

    movies_dataset = extract_str_feature(movies, 'movie_title')

    index.index_from_dataset(
      tf.data.Dataset.zip((
          movies_dataset.batch(100),
          movies_dataset.batch(100).map(model.movie_model))
      )
    )

    # Run once so that we can get the right signatures into SavedModel
    _, titles = index(tf.constant(["42"]))
    print(f"Recommendations for user 42: {titles[0, :3]}")

    signatures = {
        'serving_default':
            _get_serve_tf_examples_fn(index,
                                      tf_transform_output).get_concrete_function(
                                          tf.TensorSpec(
                                              shape=[None],
                                              dtype=tf.string,
                                              name='examples')),
    }
    index.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)

  except BaseException as err:
      logging.error('######## ERROR IN run_fn during export:\n{}\n###############'.format(err))
