
import tensorflow as tf
import tensorflow_transform as tft
import pdb

NUM_OOV_BUCKETS = 1

def preprocessing_fn(inputs):
  # We only want the user ID and the movie title, but we also need vocabularies
  # for both of them.  The vocabularies aren't features, they're only used by
  # the lookup.
  outputs = {}
  outputs['user_id'] = tft.sparse_tensor_to_dense_with_shape(inputs['user_id'], [None, 1], '-1')
  outputs['movie_title'] = tft.sparse_tensor_to_dense_with_shape(inputs['movie_title'], [None, 1], '-1')

  tft.compute_and_apply_vocabulary(
      inputs['user_id'],
      num_oov_buckets=NUM_OOV_BUCKETS,
      vocab_filename='user_id_vocab')

  tft.compute_and_apply_vocabulary(
      inputs['movie_title'],
      num_oov_buckets=NUM_OOV_BUCKETS,
      vocab_filename='movie_title_vocab')

  return outputs
