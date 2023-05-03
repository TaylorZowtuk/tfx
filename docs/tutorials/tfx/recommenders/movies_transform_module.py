
import tensorflow as tf
import tensorflow_transform as tft

def preprocessing_fn(inputs):
  # We only want the movie title
  return {'movie_title':inputs['movie_title']}
