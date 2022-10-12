import os
import tensorflow as tf
import numpy as np
from ocifs import OCIFileSystem
import ads

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def read_data(data_folder):
    with np.load(data_folder, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)

def load_data(minist_local,data_bckt):
    data_dir = '/code/data/'
    data_path = os.path.join(data_dir, 'mnist.npz')
    if os.path.exists(minist_local):
        print(f"using pre-fetched dataset from {minist_local}")
        return read_data(minist_local)
    elif data_bckt is not None:
        if not os.path.exists(data_dir):
            create_dir(data_dir)
        print(f"downloading data from {data_bckt}")
        ads.set_auth(os.environ.get("OCI_IAM_TYPE", "resource_principal"))
        authinfo = ads.common.auth.default_signer()
        oci_filesystem = OCIFileSystem(**authinfo)
        oci_filesystem.download(data_bckt, data_dir, recursive=True)
        return read_data(data_path)
    else:
        return tf.keras.datasets.mnist.load_data(data_path)



def mnist_dataset(num_of_workers, worker_id,minist_local,data_bckt):

  (x_train, y_train), _ = load_data(minist_local,data_bckt)
  # The `x` arrays are in uint8 and have values in the range [0, 255].
  # You need to convert them to float32 with values in the range [0, 1]
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000)

  print(f'sharding for worker {worker_id}. Total workers: {num_of_workers}')
  print(f'before sharding dataset size {get_data_set_size(train_dataset)}')
  train_dataset = train_dataset.shard(num_shards=num_of_workers, index=worker_id)
  print(f'after sharding dataset size {get_data_set_size(train_dataset)}')
  return train_dataset

def get_data_set_size(dataset):
    return tf.data.experimental.cardinality(dataset).numpy()

def dataset_fn(num_of_workers, worker_id,global_batch_size, input_context,minist_local=None,data_bckt=None):
  batch_size = input_context.get_per_replica_batch_size(global_batch_size)
  dataset = mnist_dataset(num_of_workers,worker_id,minist_local,data_bckt)
  print(f'sharding for device .')
  print(f'input_context {input_context}')
  print(f'replicas)sync {input_context._num_replicas_in_sync}')
  print(f"before sharding {len(dataset)}")
  dataset = dataset.shard(input_context.num_input_pipelines,
                          input_context.input_pipeline_id)
  print(f"after sharding {len(dataset)}")
  dataset = dataset.batch(batch_size)
  return dataset

def build_cnn_model():
  return tf.keras.Sequential([
      tf.keras.Input(shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
  ])