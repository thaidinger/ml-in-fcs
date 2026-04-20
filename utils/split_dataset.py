########################
# Train-test set split #
########################

def split_train_test_set(dataset, split_ratio=0.8):
  n_samples = len(dataset)
  train_size = int(n_samples * split_ratio)
  train_set = dataset[:train_size]
  test_set = dataset[train_size:]
  return train_set, test_set









