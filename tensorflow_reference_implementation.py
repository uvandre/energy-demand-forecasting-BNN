from idna import valid_contextj
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

mpl.rcParams['agg.path.chunksize'] = 100000

# Source:
# https://towardsdatascience.com/predict-daily-electric-consumption-with-neural-networks-8ba59471c1d

def data_transform(data, timesteps, var='x'):
  m = []
  s = data.to_numpy()

  for i in range(s.shape[0]-timesteps):
      m.append(s[i:i+timesteps].tolist())

  if var == 'x':
      t = np.zeros((len(m), len(m[0]), len(m[0][0])))
      for i, x in enumerate(m):
          for j, y in enumerate(x):
              for k, z in enumerate(y):
                  t[i, j, k] = z
  else:
      t = np.zeros((len(m), len(m[0])))
      for i, x in enumerate(m):
          for j, y in enumerate(x):
              t[i, j] = np.array([y])

  return t

df = pd.read_csv('DA_price_v_actual_load.csv', index_col=0)
df = df.drop(df[df['DA Price'] <= 0].index)
df = df[:-44]

train_len = int(len(df)*0.7)

train_df = df[:train_len-1000-6]
val_df = df[train_len-1000-6:train_len+1000]
test_df = df[train_len+1000:-17]

val_X = df[['hour', 'day', 'DA Price']]
val_y = df[['MW Load']]

all_X = train_df[['hour', 'day', 'DA Price']]
all_y = train_df[['MW Load']]

test_X = test_df[['hour', 'day', 'DA Price']]
test_y = test_df[['MW Load']]

HOURS_AHEAD = 24

all_y_rnn = data_transform(all_y, HOURS_AHEAD, var='y')
all_X_rnn = data_transform(all_X, HOURS_AHEAD, var='x')

val_X_rnn = data_transform(val_X, HOURS_AHEAD, var='x')
val_y_rnn = data_transform(val_y, HOURS_AHEAD, var='y') 

test_X_rnn = data_transform(test_X, HOURS_AHEAD, var='x')
test_y_rnn = data_transform(test_y, HOURS_AHEAD, var='y')

s = all_X.shape[1]
print(s)


model = tf.keras.Sequential()
model.add(layers.Dense(s, activation=tf.nn.relu, input_shape=(HOURS_AHEAD, all_X.shape[1])))
model.add(layers.Dense(s, activation=tf.nn.relu))
model.add(layers.Dense(s, activation=tf.nn.relu))
model.add(layers.Dense(s, activation=tf.nn.relu))
model.add(layers.Dense(s, activation=tf.nn.relu))
model.add(layers.Flatten())
model.add(layers.Dense(all_X.shape[1]*HOURS_AHEAD//2, activation=tf.nn.relu))
model.add(layers.Dense(HOURS_AHEAD))

nadam = tf.keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=nadam, loss='mape')
epochs = 500
batch_size = 2400
history = model.fit(
    all_X_rnn,
    all_y_rnn,
    batch_size=batch_size,
    epochs=epochs,
    validation_data = (val_X_rnn, val_y_rnn),
)

model.save('new/final/models_reference')

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(test_X_rnn, test_y_rnn, batch_size=128)
print(f'Metric types used: {model.metrics_names}')
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(test_X_rnn)
print("predictions shape:", predictions.shape)

#predictions = predictions.flatten()
x_test = test_X_rnn[:,:,-1].flatten()
y_test = test_y_rnn.flatten()

fin_test_X = np.array([test_X['DA Price']])[0]
test_y = test_y.to_numpy().flatten()

pred_list = []
for vals in predictions:
    pred_list.append(vals[0])

final_test_df = pd.DataFrame({'price':fin_test_X.flatten()[:-24], 'mw_load':test_y.flatten()[:-24]})
final_pred_df = pd.DataFrame({'price':fin_test_X.flatten()[:-24], 'mw_load':pred_list})

plt.scatter(final_test_df['price'], final_test_df['mw_load'], c='g', s=2)
plt.scatter(final_pred_df['price'], final_pred_df['mw_load'], c='r', s=2)

plt.title('Test set results')
plt.savefig(f'new/final/fig_reference_tf/test_{epochs}epochs_{batch_size}batchsize.png')