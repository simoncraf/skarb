import tensorflow as tf
import ray
import pickle
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from gym_example.envs.env_dict import PortfolioManagementDict



class DictSpyModel(TFModelV2):
    capture_index = 0

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, None, model_config, name)
        # Will only feed in sensors->pos.
        input_ = tf.keras.layers.Input(
            shape=self.obs_space["sensors"]["position"].shape
        )

        self.num_outputs = num_outputs or 64
        out = tf.keras.layers.Dense(self.num_outputs)(input_)
        self._main_layer = tf.keras.models.Model([input_], [out])

    def forward(self, input_dict, state, seq_lens):
        def spy(pos, front_cam, task):
            # TF runs this function in an isolated context, so we have to use
            # redis to communicate back to our suite
            ray.experimental.internal_kv._internal_kv_put(
                "d_spy_in_{}".format(DictSpyModel.capture_index),
                pickle.dumps((pos, front_cam, task)),
                overwrite=True,
            )
            DictSpyModel.capture_index += 1
            return np.array(0, dtype=np.int64)

        spy_fn = tf1.py_func(
            spy,
            [
                input_dict["obs"]["sensors"]["position"],
                input_dict["obs"]["sensors"]["front_cam"][0],
                input_dict["obs"]["inner_state"]["job_status"]["task"],
            ],
            tf.int64,
            stateful=True,
        )

        with tf1.control_dependencies([spy_fn]):
            output = self._main_layer([input_dict["obs"]["sensors"]["position"]])

        return output, []



















class KerasTD3Model(TFModelV2):
    """Custom model for DQN."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(KerasTD3Model, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        original_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        self.prices = tf.keras.layers.Input(shape=original_space['prices'].shape, name="prices")
        self.weights = tf.keras.layers.Input(shape=original_space['weights'].shape, name="weights")

        # Concatenating the inputs;
        # One can pass different parts of the state to different networks before concatenation.
        concatenated = tf.keras.layers.Concatenate()([self.prices, self.weights])

        # Building the dense layers
        x = concatenated
        neuron_lst = [64, 32, 16, 8, num_outputs]
        for layer_id, nr_neurons in enumerate(neuron_lst):
            x = tf.keras.layers.Dense(nr_neurons, name="dense_layer_" + str(layer_id), activation=tf.nn.relu,
                                      kernel_initializer=normc_initializer(1.0))(x)

        layer_out = x
        self.model = tf.keras.Model(concatenated, layer_out)
        self.model.summary()

    def forward(self, input_dict, state, seq_lens):
        """Custom core forward method."""
        if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(input_dict[SampleBatch.OBS], self.obs_space, "tf")

        inputs = {'prices': orig_obs["prices"], 'weights': orig_obs["weights"]}
        model_out = self.model(inputs)

        return model_out, state



class MyKerasModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyKerasModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        original_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        self.prices = tf.keras.layers.Input(shape=original_space['prices'].shape, name="prices")
        self.weights = tf.keras.layers.Input(shape=original_space['weights'].shape, name="weights")
        self.inputs = tf.keras.layers.Concatenate()([self.prices, self.weights])
        layer_1 = tf.keras.layers.Dense(
            128,
            name="my_layer1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0),
        )(self.inputs)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(layer_1)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(layer_1)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def metrics(self):
        return {"foo": tf.constant(42.0)}
