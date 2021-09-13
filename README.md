# Recurrent Implicit Quantile Networks

This repository provides the implementation of the baseline for out-of-distribution detection in 
RL benchmarks. Here is the code to the benchmark: 
[https://github.com/modanesh/anomalous_rl_envs](https://github.com/modanesh/anomalous_rl_envs).
It contains two sets of environments, one is derived from [OpenAI Gym control task](https://github.com/openai/gym) and the other
from [PyBullet3](https://github.com/bulletphysics/bullet3). 

If you ever used this repo in your work, please cite it with:
```
@inproceedings{danesh2021oodd,
  title={Out-of-Distribution Dynamics Detection: RL-Relevant Benchmarks and Results},
  author={Danesh, Mohamad H and Fern, Alan},
  booktitle={International Conference on Machine Learning, Uncertainty & Robustness in Deep Learning Workshop},
  journal={},
  year={2021}
}
```

## Installation
- python 3.6+
- To install dependencies:
```commandline
pip install -r requirements.txt
```

## Usage
The two files: `autoregressive_control.py` and `autoregressive_pybullet.py` are quite similar in structure and functionality.
Their main difference is their targeted environments. `autoregressive_control.py` works with OpenAI Gym control tasks, while
`autoregressive_pybullet.py` works with the Bullet physics simulations.

In the following, the usage of `autoregressive_control.py` is provided. However, the same would apply for the other case.

### Parameters
```commandline
usage: autoregressive_control.py [-h] [--predictive_model_training]
                                 [--predictive_model_testing]
                                 [--anomaly_detection]
                                 [--horizon_comparison_as]
                                 [--samplesize_comparison_as]
                                 [--avgvsmax_comparison_as]
                                 [--dataset_analysis] [--dists_cdf]
                                 [--detection_delay] [--is_recurrent]
                                 [--is_recurrent_v2] [--feature_part_analysis]
                                 [--scheduled_sampling_training]
                                 [--predictive_model_paths PREDICTIVE_MODEL_PATHS [PREDICTIVE_MODEL_PATHS ...]]
                                 [--batch_size BATCH_SIZE] [--lr LR]
                                 [--gru_units GRU_UNITS]
                                 [--num_quantile_sample NUM_QUANTILE_SAMPLE]
                                 [--policy_num_quantile_sample POLICY_NUM_QUANTILE_SAMPLE]
                                 [--num_tau_sample NUM_TAU_SAMPLE]
                                 [--quantile_embedding_dim QUANTILE_EMBEDDING_DIM]
                                 [--policy_quantile_embedding_dim POLICY_QUANTILE_EMBEDDING_DIM]
                                 [--test_interval TEST_INTERVAL]
                                 [--num_iterations NUM_ITERATIONS]
                                 [--env_name ENV_NAME] [--data_path DATA_PATH]
                                 [--test_data_path TEST_DATA_PATH]
                                 [--noisy_data_path NOISY_DATA_PATH]
                                 [--anomaly_inserted ANOMALY_INSERTED]
                                 [--clip_value CLIP_VALUE]
                                 [--horizons HORIZONS [HORIZONS ...]]
                                 [--sampling_sizes SAMPLING_SIZES [SAMPLING_SIZES ...]]
                                 [--given_fpr GIVEN_FPR]
                                 [--decay_type {linear,exponential}]

optional arguments:
  -h, --help            show this help message and exit
  --predictive_model_training
                        To train autoregressive models
  --predictive_model_testing
                        To test autoregressive models
  --anomaly_detection   Do the AD when anomalies injected into the system
  --horizon_comparison_as
                        Studying the affect of horizon on anomaly scores and
                        AUCs
  --samplesize_comparison_as
                        Studying the affect of sampling size on anomaly scores
                        and AUCs
  --avgvsmax_comparison_as
                        Studying the affect of combining anomaly scores using
                        avg vs. max on AUCs
  --dataset_analysis    Analyzing dataset
  --dists_cdf           Studying CDFs of internal distributions
  --detection_delay     Measuring the delay in detecting anomalies
  --is_recurrent        Determines whether the model has memory or not
  --is_recurrent_v2     Determines whether the model has memory or not -- v2
                        RNN model
  --feature_part_analysis
                        Analyzing feature participation is calculating anomaly
                        scores
  --scheduled_sampling_training
                        To train autoregressive models using scheduled
                        sampling
  --predictive_model_paths PREDICTIVE_MODEL_PATHS [PREDICTIVE_MODEL_PATHS ...]
                        Path to all predictive models
  --batch_size BATCH_SIZE
                        Batch size
  --lr LR               Learning rate
  --gru_units GRU_UNITS
                        Number of cells in the GRU
  --num_quantile_sample NUM_QUANTILE_SAMPLE
                        Number of quantile samples for IQN
  --policy_num_quantile_sample POLICY_NUM_QUANTILE_SAMPLE
                        Number of quantile samples for policy IQN
  --num_tau_sample NUM_TAU_SAMPLE
                        Number of tau samples for IQN, sets the distribution
                        sampling size.
  --quantile_embedding_dim QUANTILE_EMBEDDING_DIM
                        Qunatiles embedding dimension in IQN
  --policy_quantile_embedding_dim POLICY_QUANTILE_EMBEDDING_DIM
                        Qunatiles embedding dimension in policy IQN
  --test_interval TEST_INTERVAL
                        Intervals between train and test
  --num_iterations NUM_ITERATIONS
                        Number of iterations to update model
  --env_name ENV_NAME   Name of the main environment: to train, test, update
                        models, find threshold, and calculate performance on
                        normal envs
  --data_path DATA_PATH
                        path to the dataset json file
  --test_data_path TEST_DATA_PATH
                        path to the test dataset json file
  --noisy_data_path NOISY_DATA_PATH
                        path to the test dataset json file
  --anomaly_inserted ANOMALY_INSERTED
                        Time when the anomaly is inserted into the system
  --clip_value CLIP_VALUE
                        Clipping gradients
  --horizons HORIZONS [HORIZONS ...]
                        Horizon to go forward in time
  --sampling_sizes SAMPLING_SIZES [SAMPLING_SIZES ...]
                        Size of the sampling to build the tree of
                        distributions at time t
  --given_fpr GIVEN_FPR
                        Acceptable FPR rate to calculate the threshold for
                        anomaly detection delay
  --decay_type {linear,exponential}
                        How to decay epsilon in Scheduled sampling
```

### How To Run
First, you need to generate a dataset of nominal trajectories by the following command:
```commandline
python autoregressive_control.py --test_policy --env_name Acrobot-v1
```

To train the RIQN predictor:
```commandline
python autoregressive_control.py --predictive_model_training --env_name Acrobot-v1 --is_recurrent_v2 --predictive_model_paths "SOME PATHS"
```

To test the RIQN predictor:
```commandline
python autoregressive_control.py --predictive_model_testing --env_name Acrobot-v1 --predictive_model_paths "SOME PATHS" --is_recurrent_v2 --horizons 1 --anomaly_inserted 0
```

To detect anomalies using the RIQN predictor:
```commandline
python autoregressive_control.py --anomaly_detection --anomaly_inserted 20 --horizons 1 10 --sampling_sizes 4 8 32 128 --is_recurrent_v2 --num_iterations 5 --env_name AcrobotMod-v4 --predictive_model_paths "SOME PATHS"
```
