# TemDeep
PyTorch implementation of TemDeep: A Self-Supervised Framework for Temporal Downscaling of 

Atmospheric Fields at Arbitrary Time Resolutions

### Training TemDeep:
Simply run the following to train an encoder-decoder network using TemDeep on the your dataset:
```
python main.py 
```

### Distributed Training
With distributed data parallel (DDP) training:
```
CUDA_VISIBLE_DEVICES=0 python main.py --nodes 2 --nr 0
CUDA_VISIBLE_DEVICES=1 python main.py --nodes 2 --nr 1
CUDA_VISIBLE_DEVICES=2 python main.py --nodes 2 --nr 2
CUDA_VISIBLE_DEVICES=N python main.py --nodes 2 --nr 3
```

## What is TemDeep?
TemDeep is short for "Temporal Downscaling based on Deep learning". Numerical forecast products with high temporal resolution are crucial tools in atmospheric studies, allowing for accurate identification of rapid transitions and subtle changes that may be missed by lower-resolution data. However, the acquisition of high-resolution data is limited due to excessive computational demands and substantial storage needs in numerical models. Current deep learning methods for statistical downscaling still require massive ground truth with high temporal resolution for model training. In this paper, we present a self-supervised framework for downscaling atmospheric variables at arbitrary time resolutions by imposing a temporal coherence constraint. Firstly, we construct an encoder-decoder structured temporal downscaling network, and then pretrain this downscaling network on a subset of data that exhibits rapid transitions and is filtered out based on a composite index. Subsequently, this pretrained network is utilized to downscale the fields from adjacent time periods and generate the field at the middle time point. By leveraging the temporal coherence inherent in meteorological variables, the network is further trained based on the difference between the generated field and the actual middle field. To track the evolving trends in meteorological system movements, a flow estimation module is designed to assist with generating interpolated fields. Results show that our method can accurately recover evolution details superior to other methods, reaching 53.7% in the restoration rate on the test set. In addition, to avoid generating outlier values and guide the model out of local optima, two regularization terms are integrated into the loss function to enforce spatial and temporal continuity, which further improves the performance by 7.6%.

<p align="center">
  <img src="./pics/framework.jpg" width="1000"/>
</p>



## Usage
Simply run for single GPU or CPU training:
```
python main.py
```

For distributed training (DDP), use for every process in nodes, in which N is the GPU number you would like to dedicate the process to:
```
CUDA_VISIBLE_DEVICES=0 python main.py --nodes 2 --nr 0
CUDA_VISIBLE_DEVICES=1 python main.py --nodes 2 --nr 1
CUDA_VISIBLE_DEVICES=2 python main.py --nodes 2 --nr 2
CUDA_VISIBLE_DEVICES=N python main.py --nodes 2 --nr 3
```

`--nr` corresponds to the process number of the N nodes we make available for training.

### Testing
To test a trained model, make sure to set the `model_path` variable in the `config/config.yaml` to the log ID of the training (e.g. `logs/0`).
Set the `epoch_num` to the epoch number you want to load the checkpoints from (e.g. `4000`).

```
python eval.py
```

or in place:
```
python eval.py --model_path=./save --epoch_num=4000
```

## Logging and TensorBoard
To view results in TensorBoard, run:
```
tensorboard --logdir runs
```

## Environment

  - Python >= 3.6
  - PyTorch, tested on 1.9, but should be fine when >=1.6

## Citation

If you find our code or datasets helpful, please consider citing our related works.

## Contact

If you have questions or suggestions, please open an issue here or send an email to public_wlw@163.com.
