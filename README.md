# EDLDNet
Official pytorch implementation of Efficient Dual-line Decoder with Multi-Scale Convolutional Attention for Multi Organ segmentation.

## Architecture
![Overall architecture of EDLDNet](/images/Architecture.png "Overall Architecture of EDLDNet")

## Compare different segmentation model with Dice Score vs MACs count
![Compare model with Dice Score vs MACs count](/images/macs_vs_dice_different_markers.png "Compare different segmentation model with Dice Score vs MACs count")

## Qualitative Results
![Qualitative results for synapse dataset](/images/output_synapse.png "Qualitative results for synapse dataset")

## Quantitative Research
| Method          | MACs ↓ | DICE % ↑  | mIoU % ↑  | Aorta     | GB        | KL        | KR        | Liver     | PC        | SP        | SM        |
| --------------- | ------ | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| UNet            | 54.77G | 70.11     | 59.39     | 84.00     | 56.70     | 72.41     | 62.64     | 86.98     | 48.73     | 81.48     | 67.96     |
| AttnUNet        | 66.67G | 71.70     | 61.38     | 82.61     | 61.94     | 76.07     | 70.42     | 87.54     | 46.70     | 80.67     | 67.66     |
| R50+UNet        | 85.11G | 74.68     | -         | 84.18     | 64.82     | 79.19     | 71.29     | 93.35     | 45.23     | 84.41     | 73.92     |
| R50+AttnUNet    | 91.72G | 75.57     | -         | 85.92     | 63.91     | 79.20     | 72.71     | 93.78     | 45.19     | 84.79     | 74.95     |
| SSFormer        | 17.28G | 78.01     | 67.23     | 86.50     | 70.22     | 78.11     | 73.78     | 93.53     | 61.53     | 87.07     | 76.61     |
| PolypPVT        | 5.3G   | 78.08     | 67.43     | 88.05     | 66.14     | 81.21     | 73.87     | 94.33     | 59.34     | 87.00     | 79.40     |
| TransFuse       | 82.71G | 77.42     | -         | 85.15     | 63.06     | 80.57     | 78.58     | 94.22     | 57.06     | 87.03     | 73.69     |
| TransUNet       | 32.46G | 77.61     | 67.32     | 86.56     | 63.43     | 78.11     | 73.85     | 94.37     | 58.47     | 86.84     | 75.00     |
| DS-TransUNet    | 51.02G | 78.13     | -         | 86.11     | 63.59     | 83.63     | 78.72     | 94.36     | 57.26     | 87.88     | 73.50     |
| SwinUNet        | 6.2G   | 77.58     | 66.88     | 81.76     | 65.95     | 82.32     | 77.22     | 94.35     | 53.81     | 86.84     | 75.79     |
| MT-UNet         | 51.97G | 78.96     | -         | 88.55     | 68.65     | 82.10     | 77.29     | 94.41     | 65.67     | 91.92     | 80.81     |
| UDBANet         | 24.59G | 79.99     | 70.02     | 88.73     | 66.50     | 87.07     | 81.99     | 94.62     | 57.86     | 87.80     | 74.53     |
| PVT-CASCADE     | 8.12G  | 81.06     | 70.88     | 83.01     | 70.59     | 82.23     | 80.37     | 94.08     | 64.43     | 91.31     | 83.52     |
| TransCASCADE    | 22.47G | 82.68     | 73.48     | 88.48     | 68.48     | 87.66     | **84.56** | 94.45     | 65.33     | 90.79     | 83.52     |
| PVT-GCASCADE-B2 | 5.8G   | 83.28     | 73.91     | 86.50     | 71.71     | 87.07     | 83.77     | 95.31     | 66.72     | 90.84     | 83.58     |
| PVT-EMCAD-B2    | 5.6G   | 83.63     | 74.65     | 88.14     | 68.87     | **88.08** | 84.10     | 95.26     | **68.51** | **92.17** | 83.92     |
| EDLDNet (Our)   | 5.6G   | **84.00** | **75.03** | **89.12** | **74.15** | 87.27     | 82.63     | **95.56** | 67.89     | 91.34     | **83.99** |


## Usages
### Recommended Environment

The project is implement with ``Python 3.8`` and ``pytorch 1.11.0+cu113``. To install pytorch you can use:

```commandline
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

Now install the required library using following command:
```commandline
pip install -r requirements.txt
```
### Pretrained model:
You should download the pretrained PVTv2 model from [Google Drive](https://drive.google.com/file/d/13P7CG5efNIDgB4kt5CJiz9mMRkAhuaMW/view?usp=sharing) / [PVT GitHub](https://github.com/whai362/PVT/releases/tag/v2), and then put it in the ``./pretrained_pth/pvt/`` folder for initialization.

### Data preparation:
- **Pre-Processed Synapse Multi-organ dataset:**
Sign up in the [official Synapse website](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480) and download the dataset. Then split the 'RawData' folder into 'TrainSet' (18 scans) and 'TestSet' (12 scans) following the [TransUNet's](https://github.com/Beckschen/TransUNet/blob/main/datasets/README.md) lists and put in the './data/synapse/Abdomen/RawData/' folder. Finally, preprocess using ```python ./utils/preprocess_synapse_data.py```. **OR**
Download the [preprocessed data](https://drive.google.com/file/d/1IVp4lPcB0DSwu-gfKSDBuq-9vG73pR6q/view?usp=sharing) and save in the ```./data/synapse/``` folder. 

### Training:
```commandline
python -W ignore train_synapse.py --root_path /path/to/train/data --volume_path path/to/test/data         # replace --root_path and --volume_path arguments with your actual path
```

### Testing:
Trained Model Weight:

| Dataset | Weight     |
| --------|------------|
| Synapse | [EDLDNet_synapse.pth](https://drive.google.com/file/d/1KVGfK2MKoax1Se20rS_LPeuNT66Sws-N/view?usp=sharing) |

Download the weight or use your trained model weight to test
```commandline
python -W ignore test_synapse.py --saved_model_path /path/to/best/model/weight      # replace --saved_model_path argument with actual path
```
After completion of testing, check the ``test_log`` directory to get the test result.