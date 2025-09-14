# EDLDNet
Official pytorch implementation of Efficient Dual-line Decoder with Multi-Scale Convolutional Attention for Multi Organ segmentation.

[DOI](https://doi.org/10.1016/j.bspc.2025.108611) | [PDF](https://authors.elsevier.com/c/1lmTx_djyJXjq2)

## Architecture
![Overall architecture of EDLDNet](/images/Architecture.png "Overall Architecture of EDLDNet")

*Fig: Overall architecture of EDLDNet*

## Compare different segmentation model with Dice Score vs MACs count
![Compare model with Dice Score vs MACs count](/images/macs_vs_dice_different_markers.png "Compare different segmentation model with Dice Score vs MACs count")

*Fig: Dice score vs MACs count for different segmentation methods over synapse dataset.*

## Qualitative Results
![Qualitative results for synapse dataset](/images/output_synapse.png "Qualitative results for synapse dataset")

*Fig: The comparison of contoured segmentation images from Synapse dataset among proposed method and the competitive existing methods.*


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

### Cite this work
```
@article{HASSAN2026108611,
    title = {An efficient dual-line decoder network with multi-scale convolutional attention for multi-organ segmentation},
    journal = {Biomedical Signal Processing and Control},
    volume = {112},
    pages = {108611},
    year = {2026},
    issn = {1746-8094},
    doi = {https://doi.org/10.1016/j.bspc.2025.108611},
    url = {https://www.sciencedirect.com/science/article/pii/S174680942501122X},
    author = {Riad Hassan and M. Rubaiyat Hossain Mondal and Sheikh Iqbal Ahamed and Fahad Mostafa and Md Mostafijur Rahman}
}
```
