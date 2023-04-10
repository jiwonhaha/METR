# Git Repository

You can find all of the code used for this project from [Gitlab](https://git.cs.bham.ac.uk/projects-2022-23/jxp042) or [Github](https://github.com/jiwonhaha/finalProject).

# Installation

Opera relies on several basic packages such as MMCV, MMDetection, etc, so you need to install these packages at first.

Before Installation, clone git repository of mmcv and mmdetection in third_party folder.   
After that, you can run codes below.


1. Install `mmcv`

   ```bash
   cd /ROOT/finalProject/third_party/mmcv
   MMCV_WITH_OPS=1 pip install -e .
   ```

2. Install `mmdet`

   ```bash
   cd /ROOT/finalProject/third_party/mmdetection
   pip install -e .
   ```

3. Install `opera`

   ```bash
   cd /ROOT/finalProject
   pip install -r requirements.txt
   pip install -e .
   ```



## Requirements

- Linux
- Python 3.7+
- PyTorch 1.8+
- CUDA 10.1+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)
- [MMDetection](https://mmdetection.readthedocs.io/en/latest/#installation)

## Getting Started

### Download dataset

- [Open Monkey Challenge Dataset](https://competitions.codalab.org/competitions/34342#learn_the_details)

### Download annotation file which is converted to COCO keypoint format

- [Training Annotation File](https://drive.google.com/file/d/1-8c652RrCyKI0mAor_KPlI_WQ8EMQPYV/view?usp=share_link) 
        
- [Validation Annotation File](https://drive.google.com/file/d/1DZcVRiXCpGsbrfZz9ABFUFs68PNSfgx8/view?usp=share_link) 

- [Test Annotation File](https://drive.google.com/file/d/1AiU1KqqPLhGWDnKPvfNiTQuSY4hBUsYt/view?usp=share_link) 
         

### Download checkpoint

- [Checkpoint with 12 epochs](https://drive.google.com/file/d/1OwBYLV7y5illjyWfspIq6u76iS0CP568/view?usp=share_link)

Put dataset, and both annotation files to directory.'monkey_dataset/.'       

       
### Inference       
Run this code   

``` bash
python3 demo/image_demo.py --out-file (output filename you want to product) (file directory you want to inference) configs/petr/petr_r50_monkey_coco.py (checkpoint directory)
```

### Training        
Run this code        

``` bash
bash tools/dist_train.sh configs/petr/petr_r50_monkey_coco.py 1 --work-dir monkeyDir --gpu-id 0 --resume-from (checkpoint directory which is start point)
```

### Evaluation about Average Precision (AP)

Run this code, it will give you outcome respect to mAP, APL, AP50, AP75.   

``` bash
bash tools/dist_test.sh configs/petr/petr_r50_monkey_coco.py (checkpoint directory) 1 --eval keypoints
```

### Evaluation about Probability of Correct Keypoint (PCK)

After running Evaluation code for average precision, you can obtain PredictedTest.json.

or download already [generated result file](https://drive.google.com/file/d/1W-qtnWe2NGpxMymXJvCQEFFZ8TsD1wAJ/view?usp=share_link)

After finishing one of approach, you can get PCK @ 0.2 result for each keypoint by compiling code below.

``` bash
python3 pck@0.2.py
```

### Evaluation about Average Precision (AP) With Different OKS Loss Threshold

You can manually modify the threshold in averagePrecision.py file and compile:
``` bash
python3 averagePrecision.py
```


## Acknowledgement

This project is an open source project built upon [Opera (PETR)](https://github.com/hikvision-research/opera) and [OpenMMLab](https://github.com/open-mmlab/). Thanks to all the contributors of Opera and OpenMMLab.


## License

This project is released under the [Apache 2.0 license](LICENSE).

