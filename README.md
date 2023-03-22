## Introduction

**O**bject **Per**ception & **A**pplication (Opera) is a unified toolbox for multiple computer vision tasks: detection, segmentation, pose estimation, etc.

To date, Opera implements the following algorithms:

- [PETR (CVPR'2022 Oral)](configs/petr)
- [SOIT (AAAI'2022)](configs/soit)
- [InsPose (ACM MM'2021)](configs/inspose)

# Installation

Opera relies on several basic packages such as MMCV, MMDetection, etc, so you need to install these packages at first.

1. Install `mmcv`

   ```bash
   cd /ROOT/Opera/third_party/mmcv
   MMCV_WITH_OPS=1 pip install -e .
   ```

2. Install `mmdet`

   ```bash
   cd /ROOT/Opera/third_party/mmdetection
   pip install -e .
   ```

3. Install `opera`

   ```bash
   cd /ROOT/Opera
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

Please see [get_started.md](docs/get_started.md) for the basic usage of Opera.

## Acknowledgement

Opera is an open source project built upon [OpenMMLab](https://github.com/open-mmlab/). We appreciate all the contributors who implement this flexible and efficient toolkits.

## Citations

If you find our works useful in your research, please consider citing:
```BibTeX
@inproceedings{shi2022end,
  title={End-to-End Multi-Person Pose Estimation With Transformers},
  author={Shi, Dahu and Wei, Xing and Li, Liangqi and Ren, Ye and Tan, Wenming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11069--11078},
  year={2022}
}

@inproceedings{yu2022soit,
  title={SOIT: Segmenting Objects with Instance-Aware Transformers},
  author={Yu, Xiaodong and Shi, Dahu and Wei, Xing and Ren, Ye and Ye, Tingqun and Tan, Wenming},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  pages={3188--3196},
  year={2022}
}

@inproceedings{shi2021inspose,
  title={Inspose: instance-aware networks for single-stage multi-person pose estimation},
  author={Shi, Dahu and Wei, Xing and Yu, Xiaodong and Tan, Wenming and Ren, Ye and Pu, Shiliang},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={3079--3087},
  year={2021}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

