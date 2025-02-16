---
dataset_info:
- config_name: Full
  features:
  - name: id
    dtype: string
  - name: question_id
    dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: image_source
    dtype: string
  - name: image
    dtype: image
  - name: category
    dtype: string
  splits:
  - name: adversarial
    num_bytes: 490408158.0
    num_examples: 3000
  - name: popular
    num_bytes: 490397000.0
    num_examples: 3000
  - name: random
    num_bytes: 490394976.0
    num_examples: 3000
  download_size: 255022914
  dataset_size: 1471200134.0
- config_name: default
  features:
  - name: id
    dtype: string
  - name: question_id
    dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: image_source
    dtype: string
  - name: image
    dtype: image
  - name: category
    dtype: string
  splits:
  - name: test
    num_bytes: 1471200135.0
    num_examples: 9000
  download_size: 255022914
  dataset_size: 1471200135.0
configs:
- config_name: Full
  data_files:
  - split: adversarial
    path: Full/adversarial-*
  - split: popular
    path: Full/popular-*
  - split: random
    path: Full/random-*
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---

<p align="center" width="100%">
<img src="https://i.postimg.cc/g0QRgMVv/WX20240228-113337-2x.png"  width="100%" height="80%">
</p>

# Large-scale Multi-modality Models Evaluation Suite

> Accelerating the development of large-scale multi-modality models (LMMs) with `lmms-eval`

🏠 [Homepage](https://lmms-lab.github.io/) | 📚 [Documentation](docs/README.md) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab)

# This Dataset

This is a formatted version of [POPE](https://github.com/RUCAIBox/POPE). It is used in our `lmms-eval` pipeline to allow for one-click evaluations of large multi-modality models.

```
@article{li2023evaluating,
  title={Evaluating object hallucination in large vision-language models},
  author={Li, Yifan and Du, Yifan and Zhou, Kun and Wang, Jinpeng and Zhao, Wayne Xin and Wen, Ji-Rong},
  journal={arXiv preprint arXiv:2305.10355},
  year={2023}
}
```
