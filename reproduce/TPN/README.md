# TPN Reproduction

## Introduction

| Name:   | [TPN](https://arxiv.org/abs/1805.10002)                    |
| ------- | ---------------------------------------------------------- |
| Embed.: | Conv64F                                                    |
| Type:   | Metric                                                     |
| Venue:  | ICLR2019                                                   |
| Codes:  | [**TPN-pytorch**](https://github.com/csyanbin/TPN-pytorch) |

Cite this work with (template):

```bibtex
@misc{liu2019learningpropagatelabelstransductive,
      title={Learning to Propagate Labels: Transductive Propagation Network for Few-shot Learning}, 
      author={Yanbin Liu and Juho Lee and Minseop Park and Saehoon Kim and Eunho Yang and Sung Ju Hwang and Yi Yang},
      year={2019},
      eprint={1805.10002},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1805.10002}, 
}
```

---

## Results and Models

All the results are tested under the best model. Checkpoints of different epochs are also provided.

| dataset/task   | 5way-1shot                                                   | 5way-5shot                                                   | 10way-1shot                                                  | 10-way-5shot                                                 |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| miniImageNet   | 54.06±0.38 [:arrow_down:](https://drive.google.com/drive/folders/1Y14e-h_DcwfyxwU71GZIXQ2G39ZS8oys) | 69.27±0.30 [:arrow_down:](https://drive.google.com/drive/folders/1DIBmJ8a_GZIlEmUW0KEf_awTaB-8GB7i) | 37.36±0.23 [:arrow_down:](https://drive.google.com/drive/folders/1OeO3K7wY4y-UN979eRUQo1vvVvin3vF-) | 53.62±0.20 [:arrow_down:](https://drive.google.com/drive/folders/1c8yd0rMQhAytePcrfLq2nEYaxPDFHc8f) |
| tieredImageNet | 53.36±0.42  [:arrow_down:](https://drive.google.com/drive/folders/1_C0VA1LirJ5l3kYqEBY8HfHmXOzCkZog) | 69.83±0.35  [:arrow_down:](https://drive.google.com/drive/folders/1anwG8tjvaXQ5oq9BBjcTQjY9OdKmDYf1) | 40.29±0.28  [:arrow_down:](https://drive.google.com/drive/folders/1HnJfDHHE78YzaGvhmHHvaG4ymVhdPkrh) | 57.53±0.25  [:arrow_down:](https://drive.google.com/drive/folders/1NBaxyY60rJIxnoeOV0UAAt6KORPvZyfJ) |

