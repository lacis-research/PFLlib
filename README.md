# <img src="docs/imgs/logo-green.png" alt="icon" height="24" style="vertical-align:sub;"/> PFLlib: Personalized Federated Learning Library and Benchmark

🎯*We built a beginner-friendly federated learning (FL) library and benchmark: **master FL in 2 hours—run it on your PC!** [Contribute](#easy-to-extend) your algorithms, datasets, and metrics to grow the FL community.*

👏 The **[official website](http://www.pfllib.com)** and **[leaderboard](http://www.pfllib.com/benchmark.html)** is live! Our methods—[FedCP](https://github.com/TsingZ0/FedCP), [GPFL](https://github.com/TsingZ0/GPFL), and [FedDBE](https://github.com/TsingZ0/DBE)—lead the way. Notably, **FedDBE** stands out with robust performance across varying data heterogeneity levels.

[![JMLR](https://img.shields.io/badge/JMLR-Published-blue)](https://www.jmlr.org/papers/v26/23-1634.html)
[![arXiv](https://img.shields.io/badge/arXiv-2312.04992-b31b1b.svg)](https://arxiv.org/abs/2312.04992)
![Apache License 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)


![](docs/imgs/structure.png)
Figure 1: An Example for FedAvg. You can create a scenario using `generate_DATA.py` and run an algorithm using `main.py`, `clientNAME.py`, and `serverNAME.py`. For a new algorithm, you only need to add new features in `clientNAME.py` and `serverNAME.py`.

🎯**If you find our repository useful, please cite the corresponding paper:**

```
@article{zhang2025pfllib,
  title={PFLlib: A Beginner-Friendly and Comprehensive Personalized Federated Learning Library and Benchmark},
  author={Zhang, Jianqing and Liu, Yang and Hua, Yang and Wang, Hao and Song, Tao and Xue, Zhengui and Ma, Ruhui and Cao, Jian},
  journal={Journal of Machine Learning Research},
  volume={26},
  number={50},
  pages={1--10},
  year={2025}
}

@inproceedings{Zhang2025htfllib,
  author={Zhang, Jianqing and Wu, Xinghao and Zhou, Yanbing and Sun, Xiaoting and Cai, Qiqi and Liu, Yang and Hua, Yang and Zheng, Zhenzhe and Cao, Jian and Yang, Qiang},
  title = {HtFLlib: A Comprehensive Heterogeneous Federated Learning Library and Benchmark},
  year = {2025},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining}
}
```
### FedLoad e HERAFL

Adaptações nativas dos algoritmos de `mininetfed_iwcmc`, usando servidores, clientes, modelos e dados do PFLlib: [execução, perfis de recursos e limitações](docs/heterogeneous_algorithms.md).

### Key Features

- **39 traditional FL ([tFL](#traditional-fl-tfl)) and personalized FL ([pFL](#personalized-fl-pfl)) algorithms, 3 scenarios, and 24 datasets.**

- Real-machine deployment: [HtFL-OnDevice](https://github.com/TsingZ0//HtFL-OnDevice).

- Some **experimental results** are avalible in its [paper](https://arxiv.org/abs/2312.04992) and [here](#experimental-results). 

- Refer to [examples](#how-to-start-simulating-examples-for-fedavg) to learn how to use it.

- Refer to [easy to extend](#easy-to-extend) to learn how to add new data or algorithms.

- The benchmark platform can simulate scenarios using the 4-layer CNN on Cifar100 for **500 clients** on **one NVIDIA GeForce RTX 3090 GPU card** with only **5.08GB GPU memory** cost.

- We provide [privacy evaluation](#privacy-evaluation) and [systematical research support](#systematical-research-support). 

- You can now train on some clients and evaluate performance on new clients by setting `args.num_new_clients` in `./system/main.py`. Please note that not all tFL/pFL algorithms support this feature.

- PFLlib primarily focuses on data (statistical) heterogeneity. For algorithms and a benchmark platform that address **both data and model heterogeneity**, please refer to our extended project **[Heterogeneous Federated Learning (HtFLlib)](https://github.com/TsingZ0/HtFLlib)**.

- As we strive to meet diverse user demands, frequent updates to the project may alter default settings and scenario creation codes, affecting experimental results.
  
- [Closed issues](https://github.com/TsingZ0/PFLlib/issues?q=is%3Aissue+is%3Aclosed) may help you a lot when errors arise.

- When submitting pull requests, please provide sufficient *instructions* and *examples* in the comment box. 

The origin of the **data heterogeneity** phenomenon is the characteristics of users, who generate non-IID (not Independent and Identically Distributed) and unbalanced data. With data heterogeneity existing in the FL scenario, a myriad of approaches have been proposed to crack this hard nut. In contrast, the personalized FL (pFL) may take advantage of the statistically heterogeneous data to learn the personalized model for each user. 


## Algorithms with code (updating)

> ### Traditional FL (tFL)

  ***Basic tFL***

- **FedAvg** — [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a.html) *AISTATS 2017*

  ***Update-correction-based tFL***

- **SCAFFOLD** - [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](http://proceedings.mlr.press/v119/karimireddy20a.html) *ICML 2020*

  ***Regularization-based tFL***

- **FedProx** — [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) *MLsys 2020*
- **FedDyn** — [Federated Learning Based on Dynamic Regularization](https://openreview.net/forum?id=B7v4QMR6Z9w) *ICLR 2021*

  ***Model-splitting-based tFL***

- **MOON** — [Model-Contrastive Federated Learning](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Model-Contrastive_Federated_Learning_CVPR_2021_paper.html) *CVPR 2021*
- **FedLC** — [Federated Learning With Label Distribution Skew via Logits Calibration](https://proceedings.mlr.press/v162/zhang22p.html) *ICML 2022*

  ***Knowledge-distillation-based tFL***

- **FedGen** — [Data-Free Knowledge Distillation for Heterogeneous Federated Learning](http://proceedings.mlr.press/v139/zhu21b.html) *ICML 2021*
- **FedNTD** — [Preservation of the Global Knowledge by Not-True Distillation in Federated Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/fadec8f2e65f181d777507d1df69b92f-Abstract-Conference.html) *NeurIPS 2022*

  ***Heuristically-search-based tFL***
  
- **FedCross** - [FedCross: Towards Accurate Federated Learning via Multi-Model Cross-Aggregation](https://www.computer.org/csdl/proceedings-article/icde/2024/171500c137/1YOuaPcHF3q) *ICDE 2024*

> ### Personalized FL (pFL)

  ***Meta-learning-based pFL***

- **Per-FedAvg** — [Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach](https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html) *NeurIPS 2020*

  ***Regularization-based pFL***
  
- **pFedMe** — [Personalized Federated Learning with Moreau Envelopes](https://papers.nips.cc/paper/2020/hash/f4f1f13c8289ac1b1ee0ff176b56fc60-Abstract.html) *NeurIPS 2020*
- **Ditto** — [Ditto: Fair and robust federated learning through personalization](https://proceedings.mlr.press/v139/li21h.html) *ICML 2021*

  ***Personalized-aggregation-based pFL***

- **APFL** — [Adaptive Personalized Federated Learning](https://arxiv.org/abs/2003.13461) *2020* 
- **FedFomo** — [Personalized Federated Learning with First Order Model Optimization](https://openreview.net/forum?id=ehJqJQk9cw) *ICLR 2021*
- **FedAMP** — [Personalized Cross-Silo Federated Learning on non-IID Data](https://ojs.aaai.org/index.php/AAAI/article/view/16960) *AAAI 2021*
- **FedPHP** — [FedPHP: Federated Personalization with Inherited Private Models](https://link.springer.com/chapter/10.1007/978-3-030-86486-6_36) *ECML PKDD 2021*
- **APPLE** — [Adapt to Adaptation: Learning Personalization for Cross-Silo Federated Learning](https://www.ijcai.org/proceedings/2022/301) *IJCAI 2022*
- **FedALA** — [FedALA: Adaptive Local Aggregation for Personalized Federated Learning](https://ojs.aaai.org/index.php/AAAI/article/view/26330) *AAAI 2023* 

  ***Model-splitting-based pFL***

- **FedPer** — [Federated Learning with Personalization Layers](https://arxiv.org/abs/1912.00818) *2019*
- **LG-FedAvg** — [Think Locally, Act Globally: Federated Learning with Local and Global Representations](https://arxiv.org/abs/2001.01523) *2020*
- **FedRep** — [Exploiting Shared Representations for Personalized Federated Learning](http://proceedings.mlr.press/v139/collins21a.html) *ICML 2021*
- **FedRoD** — [On Bridging Generic and Personalized Federated Learning for Image Classification](https://openreview.net/forum?id=I1hQbx10Kxn) *ICLR 2022*
- **FedBABU** — [Fedbabu: Towards enhanced representation for federated image classification](https://openreview.net/forum?id=HuaYQfggn5u) *ICLR 2022*
- **FedGC** — [Federated Learning for Face Recognition with Gradient Correction](https://ojs.aaai.org/index.php/AAAI/article/view/20095/19854) *AAAI 2022*
- **FedCP** — [FedCP: Separating Feature Information for Personalized Federated Learning via Conditional Policy](https://arxiv.org/pdf/2307.01217v2.pdf) *KDD 2023*
- **GPFL** — [GPFL: Simultaneously Learning Generic and Personalized Feature Information for Personalized Federated Learning](https://arxiv.org/pdf/2308.10279v3.pdf) *ICCV 2023*
- **FedGH** — [FedGH: Heterogeneous Federated Learning with Generalized Global Header](https://dl.acm.org/doi/10.1145/3581783.3611781) *ACM MM 2023*
- **FedDBE** — [Eliminating Domain Bias for Federated Learning in Representation Space](https://openreview.net/forum?id=nO5i1XdUS0) *NeurIPS 2023*
- **FedCAC** — [Bold but Cautious: Unlocking the Potential of Personalized Federated Learning through Cautiously Aggressive Collaboration](https://arxiv.org/abs/2309.11103) *ICCV 2023*
- **PFL-DA** — [Personalized Federated Learning via Domain Adaptation with an Application to Distributed 3D Printing](https://www.tandfonline.com/doi/full/10.1080/00401706.2022.2157882) *Technometrics 2023*
- **FedAS** — [FedAS: Bridging Inconsistency in Personalized Federated Learning](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_FedAS_Bridging_Inconsistency_in_Personalized_Federated_Learning_CVPR_2024_paper.pdf) *CVPR 2024*

  ***Knowledge-distillation-based pFL (more in [HtFLlib](https://github.com/TsingZ0/HtFLlib))***

- **FD (FedDistill)** — [Communication-Efficient On-Device Machine Learning: Federated Distillation and Augmentation under Non-IID Private Data](https://arxiv.org/pdf/1811.11479.pdf) *2018*
- **FML** — [Federated Mutual Learning](https://arxiv.org/abs/2006.16765) *2020*
- **FedKD** — [Communication-efficient federated learning via knowledge distillation](https://www.nature.com/articles/s41467-022-29763-x) *Nature Communications 2022*
- **FedProto** — [FedProto: Federated Prototype Learning across Heterogeneous Clients](https://ojs.aaai.org/index.php/AAAI/article/view/20819) *AAAI 2022*
- **FedPCL (w/o pre-trained models)** — [Federated learning from pre-trained models: A contrastive learning approach](https://proceedings.neurips.cc/paper_files/paper/2022/file/7aa320d2b4b8f6400b18f6f77b6c1535-Paper-Conference.pdf) *NeurIPS 2022* 
- **FedPAC** — [Personalized Federated Learning with Feature Alignment and Classifier Collaboration](https://openreview.net/pdf?id=SXZr8aDKia) *ICLR 2023*

  ***Other pFL***

- **FedMTL (not MOCHA)** — [Federated multi-task learning](https://papers.nips.cc/paper/2017/hash/6211080fa89981f66b1a0c9d55c61d0f-Abstract.html) *NeurIPS 2017*
- **FedBN** — [FedBN: Federated Learning on non-IID Features via Local Batch Normalization](https://openreview.net/forum?id=6YEQUn0QICG) *ICLR 2021*

## Datasets and scenarios (updating)

We support 3 types of scenarios with various datasets and move the common dataset splitting code into `./dataset/utils` for easy extension. If you need another data set, just write another code to download it and then use the [utils](https://github.com/TsingZ0/PFLlib/tree/master/dataset/utils).

### ***label skew*** scenario

For the ***label skew*** scenario, we introduce **16** famous datasets: 

- **MNIST**
- **EMNIST**
- **FEMNIST**
- **Fashion-MNIST**
- **Cifar10**
- **Cifar100**
- **AG News**
- **Sogou News**
- **Tiny-ImageNet**
- **Country211**
- **Flowers102**
- **GTSRB**
- **Shakespeare**
- **Stanford Cars**
- **COVIDx**
- **kvasir**

The datasets can be easily split into **IID** and **non-IID** versions. In the **non-IID** scenario, we distinguish between two types of distribution:

1. **Pathological non-IID**: In this case, each client only holds a subset of the labels, for example, just 2 out of 10 labels from the MNIST dataset, even though the overall dataset contains all 10 labels. This leads to a highly skewed distribution of data across clients.

2. **Practical non-IID**: Here, we model the data distribution using a Dirichlet distribution, which results in a more realistic and less extreme imbalance. For more details on this, refer to this [paper](https://proceedings.neurips.cc/paper/2020/hash/18df51b97ccd68128e994804f3eccc87-Abstract.html).

Additionally, we offer a `balance` option, where data amount is evenly distributed across all clients.

### ***feature shift*** scenario

For the ***feature shift*** scenario, we utilize **3** widely used datasets in Domain Adaptation: 
- **Amazon Review** (raw data can be fetched from [this link](https://drive.google.com/file/d/1QbXFENNyqor1IlCpRRFtOluI2_hMEd1W/view?usp=sharing))
- **Digit5** (raw data available [here](https://drive.google.com/file/d/1sO2PisChNPVT0CnOvIgGJkxdEosCwMUb/view))
- **DomainNet**

### ***real-world*** scenario

For the ***real-world*** scenario, we introduce **5** naturally separated datasets:  
- **Camelyon17** (5 hospitals, 2 labels)  
- **iWildCam** (194 camera traps, 158 labels)  
- **Omniglot** (20 clients, 50 labels)  
- **HAR (Human Activity Recognition)** (30 clients, 6 labels)  
- **PAMAP2** (9 clients, 12 labels)  

For more details on datasets and FL algorithms in **IoT**, please refer to [FL-IoT](https://github.com/TsingZ0/FL-IoT).

### Examples for **MNIST** in the ***label skew*** scenario
```bash
cd ./dataset
# Please modify train_ratio and alpha in dataset\utils\dataset_utils.py

python generate_MNIST.py iid - - # for iid and unbalanced scenario
python generate_MNIST.py iid balance - # for iid and balanced scenario
python generate_MNIST.py noniid - pat # for pathological noniid and unbalanced scenario
python generate_MNIST.py noniid - dir # for practical noniid and unbalanced scenario
python generate_MNIST.py noniid - exdir # for Extended Dirichlet strategy 
```

The command line output of running `python generate_MNIST.py noniid - dir`
```bash
Number of classes: 10
Client 0         Size of data: 2630      Labels:  [0 1 4 5 7 8 9]
                 Samples of labels:  [(0, 140), (1, 890), (4, 1), (5, 319), (7, 29), (8, 1067), (9, 184)]
--------------------------------------------------
Client 1         Size of data: 499       Labels:  [0 2 5 6 8 9]
                 Samples of labels:  [(0, 5), (2, 27), (5, 19), (6, 335), (8, 6), (9, 107)]
--------------------------------------------------
Client 2         Size of data: 1630      Labels:  [0 3 6 9]
                 Samples of labels:  [(0, 3), (3, 143), (6, 1461), (9, 23)]
--------------------------------------------------
```
<details>
    <summary>Show more</summary>

    Client 3         Size of data: 2541      Labels:  [0 4 7 8]
                     Samples of labels:  [(0, 155), (4, 1), (7, 2381), (8, 4)]
    --------------------------------------------------
    Client 4         Size of data: 1917      Labels:  [0 1 3 5 6 8 9]
                     Samples of labels:  [(0, 71), (1, 13), (3, 207), (5, 1129), (6, 6), (8, 40), (9, 451)]
    --------------------------------------------------
    Client 5         Size of data: 6189      Labels:  [1 3 4 8 9]
                     Samples of labels:  [(1, 38), (3, 1), (4, 39), (8, 25), (9, 6086)]
    --------------------------------------------------
    Client 6         Size of data: 1256      Labels:  [1 2 3 6 8 9]
                     Samples of labels:  [(1, 873), (2, 176), (3, 46), (6, 42), (8, 13), (9, 106)]
    --------------------------------------------------
    Client 7         Size of data: 1269      Labels:  [1 2 3 5 7 8]
                     Samples of labels:  [(1, 21), (2, 5), (3, 11), (5, 787), (7, 4), (8, 441)]
    --------------------------------------------------
    Client 8         Size of data: 3600      Labels:  [0 1]
                     Samples of labels:  [(0, 1), (1, 3599)]
    --------------------------------------------------
    Client 9         Size of data: 4006      Labels:  [0 1 2 4 6]
                     Samples of labels:  [(0, 633), (1, 1997), (2, 89), (4, 519), (6, 768)]
    --------------------------------------------------
    Client 10        Size of data: 3116      Labels:  [0 1 2 3 4 5]
                     Samples of labels:  [(0, 920), (1, 2), (2, 1450), (3, 513), (4, 134), (5, 97)]
    --------------------------------------------------
    Client 11        Size of data: 3772      Labels:  [2 3 5]
                     Samples of labels:  [(2, 159), (3, 3055), (5, 558)]
    --------------------------------------------------
    Client 12        Size of data: 3613      Labels:  [0 1 2 5]
                     Samples of labels:  [(0, 8), (1, 180), (2, 3277), (5, 148)]
    --------------------------------------------------
    Client 13        Size of data: 2134      Labels:  [1 2 4 5 7]
                     Samples of labels:  [(1, 237), (2, 343), (4, 6), (5, 453), (7, 1095)]
    --------------------------------------------------
    Client 14        Size of data: 5730      Labels:  [5 7]
                     Samples of labels:  [(5, 2719), (7, 3011)]
    --------------------------------------------------
    Client 15        Size of data: 5448      Labels:  [0 3 5 6 7 8]
                     Samples of labels:  [(0, 31), (3, 1785), (5, 16), (6, 4), (7, 756), (8, 2856)]
    --------------------------------------------------
    Client 16        Size of data: 3628      Labels:  [0]
                     Samples of labels:  [(0, 3628)]
    --------------------------------------------------
    Client 17        Size of data: 5653      Labels:  [1 2 3 4 5 7 8]
                     Samples of labels:  [(1, 26), (2, 1463), (3, 1379), (4, 335), (5, 60), (7, 17), (8, 2373)]
    --------------------------------------------------
    Client 18        Size of data: 5266      Labels:  [0 5 6]
                     Samples of labels:  [(0, 998), (5, 8), (6, 4260)]
    --------------------------------------------------
    Client 19        Size of data: 6103      Labels:  [0 1 2 3 4 9]
                     Samples of labels:  [(0, 310), (1, 1), (2, 1), (3, 1), (4, 5789), (9, 1)]
    --------------------------------------------------
    Total number of samples: 70000
    The number of train samples: [1972, 374, 1222, 1905, 1437, 4641, 942, 951, 2700, 3004, 2337, 2829, 2709, 1600, 4297, 4086, 2721, 4239, 3949, 4577]
    The number of test samples: [658, 125, 408, 636, 480, 1548, 314, 318, 900, 1002, 779, 943, 904, 534, 1433, 1362, 907, 1414, 1317, 1526]

    Saving to disk.

    Finish generating dataset.
</details>

## Models
- for MNIST and Fashion-MNIST

    1. Mclr_Logistic(1\*28\*28) # convex
    2. LeNet()
    3. DNN(1\*28\*28, 100)

- for Cifar10, Cifar100 and Tiny-ImageNet

    1. Mclr_Logistic(3\*32\*32) # convex
    2. FedAvgCNN()
    3. DNN(3\*32\*32, 100)
    4. ResNet18, AlexNet, MobileNet, GoogleNet, etc.

- for AG_News and Sogou_News

    - LSTM()
    - fastText() in [Bag of Tricks for Efficient Text Classification](https://aclanthology.org/E17-2068/) 
    - TextCNN() in [Convolutional Neural Networks for Sentence Classification](https://aclanthology.org/D14-1181/)
    - TransformerModel() in [Attention is all you need](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)

- for AmazonReview

    - AmazonMLP() in [Curriculum manager for source selection in multi-source domain adaptation](https://link.springer.com/chapter/10.1007/978-3-030-58568-6_36)

- for Omniglot

    - FedAvgCNN()

- for HAR and PAMAP

    - HARCNN() in [Convolutional neural networks for human activity recognition using mobile sensors](https://eudl.eu/pdf/10.4108/icst.mobicase.2014.257786)

## Environments
Install [CUDA](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html). 

Install [conda latest](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) and activate conda. 

For additional configurations, refer to the `prepare.sh` script.  

```bash
conda env create -f env_cuda_latest.yaml  # Downgrade torch via pip if needed to match the CUDA version
```

## How to start simulating (examples for FedAvg)

- Download [this project](https://github.com/TsingZ0/PFLlib) to an appropriate location using [git](https://git-scm.com/).
    ```bash
    git clone https://github.com/TsingZ0/PFLlib.git
    ```

- Create proper environments (see [Environments](#environments)).

- Build evaluation scenarios (see [Datasets and scenarios (updating)](#datasets-and-scenarios-updating)).

- Run evaluation: 
    ```bash
    cd ./system
    python main.py -data MNIST -m CNN -algo FedAvg -gr 2000 -did 0 # using the MNIST dataset, the FedAvg algorithm, and the 4-layer CNN model
    python main.py -data MNIST -m CNN -algo FedAvg -gr 2000 -did 0,1,2,3 # running on multiple GPUs
    ```

**Note**: It is preferable to tune algorithm-specific hyper-parameters before using any algorithm on a new machine. 

## Easy to extend

This library is designed to be easily extendable with new algorithms and datasets. Here’s how you can add them:

- **New Dataset**: To add a new dataset, simply create a `generate_DATA.py` file in `./dataset` and then write the download code and use the [utils](https://github.com/TsingZ0/PFLlib/tree/master/dataset/utils) as shown in `./dataset/generate_MNIST.py` (you can consider it as a template):
  ```python
  # `generate_DATA.py`
  import necessary pkgs
  from utils import necessary processing funcs

  def generate_dataset(...):
    # download dataset as usual
    # pre-process dataset as usual
    X, y, statistic = separate_data((dataset_content, dataset_label), ...)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, statistic, ...)

  # call the generate_dataset func
  ```
  
- **New Algorithm**: To add a new algorithm, extend the base classes **Server** and **Client**, which are defined in `./system/flcore/servers/serverbase.py` and `./system/flcore/clients/clientbase.py`, respectively.
  - Server
    ```python
    # serverNAME.py
    import necessary pkgs
    from flcore.clients.clientNAME import clientNAME
    from flcore.servers.serverbase import Server

    class NAME(Server):
        def __init__(self, args, times):
            super().__init__(args, times)

            # select slow clients
            self.set_slow_clients()
            self.set_clients(clientAVG)
        def train(self):
            # server scheduling code of your algorithm
    ```
  - Client
    ```python
    # clientNAME.py
    import necessary pkgs
    from flcore.clients.clientbase import Client

    class clientNAME(Client):
        def __init__(self, args, id, train_samples, test_samples, **kwargs):
            super().__init__(args, id, train_samples, test_samples, **kwargs)
            # add specific initialization
        
        def train(self):
            # client training code of your algorithm
    ```
  
- **New Model**: To add a new model, simply include it in `./system/flcore/trainmodel/models.py`.
  
- **New Optimizer**: If you need a new optimizer for training, add it to `./system/flcore/optimizers/fedoptimizer.py`.
  
- **New Benchmark Platform or Library**: Our framework is flexible, allowing users to build custom platforms or libraries for specific applications, such as [FL-IoT](https://github.com/TsingZ0/FL-IoT) and [HtFLlib](https://github.com/TsingZ0/HtFLlib).


## Privacy Evaluation

You can use the following privacy evaluation methods to assess the privacy-preserving capabilities of tFL/pFL algorithms in PFLlib. Please refer to `./system/flcore/servers/serveravg.py` for an example. Note that most of these evaluations are not typically considered in the original papers. _We encourage you to add more attacks and metrics for privacy evaluation._ 

### Currently supported attacks:
- [DLG (Deep Leakage from Gradients)](https://www.ijcai.org/proceedings/2022/0324.pdf) attack

### Currently supported metrics:
- **PSNR (Peak Signal-to-Noise Ratio)**: an objective metric for image evaluation, defined as the logarithm of the ratio of the squared maximum value of RGB image fluctuations to the Mean Squared Error (MSE) between two images. A lower PSNR score indicates better privacy-preserving capabilities.


## Systematical research support

To simulate Federated Learning (FL) under practical conditions, such as **client dropout**, **slow trainers**, **slow senders**, and **network TTL (Time-To-Live)**, you can adjust the following parameters:

- `-cdr`: Dropout rate for clients. Clients are randomly dropped at each training round based on this rate.
- `-tsr` and `-ssr`: Slow trainer and slow sender rates, respectively. These parameters define the proportion of clients that will behave as slow trainers or slow senders. Once a client is selected as a "slow trainer" or "slow sender," it will consistently train/send slower than other clients.
- `-tth`: Threshold for network TTL in milliseconds.

Thanks to [@Stonesjtu](https://github.com/Stonesjtu/pytorch_memlab/blob/d590c489236ee25d157ff60ecd18433e8f9acbe3/pytorch_memlab/mem_reporter.py#L185), this library can also record the **GPU memory usage** for the model. 

## Experimental Results

If you're interested in **experimental results (e.g., accuracy)** for the algorithms mentioned above, you can find results in our accepted FL papers, which also utilize this library. These papers include:

- [FedALA](https://github.com/TsingZ0/FedALA)
- [FedCP](https://github.com/TsingZ0/FedCP)
- [GPFL](https://github.com/TsingZ0/GPFL)
- [DBE](https://github.com/TsingZ0/DBE)

Please note that while these results were based on this library, **reproducing the exact results may be challenging** as some settings might have changed in response to community feedback. For example, in earlier versions, we set `shuffle=False` in `clientbase.py`.

Here are the relevant papers for your reference:

```
@inproceedings{zhang2023fedala,
  title={Fedala: Adaptive local aggregation for personalized federated learning},
  author={Zhang, Jianqing and Hua, Yang and Wang, Hao and Song, Tao and Xue, Zhengui and Ma, Ruhui and Guan, Haibing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={9},
  pages={11237--11244},
  year={2023}
}

@inproceedings{Zhang2023fedcp,
  author = {Zhang, Jianqing and Hua, Yang and Wang, Hao and Song, Tao and Xue, Zhengui and Ma, Ruhui and Guan, Haibing},
  title = {FedCP: Separating Feature Information for Personalized Federated Learning via Conditional Policy},
  year = {2023},
  booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining}
}

@inproceedings{zhang2023gpfl,
  title={GPFL: Simultaneously Learning Global and Personalized Feature Information for Personalized Federated Learning},
  author={Zhang, Jianqing and Hua, Yang and Wang, Hao and Song, Tao and Xue, Zhengui and Ma, Ruhui and Cao, Jian and Guan, Haibing},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5041--5051},
  year={2023}
}

@inproceedings{zhang2023eliminating,
  title={Eliminating Domain Bias for Federated Learning in Representation Space},
  author={Jianqing Zhang and Yang Hua and Jian Cao and Hao Wang and Tao Song and Zhengui XUE and Ruhui Ma and Haibing Guan},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
  url={https://openreview.net/forum?id=nO5i1XdUS0}
}
```

## FedLoad e HeraFL com os datasets nativos do PFLlib

FedLoad e HERAFL usam os geradores existentes em `dataset/`, como
`generate_MNIST.py`, `generate_Cifar10.py` e `generate_Cifar100.py`.
IID/non-IID, balanceamento, distribuição de rótulos, normalização e separação
entre treino e teste seguem as regras nativas do framework.

Exemplo a partir de `PFLlib/`:

```bash
cd dataset
python generate_MNIST.py noniid balance dir
cd ../system
python main.py -data MNIST -algo FedLoad -dev cpu
python main.py -data MNIST -algo HERAFL -dev cpu
```

Para CIFAR-100:

```bash
cd dataset
python generate_Cifar100.py noniid balance dir
cd ../system
python main.py -data Cifar100 -algo HERAFL -m CNN -dev cpu
```

Para executar HeraFL, FedLoad e FedAvg e gerar a comparação em PNG, use `start.sh`
a partir de `PFLlib/`. O script ativa automaticamente o ambiente Conda `pfllib`:

```bash
./start.sh
DATASET=cifar100 NUM_CLIENTS=20 DEVICE=cuda ./start.sh
NUM_ROUNDS=1 LOCAL_EPOCHS=1 DEVICE=cpu ./start.sh
ALGORITHMS=HERAFL STRAGGLER_PREDICTOR=false ./start.sh
```

O `start.sh` usa o setup do artigo FedLoad: CIFAR-10/VGG11, IID, 30 clientes,
6 participantes por rodada (`SELECTION_FRACTION=0.2`), 5 épocas locais como
adaptação das 5 iterações locais do artigo, 10 braços `linspace(0.5, 1, 10)`,
alpha 0.5, beta 0.75, janela de largura de banda 10 e pesos 0.6/0.1/0.3.
O preset `paper` usa capacidades de rede em {75, 25, 5} Mbps, throughput
uniforme entre 1 e a capacidade, frequência entre 1 e 2, ciclos por amostra
entre 5 e 30 e potências computacionais {7, 4, 3} W e de transmissão
{2, 1.8, 1.7} W. As potências são sorteadas independentemente.
As 200 rodadas, batch 64, learning rate 0.0001 para VGG e seed 42 são escolhas da
implementação; esses valores não são especificados no setup do artigo.
CINIC-10 seleciona VGG16, mas não possui gerador nativo disponível neste PFLlib;
CIFAR-100/VGG16 é uma extensão, sem equivalência com o dataset do artigo.
HeraFL permanece na comparação como extensão ao estudo original, que não o avalia.
`MODEL`, `SCENARIO`, `NUM_CLIENTS`, `SELECTION_FRACTION`, `LOCAL_EPOCHS`,
`PRUNING_ARMS`, `BANDIT_ALPHA`, `BANDIT_BETA`, `REWARD_WEIGHTS`,
`LEARNING_RATE`, `POLICY_SEED` e `RESOURCE_PRESET` permitem sobrescritas.
O preset `mininet` preserva os perfis anteriores. Os defaults de `main.py`
permanecem independentes; o `start.sh` passa explicitamente o setup do artigo.
O `start.sh` ativa `--aligned-comparison` para os três algoritmos: exatamente
`NUM_ROUNDS` rodadas síncronas, avaliação após agregação, mesmos participantes
por rodada e mesmos perfis completos, inclusive limites de normalização.
Por padrão, `WAIT_FOR_ALL=1` ativa `--wait-for-all`: a agregação espera todas
as atualizações selecionadas, sem deadline, filtro de tempo ou dropout simulado.
O preditor do HERAFL continua orientando a poda. Para comparar com deadline fixo
compartilhado, use `WAIT_FOR_ALL=0 ROUND_TIMEOUT=180 ./start.sh` (segundos estimados).
`--wait-for-all` também ativa o protocolo alinhado ao executar `main.py` diretamente.
Nesse modo, `config.json` e os relatórios registram `round_timeout: null`,
e a configuração registra `wait_for_all: true`, dropout zero e filtro desativado.
A seleção estratificada do HERAFL é substituída pela seleção uniforme comum
neste protocolo controlado. Seeds de treino e condições de recursos são
indexadas por execução, rodada e cliente, sem depender dos sorteios da política.
FedAvg usa o mesmo treinamento local e ciclo de agregação, com retenção 1.0;
os otimizadores locais são recriados a cada rodada nos três métodos.
Os arquivos `config.json` registram o protocolo, deadline, seleção, seed efetiva
e parâmetros da política. Resultados antigos são preservados e precisam ser
regenerados para comparar sob o protocolo alinhado. Sem `--aligned-comparison`,
o FedAvg mantém seu ciclo nativo e o HERAFL mantém sua seleção estratificada.
O script verifica automaticamente `config.json` e todas as partições de treino e
teste. Reutiliza os dados somente quando clientes, cenário IID/non-IID,
balanceamento e partição correspondem a `NUM_CLIENTS`, `SCENARIO`, `BALANCE` e
`PARTITION`. Se os dados estiverem ausentes, inválidos ou com configuração
diferente, chama o gerador nativo do dataset antes de treinar; o gerador baixa
os dados brutos quando necessário. Não é preciso definir `GENERATE_DATA`.
`NUM_CLIENTS` omitido usa o tamanho da configuração existente ou, para um
dataset novo, o padrão do gerador. `BALANCE=unbalanced` permite partições
não balanceadas; `PARTITION` aceita `pat`, `dir` ou `exdir` para non-IID.
Alterar esses parâmetros pode regenerar as partições existentes.
`BATCH_SIZE` controla o treino, sem alterar as constantes nativas de particionamento.
`PYTHON` permite informar o executável do ambiente. Também aceita
`MODEL`, `BATCH_SIZE`, `LOCAL_EPOCHS`, `SELECTION_FRACTION`, `DEVICE_ID`, `GOAL`
e `COMPARE=0`. Cada chamada cria um `GOAL` próprio para comparar somente seus
resultados. Argumentos adicionais são repassados ao `main.py`.

No `main.py`, o modelo padrão é a CNN nativa do PFLlib. O número de clientes, o número de
classes e o cenário são lidos do `config.json` produzido pelo gerador; parâmetros
explícitos como `-nc`, `-ncl` e `--scenario` prevalecem. Os geradores normalmente
configuram 20 clientes; ajuste `num_clients` no gerador para outro tamanho.
CNN, DNN, MLR, VGG11 e VGG16 suportam a poda estruturada de FedLoad/HERAFL. A acurácia local
usada na recompensa vem do teste de cada cliente; a avaliação agregada usa as
partições de teste do framework, sem exigir validação ou teste global separado.
O tamanho dos dados no contexto do bandit reflete a quantidade real de treino de
cada cliente, normalizada pelo maior cliente.

### VGG11 e VGG16 do artigo FedLoad

As opções `MODEL=VGG11` e `MODEL=VGG16` funcionam com HeraFL, FedLoad e FedAvg:

```bash
DATASET=cifar10 MODEL=VGG11 ./start.sh
DATASET=cifar100 MODEL=VGG16 ./start.sh
# Diretamente pelo main.py:
python main.py -data Cifar10 -m VGG11 -algo FedLoad
```

O artigo usa VGG11 no CIFAR-10 e VGG16 no CINIC-10. O exemplo com CIFAR-100
é uma extensão para um dataset nativo do PFLlib; a saída acompanha seu número
de classes. Ambas recebem imagens RGB 32×32. A adaptação segue o MininetFed:
VGG11 possui 8 convoluções e VGG16, 13; convoluções 3×3 com padding 1 e ReLU,
5 MaxPools 2×2, Flatten de 512 elementos, duas camadas densas de 4096 com ReLU
e dropout 0.5, e uma camada final com logits para todas as classes. Sem BatchNorm.

O treino usa Adam (learning rate padrão 0.0001 para VGG, sobrescrito por `-lr`), espelhamento
horizontal aleatório e translação aleatória de até 10%, somente durante treino.
O artigo não especifica a learning rate. O padrão anterior de 0.001 causou
colapso das ativações do FedAvg com VGG11 no protocolo alinhado; 0.0001 é o
novo padrão compartilhado entre os três métodos. `LEARNING_RATE` permite
sobrescrevê-lo no `start.sh`; os demais modelos mantêm 0.001 nesse script.
O cabeçalho adaptado a 32×32 segue a implementação local de referência. FedAvg treina o modelo completo;
HeraFL/FedLoad podam todas as camadas ocultas, preservando as classes de saída.
No `start.sh`, o modelo é escolhido pelo dataset, salvo sobrescrita por `MODEL`.

Ao chamar `main.py` sem os parâmetros do setup do artigo, os defaults de
heterogeneidade de recursos e das políticas seguem os experimentos de
`../mininetfed_iwcmc/`: seed 42, 100 rodadas (1–100), seleção de 10%, batch 64,
10 épocas locais, 20 braços entre 0.0625 e 1, alpha 1, beta 0.75, exploração
binária e pesos 0.75/0.05/0.2 para acurácia/energia/tempo. FedLoad seleciona
aleatoriamente; HeraFL estratifica por capacidade e usa o preditor online e a
orientação por capacidade da referência. Deadlines base: 180 s e 360 s.
Os parâmetros explícitos permitem outros experimentos.

Os perfis de largura de banda, frequência, ciclos, potências e estado variável
por rodada são gerados automaticamente com os sorteios da referência.
`--client-profiles arquivo.json` sobrescreve campos por cliente.
`--straggler-predictor false` desativa o preditor do HeraFL.
`-m MininetCNN` é uma opção para as arquiteturas de referência em MNIST,
CIFAR-10 e CINIC-10, usando as mesmas partições nativas; não é o modelo padrão.

A computação é medida no processo local. Os perfis não impõem quotas de
CPU/memória ou links de rede como os containers do Mininet. Comunicação
contabiliza bytes dos parâmetros, sem o envelope MQTT/pickle. Rede e energia
usam as fórmulas dos perfis. As partições e os modelos nativos também diferem
da referência, portanto os resultados não são numericamente idênticos.

Para limitar o consumo de memória da GPU com VGG, os clientes aguardam na CPU
e somente o cliente em treino ocupa a GPU. HeraFL/FedLoad descartam o estado do
otimizador ao terminar cada submodelo (ele já era recriado por rodada), agregam
as coordenadas na CPU por camada e avaliam uma única cópia do modelo global.
O FedAvg com registro de recursos também guarda seus modelos e estados do Adam
na CPU entre treinos, preservando esses estados entre rodadas.

Resultados seguem a organização:

```text
results/<metodo>/<data>_<experimento>_<dataset>_<cenario>_run<N>/
  config.json
  <dataset>_<algoritmo>_<goal>_<N>.h5
  <dataset>_<algoritmo>_<goal>_<N>_policy.json
  analysis/<nome-da-execucao>/csv/
  analysis/<nome-da-execucao>/plots/*.png
results/comparison/<comparacao>/plots/*.png
```

Para refazer a comparação da última execução completa sem repetir treinamentos:

```bash
./start.sh --compare-only
# Ou escolha uma execução específica:
GOAL=start-20261014-133101-241695 ./start.sh --compare-only
```

O modo automático usa um único `GOAL` presente em todos os métodos solicitados,
sem misturar resultados de execuções diferentes. Para comparar execuções:

```bash
python tools/plot_comparison.py --dataset MNIST --goal test --algorithms FedLoad HERAFL FedAvg
```

Gráficos usam somente PNG. Os resultados antigos sob `legacy` não foram
recalculados. Validação: `python -m unittest discover -s tests -v`.


### HERAFL: piso de retenção e contexto consistente

O HERAFL restringe os braços pela capacidade computacional estática
`(static_cpu_capacity + 1 - static_compute_cost) / 2`, calculada a partir do
contexto inicial, antes do treinamento. Clientes com capacidade >= 0,70 usam
retenção >= 0,85; clientes com capacidade >= 0,40 usam retenção >= 0,70.
Os demais mantêm todos os braços disponíveis. A retenção representa a fração
preservada dos canais/neurônios ocultos, não a fração total de parâmetros.

Os limites podem ser ajustados sem alterar FedLoad/FedAvg:

```bash
./start.sh --hera-retention-floors 0.70 0.85 --hera-capacity-thresholds 0.40 0.70
```

A política escolhe somente braços configurados que atendem ao piso. Se o maior
braço estiver abaixo do piso, utiliza esse braço e registra
`retention_floor_satisfied=false` em `alpha_info` no relatório da política.
Para desativar apenas o piso em uma ablação, use `--hera-retention-floors 0 0`.

A atualização do LinUCB e do preditor do HERAFL recebe o mesmo contexto usado na
escolha. As métricas observadas continuam fornecendo a recompensa e o alvo do
preditor. Os relatórios JSON preservam `selection_context`, `learning_context`
e o contexto observado; o `config.json` salva pisos e limiares utilizados.
