# SurfEmb

**SurfEmb: Dense and Continuous Correspondence Distributions  
for Object Pose Estimation with Learnt Surface Embeddings**  
Rasmus Laurvig Haugard, Anders Glent Buch  
IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2022  
[pre-print](https://arxiv.org/abs/2111.13489) |
[project-site](https://surfemb.github.io/)

The easiest way to explore correspondence distributions is through the [project site](https://surfemb.github.io/).

The following describes how to reproduce the results.

## Install

Download surfemb:

```shell
$ git clone https://github.com/rasmushaugaard/surfemb.git
$ cd surfemb
```

All following commands are expected to be run in the project root directory.

[Install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
, create a new environment, *surfemb*, and activate it:

```shell
$ conda env create -f environment.yml
$ conda activate surfemb
```

## Download BOP data

Download and extract datasets from the [BOP site](https://bop.felk.cvut.cz/datasets/).
*Base archive*, and *object models* are needed for both training and inference. For training, *PBR-BlenderProc4BOP
training images* are needed as well, and for inference, the *BOP'19/20 test images* are needed.

Extract the datasets under ```data/bop``` (or make a symbolic link).

## Model

Download a trained model (see *releases*):

```shell
$ wget https://github.com/rasmushaugaard/surfemb/releases/download/v0.0.1/tless-2rs64lwh.compact.ckpt -P data/models
```

**OR**

Train a model:

```shell
$ python -m surfemb.scripts.train [dataset] --gpus [gpu ids]
```

For example, to train a model on *T-LESS* on *cuda:0*

```shell
$ python -m surfemb.scripts.train tless --gpus 0
```

## Inference data

We use the detections from [CosyPose's](https://github.com/ylabbe/cosypose) MaskRCNN models, and sample surface points
evenly for inference.  
For ease of use, this data can be downloaded and extracted as follows:

```shell
$ wget https://github.com/rasmushaugaard/surfemb/releases/download/v0.0.1/inference_data.zip
$ unzip inference_data.zip
```

**OR**

<details>
<summary>Extract detections and sample surface points</summary>

### Surface samples

First, flip the normals of ITODD object 18, which is inside out. 

Then remove invisible parts of the objects

```shell
$ python -m surfemb.scripts.misc.surface_samples_remesh_visible [dataset] 
```

sample points evenly from the mesh surface

```shell
$ python -m surfemb.scripts.misc.surface_samples_sample_even [dataset] 
```

and recover the normals for the sampled points.

```shell
$ python -m surfemb.scripts.misc.surface_samples_recover_normals [dataset] 
```

### Detection results

Download CosyPose in the same directory as SurfEmb was downloaded in, install CosyPose and follow their guide to
download their BOP-trained detection results. Then:

```shell
$ python -m surfemb.scripts.misc.load_detection_results [dataset]
```

</details>

## Inference inspection

To see pose estimation examples on the training images run

```shell
$ python -m surfemb.scripts.infer_debug [model_path] --device [device]
```

*[device]* could for example be *cuda:0* or *cpu*.

Add ```--real``` to use the test images with simulated crops based on the ground truth poses, or further
add ```--detections``` to use the CosyPose detections.

## Inference for BOP evaluation

Inference is run on the (real) test images with CosyPose detections:

```shell
$ python -m surfemb.scripts.infer [model_path] --device [device]
```

Pose estimation results are saved to ```data/results```.  
To obtain results with depth (requires running normal inference first), run

```shell
$ python -m surfemb.scripts.infer_refine_depth [model_path] --device [device]
```

The results can be formatted for BOP evaluation using

```shell
$ python -m surfemb.scripts.misc.format_results_for_eval [poses_path]
```

Either upload the formatted results to the BOP Challenge website or evaluate using
the [BOP toolkit](https://github.com/thodan/bop_toolkit).

## Extra

Custom dataset:
Format the dataset as a BOP dataset and put it in *data/bop*.