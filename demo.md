# Demo for BigOne


## 1、开发环境

### 1.1 自定义开发环境镜像、使用及切换实例类型
SageMaker Studio已经内置了多种 Notebook kernel环境，如多个版本的PyTorch（CPU/GPU），TensorFlow（CPU/GPU）等等。SageMaker Studio支持以 docker image的方式来自定义环境，并在开发过程中使用自己设定的 image。

#### 1.1.1 打包Image并推送到 ECR (只需要做一次）
以自己构建PyTorch 最新版本 1.12为例（目前SageMaker Studio未提供），我们接下来我们首先 build 一个image，我们将通过dockerfile的方式。
- 如下Dockerfile为参考：
未简化安装，我们用 AWS Deep Learning Container 做为base image，卸载掉之前的PyTorch版本，并安装最新的1.12版本。

```
From 727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/pytorch-training:1.10.2-gpu-py38-cu113-ubuntu20.04-e3

RUN pip3 uninstall torch torchvision -y

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip3 install ipykernel && \
    python -m ipykernel install --sys-prefix

```
- Docker image build

```
aws ecr get-login-password --region cn-north-1 | docker login --username AWS --password-stdin 727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn

docker build -t sagemaker-studio-byo-image-demo .
```


- 创建 ECR 

可在 console中完成，这里不做展开。我们将创建一个名字为 **sagemaker-studio-byo-image-demo**
 的 ECR Repo。

- push 到 ECR

```
aws ecr get-login-password --region cn-north-1 | docker login --username AWS --password-stdin 346044390830.dkr.ecr.cn-north-1.amazonaws.com.cn

docker tag sagemaker-studio-byo-image-demo:latest 346044390830.dkr.ecr.cn-north-1.amazonaws.com.cn/sagemaker-studio-byo-image-demo:latest

docker push 346044390830.dkr.ecr.cn-north-1.amazonaws.com.cn/sagemaker-studio-byo-image-demo:latest
```

#### 1.1.2 在Studio设置自定义 image

创建image

<img src="https://shishuai-share-external.s3.cn-north-1.amazonaws.com.cn/images/demo/bo/bigone/image/001.png" width="500">

填入上一步创建的image 在ECR中的URI

<img src="https://shishuai-share-external.s3.cn-north-1.amazonaws.com.cn/images/demo/bo/bigone/image/002.png" width="500">

设定名字

<img src="https://shishuai-share-external.s3.cn-north-1.amazonaws.com.cn/images/demo/bo/bigone/image/003.png" width="500">

完成后可以看见状态是 CREATED

<img src="https://shishuai-share-external.s3.cn-north-1.amazonaws.com.cn/images/demo/bo/bigone/image/004.png" width="500">

在 Studio 中添加 image

<img src="https://shishuai-share-external.s3.cn-north-1.amazonaws.com.cn/images/demo/bo/bigone/image/005.png" width="500">

选择 Existing image

<img src="https://shishuai-share-external.s3.cn-north-1.amazonaws.com.cn/images/demo/bo/bigone/image/006.png" width="500">

注意 Kernel Name设置为 python3

<img src="https://shishuai-share-external.s3.cn-north-1.amazonaws.com.cn/images/demo/bo/bigone/image/007.png" width="500">

#### 1.1.3 在 Studio中使用自定义的 image

打开 SageMaker Studio，在 Launcher下可以选择 SageMaker image，选择我们前面创建的。

<img src="https://shishuai-share-external.s3.cn-north-1.amazonaws.com.cn/images/demo/bo/bigone/image/008.png" width="800">

创建一个新的notebook，首次打开会自动配置资源（2c4GB）

<img src="https://shishuai-share-external.s3.cn-north-1.amazonaws.com.cn/images/demo/bo/bigone/image/009.png" width="800">

资源Ready之后，运行代码，输出 PyTorch 版本，符合预期

<img src="https://shishuai-share-external.s3.cn-north-1.amazonaws.com.cn/images/demo/bo/bigone/image/010.png" width="800">

接下来我们修改资源配置为 g4dn.xlarge(配置有一块T4 GPU卡）

<img src="https://shishuai-share-external.s3.cn-north-1.amazonaws.com.cn/images/demo/bo/bigone/image/011.png" width="800">

在弹出的框里面选择机型，关闭 **Fast launch only** , 可以看到更多机型

<img src="https://shishuai-share-external.s3.cn-north-1.amazonaws.com.cn/images/demo/bo/bigone/image/012.png" width="500">

几分钟后，g4实例ready后可以看见资源配置显示已经更改了

<img src="https://shishuai-share-external.s3.cn-north-1.amazonaws.com.cn/images/demo/bo/bigone/image/013.png" width="800">

再次执行输出pytorch 版本，符合预期

<img src="https://shishuai-share-external.s3.cn-north-1.amazonaws.com.cn/images/demo/bo/bigone/image/014.png" width="800">

[了解更多，参考链接](
https://docs.amazonaws.cn/sagemaker/latest/dg/studio-byoi-attach.html?icmpid=docs_sagemaker_console_studio)

### 1.2 git集成

如果需要和已有 git 库集成，可以在Studio左侧找到 git 图标

<img src="https://shishuai-share-external.s3.cn-north-1.amazonaws.com.cn/images/demo/bo/bigone/image/015.png" width="500">

可以选择初始化一个repo或者clone已有 repo

<img src="https://shishuai-share-external.s3.cn-north-1.amazonaws.com.cn/images/demo/bo/bigone/image/016.png" width="500">

本地文件改动后，可以选择提交到stage，并加入comment之后，选择commit

<img src="https://shishuai-share-external.s3.cn-north-1.amazonaws.com.cn/images/demo/bo/bigone/image/017.png" width="500">

如果要提交代码，点击红框中的图标

<img src="https://shishuai-share-external.s3.cn-north-1.amazonaws.com.cn/images/demo/bo/bigone/image/018.png" width="500">

在弹出的窗口中，填入用户名密码，完成 git push操作

<img src="https://shishuai-share-external.s3.cn-north-1.amazonaws.com.cn/images/demo/bo/bigone/image/019.png" width="500">

---

## 2、机器学习工作流

通过构建成熟的机器学习工作流，可以自动化数据处理，模型训练，模型推理等步骤，并针对模型版本进行管理，可复现每一步，并对每一步的输入输出进行追踪和管理，并可基于事件或者时间自动化触发执行整个工作流。

本例中，我们将使用 Hugging Face :hugs: 跑一个文本分类的示例。
[Hugging Face on Amazon SageMaker](https://huggingface.co/docs/sagemaker/main)

[原始示例代码地址](https://github.com/huggingface/notebooks.git)

按照前文1.2的说明，我们将如上code clone到1.1创建的环境中。

我们选择```notebooks/sagemaker/01_getting_started_pytorch/sagemaker-notebook.ipynb```为示例，该示例中将使用 Hugging Faces **transformers** 和 **datasets**库来fine tune一个文本二分类模型，数据集使用 imdb数据集，我们将完成数据的预处理（tokenization），通过SageMaker完成模型的训练，并部署为一个实时推理终端节点。

我们将在第二部分对此进行改造，将数据处理部分使用SageMaker Processing完成，并加入使用批量推理的Batch Transform，最后使用SageMaker Pipeline完成整个Pipeline的构建。

<img src="https://shishuai-share-external.s3.cn-north-1.amazonaws.com.cn/images/demo/bo/bigone/image/020.png" width="500">

[最终示例代码地址](https://github.com/snowolf/huggingfaces-text-classification-sagemaker-pipeline)

---
## 参考链接 
[SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html)