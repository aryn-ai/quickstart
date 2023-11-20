# Write and iterate on Sycamore jobs locally

The Quickstart configuration for Aryn Search easily launches containers for the full stack. However, you may prefer to write and iterate on your Sycamore data processing scripts locally, and load the output of these tests into the containerized Aryn stack. A benefit of this approach is that you can use a Jupyter Notebook to develop your scripts.

In this example, we will:

- Install and run components
- Write a Sycamore job
- Iterate on that job
- Run conversational search using the demo UI

## Install and run components

1. Launch Aryn Search using the containerized Quickstart following [these instructions](https://github.com/aryn-ai/quickstart#readme). However, a few notes on this step specific to this example:

- This example doesn't need Amazon Textract or Amazon S3, so you do not need to have or provide AWS credentials.
- You do not need to load the full demo dataset referred to in the Quickstart README.

2. Install [Sycamore](https://github.com/aryn-ai/sycamore) locally.

```
pip install sycamore
```

For certain PDF processing operations, you also need to install poppler, which you can do with the OS-native package manager of your choice. For example, the command for Homebrew on Mac OS is:

```
brew install poppler
```

3. [Optional] [Pillow](https://pillow.readthedocs.io/en/stable/) is used in our example to visually show the document segmentation in the notebook, though you can omit it. To install Pillow:

```
pip install pillow
```

4. Install [Jupyter Notebook](https://jupyter.org/). If you already have a python notebook enviornment, you can choose to use that instead.

```
pip install notebook
```


## 2. Write an initial Sycamore job

1. Let's download the PDFs we want to use for our sample dataset. We will use two journal articles, "Attention Is All You Need" and "A Comprehensive Survey On Applications Of Transformers For Deep Learning Tasks." We will download these locally:

```
wget -P /tmp/sycamore/data/ "https://arxiv.org/pdf/1706.03762.pdf" 
wget -P /tmp/sycamore/data/ "https://arxiv.org/pdf/2306.07303.pdf"
```


