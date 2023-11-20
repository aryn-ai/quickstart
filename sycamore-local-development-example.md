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


## Write an initial Sycamore job

1. Download the PDFs we want to use for our sample dataset. We will use two journal articles, "Attention Is All You Need" and "A Comprehensive Survey On Applications Of Transformers For Deep Learning Tasks." We will download these locally:

```
wget -P /tmp/sycamore/data/ "https://arxiv.org/pdf/1706.03762.pdf" 
wget -P /tmp/sycamore/data/ "https://arxiv.org/pdf/2306.07303.pdf"
```

2. Launch Juypter Notebook

```
juypter notebook
```

And create a new notebook for our Sycamore job. If you haven't set your OpenAI Key, do so before starting your notebook:

```
export OPENAI_API_KEY=YOUR-KEY
```

3. Write initial Sycamore Job. The actual notebook with this script is here. However, we will go through how to contruct it below.

a. First, we will import our dependencies from IPython, JSON, Pillow, and Sycamore:

```python
import sys
print(sys.version)


from IPython.display import display, Image
from IPython.display import IFrame
from PIL import Image as PImage

import sycamore
from sycamore.data import Document
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.llms import OpenAIModels, OpenAI, LLM
from sycamore.transforms.partition import UnstructuredPdfPartitioner, HtmlPartitioner
from sycamore.llms.prompts.default_prompts import TEXT_SUMMARIZER_GUIDANCE_PROMPT_CHAT
from sycamore.transforms.summarize import Summarizer
from sycamore.transforms.extract_table import TextractTableExtractor
from sycamore.functions.document import split_and_convert_to_image, DrawBoxes
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.merge_elements import GreedyTextElementMerger
from sycamore.functions.tokenizer import HuggingFaceTokenizer
from sycamore.scans.file_scan import JsonManifestMetadataProvider
```

b. Next, we will include the creation of a metadata file that enables our demo UI to show and highlight the source documents when clicking on a search result. In this example, the demo UI will pull the document from a publically accessible URL. However, you could choose to host the documents in Amazon S3 (common for enterprise data) or other locations accessible by the demo UI container.

```python
metadata = {
  "tmp/sycamore/data/1706.03762.pdf" : {
    "_location": "https://arxiv.org/pdf/1706.03762.pdf"
  }, "tmp/sycamore/data/2306.07303.pdf" : {
    "_location": "https://arxiv.org/pdf/2306.07303.pdf"
  }
}

with open("/tmp/sycamore/manifest.json", "w") as f:
    json.dump(metadata, f)
manifest_path = "/tmp/sycamore/manifest.json"
```

c. The next two cells just show a quick view of the PDF documents we will ingest, if we want to inspect them or take a closer look:

```
IFrame(str("tmp/sycamore/data/1706.03762.pdf"), width=700, height=600)
```

```
IFrame(str("tmp/sycamore/data/2306.07303.pdf"), width=700, height=600)
```

d. We will now set some variables:

```python
paths = "tmp/sycamore/data/"
font_path = "Arial.ttf"

openai_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)
```

e. Now we initialize Sycamore, and create a [DocSet](https://sycamore.readthedocs.io/en/stable/key_concepts/concepts.html):

```python
context = sycamore.init()
pdf_docset = context.read.binary(paths, binary_format="pdf", metadata_provider=JsonManifestMetadataProvider(manifest_path))

pdf_docset.show(show_binary = False)
```

The output of this cell will show informatoin about the DocSet, and show that there are two doucments included in it.
