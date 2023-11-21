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

e. Now, we initialize Sycamore, and create a [DocSet](https://sycamore.readthedocs.io/en/stable/key_concepts/concepts.html):

```python
context = sycamore.init()
pdf_docset = context.read.binary(paths, binary_format="pdf", metadata_provider=JsonManifestMetadataProvider(manifest_path))

pdf_docset.show(show_binary = False)
```

The output of this cell will show informatoin about the DocSet, and show that there are two doucments included in it.

f. This will segment the PDFs and visually show how a few pages are segmented. If you didn't install Pillow, then you will need to remove this part.

```python
def filter_func(doc: Document) -> bool:
    return doc.properties["page_number"] == 1

partitioned_docset = pdf_docset.partition(partitioner=UnstructuredPdfPartitioner())
docset = (partitioned_docset
              .partition(partitioner=UnstructuredPdfPartitioner())

#this part is for the visualization
              .flat_map(split_and_convert_to_image)
              .map_batch(DrawBoxes, f_constructor_args=[font_path])
              .filter(filter_func))

for doc in docset.take(2):
    display(Image(doc.binary_representation, height=500, width=500))
```

g. Next, we will merge the intital chunks from the document segmentation into larger chunks. We will set the maximum token size so the larger chunk will still fit in the context window of our transformer model, which we will use to create vector embeddings in a later step. We have seen larger chunk sizes improve search relevance, as the larger chunk gives more contextual information about the data in the chunk to the transformer model.

```python
pdf_docset = pdf_docset.merge(GreedyTextElementMerger(tokenizer=HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2"), max_tokens=512))
```

h. Now, we will explode the DocSet and prepare it for creating vector embeddings and loading into OpenSearch. The explode transform converts the elements of each document into top-level documents.

```python
pdf_docset = pdf_docset.explode()
pdf_docset.show(show_binary = False)
```

The output should show the exploded DocSet.

i. We will create the vector embeddings for our DocSet. The model we selected is MiniLM, and you could a different embedding model depending on your use case.


```python
pdf_docset = (pdf_docset
              .embed(embedder=SentenceTransformerEmbedder(batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2")))
pdf_docset.show(show_binary = False)
```

The output should show the DocSet with vector embeddings.

j. Before loading the OpenSearch component of Aryn Search, we need to configure the Sycamore job to: 1/communicate with the Aryn OpenSearch container and 2/have the proper configuration for the vector and keyword indexes for hybrid search. Sycamore will then create and load those indexes in the final step.

The rest endpoint for the Aryn OpenSearch container from the Quickstart is at localhost:9200.  Make sure to provide the name for the index you will create. OpenSearch is a enterprise-grade, cusotmizeable search engine and vector database, and you can adjust these settings depending on your use case.


```python
os_client_args = {
        "hosts": [{"host": "localhost", "port": 9200}],
        "http_compress": True,
        "http_auth": ("admin", "admin"),
        "use_ssl": False,
        "verify_certs": False,
        "ssl_assert_hostname": False,
        "ssl_show_warn": False,
        "timeout": 120,
    }

index = "YOUR-INDEX-NAME"

index_settings =  {
        "body": {
            "settings": {"index.knn": True, "number_of_shards": 5, "number_of_replicas": 1},
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "embedding": {
                        "dimension": 384,
                        "method": {"engine": "nmslib", "space_type": "l2", "name": "hnsw", "parameters": {}},
                        "type": "knn_vector",
                    },
                    "title": {"type": "text"},
                    "searchable_text": {"type": "text"},
                    "title_embedding": {
                        "dimension": 384,
                        "method": {"engine": "nmslib", "space_type": "l2", "name": "hnsw", "parameters": {}},
                        "type": "knn_vector",
                    },
                    "url": {"type": "text"},
                }
            },
        }
    }
```


