# Write and iterate on Sycamore jobs locally

The Quickstart configuration for Aryn Search easily launches containers for the full stack. However, you may prefer to write and iterate on your Sycamore data processing scripts locally, and load the output of these tests into the containerized Aryn stack. A benefit of this approach is that you can use a local Jupyter Notebook to develop your scripts.

In this example, we will:

- [Install and run components](#Install-and-run-components)
- [Write an initial Sycamore job](#Write-an-initial-Sycamore-job)
- [Add metadata exctraction using GenAI](#Add-metadata-exctraction-using-GenAI)

The full notebook that includes the final code of the Sycamore job is [here](https://github.com/aryn-ai/quickstart/blob/main/sycamore_local_dev_example.ipynb).

## Install and run components

1. Launch Aryn Search using the containerized Quickstart following [these instructions](https://github.com/aryn-ai/quickstart#readme). However, a few notes on this step specific to this example:

- This example doesn't need Amazon Textract or Amazon S3, so you do not need to have or provide AWS credentials.
- You do not need to load the full Sort Benchmark sample dataset referred to in the Quickstart README.

2. Install [Sycamore](https://github.com/aryn-ai/sycamore) locally.


```
python3 -m venv .
. bin/activate
```


```
pip install sycamore-ai
```

For certain PDF processing operations, you also need to install poppler, which you can do with the OS-native package manager of your choice. For example, the command for Homebrew on Mac OS is:

For MacOS:

```
brew install poppler
```

For Linux:
```
sudo apt install poppler-utils
```

3. Install [Jupyter Notebook](https://jupyter.org/). If you already have a python notebook enviornment, you can choose to use that instead.

```
pip install notebook
```


## Write an initial Sycamore job

1. Download the PDFs we want to use for our sample dataset. We will use two journal articles, "Attention Is All You Need" and "A Comprehensive Survey On Applications Of Transformers For Deep Learning Tasks." We will download these locally:

```
(
    mkdir -p tmp/sycamore/data
    cd tmp/sycamore/data
    curl -O https://arxiv.org/pdf/1706.03762.pdf
    curl -O https://arxiv.org/pdf/2306.07303.pdf
)
```

2. Launch Juypter Notebook

Return to the root of the working directory:

```
cd ~/.sycamore
```

If you haven't set your OpenAI Key, do so before starting your notebook:
```
export OPENAI_API_KEY=YOUR-KEY
```

Before you start up the notebook, make sure OpenSearch is running.  

```
curl localhost:9200
```

If it's not, then start up the container using the [Quickstart instructions](https://github.com/aryn-ai/quickstart).

If the OpenSearch is running (and you get a response), then run:
```
jupyter notebook
```

In the browser window that appears, navigate to File -> New -> Notebook; and then choose the
preferred kernel.

3. Write initial Sycamore Job.

3a. First, we will import our dependencies from IPython, JSON, Pillow, and Sycamore:

```python
import json
import sys


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

3b. Next, we will include the creation of a metadata file that enables our demo UI to show and highlight the source documents when clicking on a search result. In this example, the demo UI will pull the document from a publicly accessible URL. However, you could choose to host the documents in Amazon S3 (common for enterprise data) or other locations accessible by the demo UI container.

You can make addditional cells with the rectangle over a + button on the right side of each cell.

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

3c. The next two cells will show a quick view of the PDF documents we will ingest, if we want to inspect them or take a closer look:

```
IFrame(str("tmp/sycamore/data/1706.03762.pdf"), width=700, height=600)
```

```
IFrame(str("tmp/sycamore/data/2306.07303.pdf"), width=700, height=600)
```

3d. Next, we will set some variables:

```python
paths = "tmp/sycamore/data/"
font_path = "Arial.ttf"

openai_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)
```
If you are using Linux, you may not have the Arial font installed. If so, choose a different font.

3e. Now, we initialize Sycamore, and create a [DocSet](https://sycamore.readthedocs.io/en/stable/key_concepts/concepts.html):

```python
context = sycamore.init()
pdf_docset = context.read.binary(paths, binary_format="pdf", metadata_provider=JsonManifestMetadataProvider(manifest_path))

pdf_docset.show(show_binary = False)
```

If you are using Linux, you may need to install this package:

```
pip install async_timeout
```

The output of this cell will show information about the DocSet, and show that there are two doucments included in it.

3f. This cell will segment the PDFs and visually show how a few pages are segmented. If you didn't install Pillow, then you will need to remove the visual part.

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

3g. Next, we will merge the intital chunks from the document segmentation into larger chunks. We will set the maximum token size so the larger chunk will still fit in the context window of our transformer model, which we will use to create vector embeddings in a later step. We have seen larger chunk sizes improve search relevance, as the larger chunk gives more contextual information about the data in the chunk to the transformer model.

```python
pdf_docset = pdf_docset.merge(GreedyTextElementMerger(tokenizer=HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2"), max_tokens=512))
```

3h. Now, we will explode the DocSet and prepare it for creating vector embeddings and loading into OpenSearch. The explode transform converts the elements of each document into top-level documents.

```python
pdf_docset = pdf_docset.explode()
pdf_docset.show(show_binary = False)
```

The output should show the exploded DocSet.

3i. We will now create the vector embeddings for our DocSet. The model we selected is MiniLM, and you could choose a different embedding model depending on your use case.


```python
pdf_docset = (pdf_docset
              .embed(embedder=SentenceTransformerEmbedder(batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2")))
pdf_docset.show(show_binary = False)
```

The output should show the DocSet with vector embeddings.

3j. Before loading the OpenSearch component of Aryn Search, we need to configure the Sycamore job to: 1/communicate with the Aryn OpenSearch container and 2/have the proper configuration for the vector and keyword indexes for hybrid search. Sycamore will then create and load those indexes in the final step.

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

3k. This is the final part of the Sycamore job. We will load the data and vector embeddings into the OpenSearch container using the configuration supplied above.

```python
pdf_docset.write.opensearch(os_client_args=os_client_args, index_name=index, index_settings=index_settings)
```

4. Once the data is loaded into OpenSearch, you can use the demo UI for conversational search on it.
- Using your internet browser, visit http://localhost:3000 . Make sure the demo UI container is still running from the Quickstart.
- Make sure the index selected in the dropdown has the same name you provided in step 3j
- Create a new conversation. Enter the name for your conversation in the text box in the left "Conversations" panel, and hit enter or click the "add convo" icon on the right of the text box.
- As a sample question, you can ask "Who wrote Attention Is All You Need?"

The results of the hybrid search are in the right hand panel, and you can click through to find the highlighted passage (step 3b enabled this). Though we are getting good results back from hybrid search, it would be nice if we could have the titles and other information for each passage. In the next section, we will iterate on our Sycamore job, and use generative AI to extract some metadata.

## Add metadata exctraction using GenAI

1. Going back to our notebook, let's add a cell after the cell we created in step 3f above. Then, restart the Sycamore processing job by rerunning the cells prior to this one.

2. In this cell, we will add prompt templates for extracting titles and authors. These prompts train a generative AI model to identify a title (or author) by giving examples, and then we will use the trained model to identify and extract them for each document.

```python
 title_context_template = """
    ELEMENT 1: Jupiter's Moons
    ELEMENT 2: Ganymede 2020
    ELEMENT 3: by Audi Lauper and Serena K. Goldberg. 2011
    ELEMENT 4: From Wikipedia, the free encyclopedia
    ELEMENT 5: Ganymede, or Jupiter III, is the largest and most massive natural satellite of Jupiter as well as in the Solar System, being a planetary-mass moon. It is the largest Solar System object without an atmosphere, despite being the only moon of the Solar System with a magnetic field. Like Titan, it is larger than the planet Mercury, but has somewhat less surface gravity than Mercury, Io or the Moon.
    =========
    "Ganymede 2020"

    ELEMENT 1: FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation
    ELEMENT 2: Tarun Kalluri * UCSD
    ELEMENT 3: Deepak Pathak CMU
    ELEMENT 4: Manmohan Chandraker UCSD
    ELEMENT 5: Du Tran Facebook AI
    ELEMENT 6: https://tarun005.github.io/FLAVR/
    ELEMENT 7: 2 2 0 2
    ELEMENT 8: b e F 4 2
    ELEMENT 9: ]
    ELEMENT 10: V C . s c [
    ========
    "FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation"
    
    """
author_context_template = """
    ELEMENT 1: Jupiter's Moons
    ELEMENT 2: Ganymede 2020
    ELEMENT 3: by Audi Lauper and Serena K. Goldberg. 2011
    ELEMENT 4: From Wikipedia, the free encyclopedia
    ELEMENT 5: Ganymede, or Jupiter III, is the largest and most massive natural satellite of Jupiter as well as in the Solar System, being a planetary-mass moon. It is the largest Solar System object without an atmosphere, despite being the only moon of the Solar System with a magnetic field. Like Titan, it is larger than the planet Mercury, but has somewhat less surface gravity than Mercury, Io or the Moon.
    =========
    Audi Laupe, Serena K. Goldberg

    ELEMENT 1: FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation
    ELEMENT 2: Tarun Kalluri * UCSD
    ELEMENT 3: Deepak Pathak CMU
    ELEMENT 4: Manmohan Chandraker UCSD
    ELEMENT 5: Du Tran Facebook AI
    ELEMENT 6: https://tarun005.github.io/FLAVR/
    ELEMENT 7: 2 2 0 2
    ELEMENT 8: b e F 4 2
    ELEMENT 9: ]
    ELEMENT 10: V C . s c [
    ========
    Tarun Kalluri, Deepak Pathak, Manmohan Chandraker, Du Tran

  """
```

3. Add another cell. In this cell, we will use Sycamore's entity extractor with the prompt templates. We are selecting OpenAI as the generative AI model to use for this extraction.

```python
pdf_docset = (partitioned_docset
              .extract_entity(entity_extractor=OpenAIEntityExtractor("title", llm=openai_llm, prompt_template=title_context_template))
              .extract_entity(entity_extractor=OpenAIEntityExtractor("authors", llm=openai_llm, prompt_template=author_context_template)))

pdf_docset = pdf_docset.spread_properties(["title", "authors"])

pdf_docset.show(show_binary = False, show_elements=False)
```

The output should show the title and author added to the elements in the DocSet.

4. Change the index name you added in step 3j, so you can create and load a new index with your reprocessed data. Run the rest of the cells in the notebook, and load the data into OpenSearch.

5. Once the data is loaded into OpenSearch, you can use the demo UI for conversational search on it.
- Using your internet browser, visit http://localhost:3000 . Make sure the demo UI container is still running from the Quickstart
- Make sure the index selected in the dropdown has the same name you provided in the previous step
- The titles should appear with the hybrid search results in the right panel

Congrats! You've developed and iterated on a Sycamore data preparation script locally, and used generative AI to extract metatdata and enrich your dataset. To productionize this use case, you could automate this processing job using the Sycamore container deployed in the Quickstart configuration.
