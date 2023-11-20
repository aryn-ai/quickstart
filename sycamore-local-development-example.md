Test


### 2. Iterate with a local version of Sycamore
You may prefer to use a local IDE or notebook to iterate on your Sycamore script. You can install Scyamore locally, and configure it to load the output to the Aryn OpenSearch container from the quickstart.

1. Install Sycamore locally. You can find the instructions [here](https://github.com/aryn-ai/sycamore#installation).

2. [Optional] Install Jupyter to easily iterate on your script with a notebook. Instructions are [here](https://jupyter.org/install).


3. To configure Sycamore to ingest into the local Aryn OpenSearch container:

```
# Write the embedded documents to a local OpenSearch index.
os_client_args = {
    "hosts" : [{"host": "localhost", "port": 9200}],
    "use_ssl" : True,
    "verify_certs" : False,
    "http_auth" : ("admin", "admin")
}
```
