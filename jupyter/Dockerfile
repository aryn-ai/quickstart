# Repo name: arynai/jupyter

ARG TAG=stable
FROM arynai/sycamore:$TAG

# TODO: eric - remove this once we switch other containers over to non-root
RUN useradd -d /app --uid 54321 --gid users app
RUN chown -R app:users /app

COPY jupyter/docker-apt-packages.sh ./
RUN /bin/bash docker-apt-packages.sh

USER 54321
RUN poetry run pip install notebook ipywidgets 'opensearch-py>=2.4' && rm -rf $POETRY_CACHE_DIR
RUN mkdir -p /app/work/docker_volume /app/work/bind_dir /app/work/examples
RUN touch /app/work/AAA_SEE_README_FOR_PERSISTENT_DATA_DIRECTORIES
COPY jupyter/entrypoint.sh ./
COPY jupyter/README.md ./work
COPY sycamore_local_dev_example.ipynb ./work/examples
RUN perl -i -pe 's/localhost/opensearch/ if /9200/;s,tmp/sycamore/data,/app/work/docker_volume,' /app/work/examples/sycamore_local_dev_example.ipynb

ARG GIT_BRANCH="main"
ARG GIT_COMMIT="unknown"
ARG GIT_DIFF="unknown"

ENV GIT_BRANCH=${GIT_BRANCH}
ENV GIT_COMMIT=${GIT_COMMIT}
ENV GIT_DIFF=${GIT_DIFF}

LABEL org.opencontainers.image.authors="opensource@aryn.ai"
LABEL git_branch=${GIT_BRANCH}
LABEL git_commit=${GIT_COMMIT}
LABEL git_diff=${GIT_DIFF}

CMD /bin/bash entrypoint.sh
