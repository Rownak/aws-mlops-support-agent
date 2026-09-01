"""This project's rag_core source types.

Importing this package registers `awsdocs_git` with rag_core's source
REGISTRY, which is what lets config.yml name it as a `type:`. Anything that
builds sources from this project's config must import this first — see
`ingest.py`.
"""

from aws_mlops_support_agent.sources.fetch import AwsDocsGitSource

__all__ = ["AwsDocsGitSource"]
