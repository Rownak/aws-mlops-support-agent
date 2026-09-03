"""The eval set for THIS corpus, declared as data.

The questions are corpus-specific, so they live with the project, alongside
the `EvalCase` type and the runner in `aws_mlops_support_agent.evals`. Adding
an eval case = appending one entry here.

Labels are FILE-level, not chunk-level: a retrieval "hit" means any expected
doc file appears among the top-k chunks' sources. Chunk-level labels would
break every time chunking parameters change; filenames survive re-ingestion
and are already carried in every chunk's metadata.

Expected files are plain filenames (e.g. "build-caching.md"), matched against
`evals/run.py`'s `_basename_retriever`, which rewrites each retrieved chunk's
`metadata["source"]` (a full, OS-specific local path) down to just the
filename before the generic runner compares it — rag_core's `sources` are
`type: local` folders now, with no `source_id` concept to prefix a label
with. A few filenames exist in both repos (`concepts.md`,
`troubleshooting.md`); a case using one of those can't tell the corpora
apart, so it's noted case-by-case below.

Two kinds of cases:
- On-corpus (`expected_files` non-empty, `should_escalate=False`): the
  answer IS in the CodeBuild/CodePipeline user guides; retrieval should
  surface one of the listed files, and the agent should answer.
- Off-corpus (`expected_files=()`, `should_escalate=True`): plausible AWS
  questions our corpus can't answer. The right behavior is to escalate, not
  bluff. These negative cases are what actually test the min_top_score
  threshold from the other side — a set of only answerable questions can't
  catch a threshold that never escalates.
"""

from aws_mlops_support_agent.evals.runner import EvalCase

EVAL_CASES: list[EvalCase] = [
    # --- CodeBuild, on-corpus ---
    EvalCase(
        question="How do I cache dependencies between builds in CodeBuild?",
        expected_files=("build-caching.md",),
        should_escalate=False,
        notes="Dedicated page on S3 vs local caching and the three local cache modes.",
    ),
    EvalCase(
        question="What phases can I define in a CodeBuild buildspec file?",
        expected_files=("build-spec-ref.md",),
        should_escalate=False,
        notes="Buildspec reference documents install/pre_build/build/post_build phases.",
    ),
    EvalCase(
        question="How do I set environment variables for a CodeBuild build?",
        expected_files=(
            "build-env-ref-env-vars.md",
            "build-spec-ref.md",
        ),
        should_escalate=False,
        notes="Env-vars reference page; buildspec ref also covers the `env` block.",
    ),
    EvalCase(
        question="How do I trigger a CodeBuild build automatically when I push to a GitHub branch?",
        expected_files=("github-webhook.md", "webhooks.md"),
        should_escalate=False,
        notes="GitHub webhook events page (filter groups per branch) or the webhooks overview.",
    ),
    EvalCase(
        question="What compute types and memory sizes are available for CodeBuild builds?",
        expected_files=("build-env-ref-compute-types.md",),
        should_escalate=False,
        notes="Compute types page lists memory/vCPU/disk per instance type.",
    ),
    EvalCase(
        question="Can I run CodeBuild builds locally on my own machine to debug a buildspec?",
        expected_files=("use-codebuild-agent.md",),
        should_escalate=False,
        notes="'Run builds locally with the AWS CodeBuild agent' page.",
    ),
    EvalCase(
        question="How do I view the test reports for my CodeBuild builds?",
        expected_files=(
            "test-view-reports.md",
            "test-report.md",
            "test-reporting.md",
        ),
        should_escalate=False,
        notes="Test-reporting pages; any of the three covers viewing reports.",
    ),
    # --- CodePipeline, on-corpus ---
    EvalCase(
        question="How do I add a manual approval step to my CodePipeline pipeline?",
        expected_files=(
            "approvals-action-add.md",
            "approvals.md",
        ),
        should_escalate=False,
        notes="Dedicated 'add a manual approval action' page plus the approvals overview.",
    ),
    EvalCase(
        question="How do I make my pipeline start automatically when the source repo changes?",
        expected_files=(
            "pipelines-trigger-source-repo-changes-console.md",
            "pipelines-trigger-source-repo-changes-cli.md",
            "pipelines-trigger-source-repo-changes-cfn.md",
            "triggering.md",
            "pipelines-about-starting.md",
        ),
        should_escalate=False,
        notes="Change-detection/trigger pages (console/CLI/CFN variants) or the starting overview.",
    ),
    EvalCase(
        question="What are stages, actions, and transitions in a CodePipeline pipeline?",
        expected_files=(
            "concepts.md",
            "reference-pipeline-structure.md",
        ),
        should_escalate=False,
        notes=(
            "Concepts page defines all three; structure reference is also relevant. "
            "concepts.md also exists in the codebuild corpus (filename-only matching "
            "can't tell them apart), but that's a false-hit risk, not a false-miss one."
        ),
    ),
    EvalCase(
        question="How do I retry a failed action in a CodePipeline stage?",
        expected_files=("actions-retry.md",),
        should_escalate=False,
        notes="Dedicated 'retry a failed action' page.",
    ),
    EvalCase(
        question="How do I stop a pipeline execution that is currently in progress?",
        expected_files=("pipelines-stop.md",),
        should_escalate=False,
        notes="'Stop a pipeline execution' page (complete vs abandon in-progress actions).",
    ),
    # --- Off-corpus: the agent should escalate, not bluff ---
    EvalCase(
        question="How do I configure a Kubernetes ingress controller on Amazon EKS?",
        expected_files=(),
        should_escalate=True,
        notes="EKS/Kubernetes — different AWS service entirely; not in the corpus.",
    ),
    EvalCase(
        question="How do I train and deploy a machine learning model with Amazon SageMaker?",
        expected_files=(),
        should_escalate=True,
        notes="SageMaker — 'deploy'/'pipeline' vocabulary overlaps CI/CD, a good hard negative.",
    ),
    EvalCase(
        question="How do I reset the root user password for my AWS account?",
        expected_files=(),
        should_escalate=True,
        notes="Account management — nothing to do with CI/CD docs.",
    ),
]
