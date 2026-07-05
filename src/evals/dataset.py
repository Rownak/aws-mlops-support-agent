"""Task 5.1 — The eval set, declared as data.

Same philosophy as `src/ingest/sources.py`: adding an eval case = appending
one entry here; the runner (`run.py`) never hardcodes a question.

Labels are FILE-level, not chunk-level: a retrieval "hit" means any expected
doc file appears among the top-k chunks' sources. Chunk-level labels would
break every time chunking parameters change; filenames survive re-ingestion
and are already carried in every chunk's metadata (task 1.3).

Because some filenames exist in both repos (e.g. `concepts.md`,
`troubleshooting.md`), expected files are written as "service/filename",
matching `f"{chunk.service}/{chunk.source_file}"`.

Two kinds of cases:
- On-corpus (`expected_files` non-empty, `should_escalate=False`): the
  answer IS in the CodeBuild/CodePipeline user guides; retrieval should
  surface one of the listed files, and the agent should answer.
- Off-corpus (`expected_files=()`, `should_escalate=True`): plausible AWS
  questions our corpus can't answer. The right behavior is to escalate, not
  bluff. These negative cases are what actually test MIN_TOP_SCORE from the
  other side — a set of only answerable questions can't catch a threshold
  that never escalates.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    question: str
    # "service/filename" entries that count as relevant; empty = off-corpus.
    expected_files: tuple[str, ...]
    # Should the agent route to escalate instead of answering?
    should_escalate: bool
    # Why these files / why off-corpus — the "notes on expected relevant
    # docs" the task asks for.
    notes: str


EVAL_CASES: list[EvalCase] = [
    # --- CodeBuild, on-corpus ---
    EvalCase(
        question="How do I cache dependencies between builds in CodeBuild?",
        expected_files=("codebuild/build-caching.md",),
        should_escalate=False,
        notes="Dedicated page on S3 vs local caching and the three local cache modes.",
    ),
    EvalCase(
        question="What phases can I define in a CodeBuild buildspec file?",
        expected_files=("codebuild/build-spec-ref.md",),
        should_escalate=False,
        notes="Buildspec reference documents install/pre_build/build/post_build phases.",
    ),
    EvalCase(
        question="How do I set environment variables for a CodeBuild build?",
        expected_files=(
            "codebuild/build-env-ref-env-vars.md",
            "codebuild/build-spec-ref.md",
        ),
        should_escalate=False,
        notes="Env-vars reference page; buildspec ref also covers the `env` block.",
    ),
    EvalCase(
        question="How do I trigger a CodeBuild build automatically when I push to a GitHub branch?",
        expected_files=("codebuild/github-webhook.md", "codebuild/webhooks.md"),
        should_escalate=False,
        notes="GitHub webhook events page (filter groups per branch) or the webhooks overview.",
    ),
    EvalCase(
        question="What compute types and memory sizes are available for CodeBuild builds?",
        expected_files=("codebuild/build-env-ref-compute-types.md",),
        should_escalate=False,
        notes="Compute types page lists memory/vCPU/disk per instance type.",
    ),
    EvalCase(
        question="Can I run CodeBuild builds locally on my own machine to debug a buildspec?",
        expected_files=("codebuild/use-codebuild-agent.md",),
        should_escalate=False,
        notes="'Run builds locally with the AWS CodeBuild agent' page.",
    ),
    EvalCase(
        question="How do I view the test reports for my CodeBuild builds?",
        expected_files=(
            "codebuild/test-view-reports.md",
            "codebuild/test-report.md",
            "codebuild/test-reporting.md",
        ),
        should_escalate=False,
        notes="Test-reporting pages; any of the three covers viewing reports.",
    ),
    # --- CodePipeline, on-corpus ---
    EvalCase(
        question="How do I add a manual approval step to my CodePipeline pipeline?",
        expected_files=(
            "codepipeline/approvals-action-add.md",
            "codepipeline/approvals.md",
        ),
        should_escalate=False,
        notes="Dedicated 'add a manual approval action' page plus the approvals overview.",
    ),
    EvalCase(
        question="How do I make my pipeline start automatically when the source repo changes?",
        expected_files=(
            "codepipeline/pipelines-trigger-source-repo-changes-console.md",
            "codepipeline/pipelines-trigger-source-repo-changes-cli.md",
            "codepipeline/pipelines-trigger-source-repo-changes-cfn.md",
            "codepipeline/triggering.md",
            "codepipeline/pipelines-about-starting.md",
        ),
        should_escalate=False,
        notes="Change-detection/trigger pages (console/CLI/CFN variants) or the starting overview.",
    ),
    EvalCase(
        question="What are stages, actions, and transitions in a CodePipeline pipeline?",
        expected_files=(
            "codepipeline/concepts.md",
            "codepipeline/reference-pipeline-structure.md",
        ),
        should_escalate=False,
        notes="Concepts page defines all three; structure reference is also relevant.",
    ),
    EvalCase(
        question="How do I retry a failed action in a CodePipeline stage?",
        expected_files=("codepipeline/actions-retry.md",),
        should_escalate=False,
        notes="Dedicated 'retry a failed action' page.",
    ),
    EvalCase(
        question="How do I stop a pipeline execution that is currently in progress?",
        expected_files=("codepipeline/pipelines-stop.md",),
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
