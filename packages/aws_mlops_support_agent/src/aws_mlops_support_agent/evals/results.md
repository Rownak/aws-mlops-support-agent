Retrieval eval — hit@4 against the live Pinecone index, escalation decided by `route_after_retrieve` (threshold: top cosine < 0.35 → escalate).

| # | Question | Expected doc(s) | Hit@4 | Top score | Gap | Escalated (want) | OK |
|---|----------|-----------------|-------|-----------|-----|------------------|----|
| 1 | How do I cache dependencies between builds in CodeBuild? | build-caching.md | yes | 0.625 | 0.033 | no (no) | ✅ |
| 2 | What phases can I define in a CodeBuild buildspec file? | build-spec-ref.md | yes | 0.746 | 0.041 | no (no) | ✅ |
| 3 | How do I set environment variables for a CodeBuild build? | build-env-ref-env-vars.md, build-spec-ref.md | yes | 0.675 | 0.029 | no (no) | ✅ |
| 4 | How do I trigger a CodeBuild build automatically when I push to a GitHub branch? | github-webhook.md, webhooks.md | **no** | 0.624 | 0.033 | no (no) | ❌ |
| 5 | What compute types and memory sizes are available for CodeBuild builds? | build-env-ref-compute-types.md | yes | 0.672 | 0.032 | no (no) | ✅ |
| 6 | Can I run CodeBuild builds locally on my own machine to debug a buildspec? | use-codebuild-agent.md | yes | 0.577 | 0.011 | no (no) | ✅ |
| 7 | How do I view the test reports for my CodeBuild builds? | test-view-reports.md, test-report.md, test-reporting.md | yes | 0.774 | 0.008 | no (no) | ✅ |
| 8 | How do I add a manual approval step to my CodePipeline pipeline? | approvals-action-add.md, approvals.md | yes | 0.799 | 0.017 | no (no) | ✅ |
| 9 | How do I make my pipeline start automatically when the source repo changes? | pipelines-trigger-source-repo-changes-console.md, pipelines-trigger-source-repo-changes-cli.md, pipelines-trigger-source-repo-changes-cfn.md, triggering.md, pipelines-about-starting.md | yes | 0.630 | 0.006 | no (no) | ✅ |
| 10 | What are stages, actions, and transitions in a CodePipeline pipeline? | concepts.md, reference-pipeline-structure.md | yes | 0.718 | 0.002 | no (no) | ✅ |
| 11 | How do I retry a failed action in a CodePipeline stage? | actions-retry.md | yes | 0.753 | 0.021 | no (no) | ✅ |
| 12 | How do I stop a pipeline execution that is currently in progress? | pipelines-stop.md | yes | 0.634 | 0.007 | no (no) | ✅ |
| 13 | How do I configure a Kubernetes ingress controller on Amazon EKS? | — | — | 0.415 | 0.006 | no (yes) | ❌ |
| 14 | How do I train and deploy a machine learning model with Amazon SageMaker? | — | — | 0.479 | 0.021 | no (yes) | ❌ |
| 15 | How do I reset the root user password for my AWS account? | — | — | 0.658 | 0.188 | no (yes) | ❌ |

**Hit@4 (on-corpus):** 11/12
**Escalation accuracy:** 12/15

### Failures — what was actually retrieved

- **#4** How do I trigger a CodeBuild build automatically when I push to a GitHub branch?
  - codebuild/sample-bitbucket-pull-request.md
  - codebuild/sample-github-pull-request.md
  - codebuild/run-build-cli-auto-start.md
- **#13** How do I configure a Kubernetes ingress controller on Amazon EKS?
  - codepipeline/action-reference-ECS.md
  - codepipeline/create-cwe-ecr-source-console.md
  - codepipeline/tutorials-ecs-ecr-codedeploy.md
  - codepipeline/integrations-action-type.md
- **#14** How do I train and deploy a machine learning model with Amazon SageMaker?
  - codepipeline/tutorials-simple-codecommit.md
  - codepipeline/integrations-action-type.md
  - codepipeline/tutorials-ecs-ecr-codedeploy.md
- **#15** How do I reset the root user password for my AWS account?
  - codepipeline/security-iam.md
  - codepipeline/getting-started-codepipeline.md
  - codebuild/auth-and-access-control.md
