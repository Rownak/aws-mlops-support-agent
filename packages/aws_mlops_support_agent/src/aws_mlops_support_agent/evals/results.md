Retrieval eval — hit@4 against the live index, escalation decided by the agent's own router (threshold: top cosine < 0.35 → escalate).

| # | Question | Expected doc(s) | Hit@4 | Top score | Gap | Escalated (want) | OK |
|---|----------|-----------------|-------|-----------|-----|------------------|----|
| 1 | How do I cache dependencies between builds in CodeBuild? | build-caching.md | yes | 0.813 | 0.027 | no (no) | ✅ |
| 2 | What phases can I define in a CodeBuild buildspec file? | build-spec-ref.md | **no** | 0.856 | 0.012 | no (no) | ❌ |
| 3 | How do I set environment variables for a CodeBuild build? | build-env-ref-env-vars.md, build-spec-ref.md | yes | 0.862 | 0.025 | no (no) | ✅ |
| 4 | How do I trigger a CodeBuild build automatically when I push to a GitHub branch? | github-webhook.md, webhooks.md | **no** | 0.817 | 0.009 | no (no) | ❌ |
| 5 | What compute types and memory sizes are available for CodeBuild builds? | build-env-ref-compute-types.md | yes | 0.840 | 0.025 | no (no) | ✅ |
| 6 | Can I run CodeBuild builds locally on my own machine to debug a buildspec? | use-codebuild-agent.md | yes | 0.787 | 0.006 | no (no) | ✅ |
| 7 | How do I view the test reports for my CodeBuild builds? | test-view-reports.md, test-report.md, test-reporting.md | yes | 0.869 | 0.033 | no (no) | ✅ |
| 8 | How do I add a manual approval step to my CodePipeline pipeline? | approvals-action-add.md, approvals.md | **no** | 0.776 | 0.003 | no (no) | ❌ |
| 9 | How do I make my pipeline start automatically when the source repo changes? | pipelines-trigger-source-repo-changes-console.md, pipelines-trigger-source-repo-changes-cli.md, pipelines-trigger-source-repo-changes-cfn.md, triggering.md, pipelines-about-starting.md | **no** | 0.747 | 0.014 | no (no) | ❌ |
| 10 | What are stages, actions, and transitions in a CodePipeline pipeline? | concepts.md, reference-pipeline-structure.md | **no** | 0.802 | 0.006 | no (no) | ❌ |
| 11 | How do I retry a failed action in a CodePipeline stage? | actions-retry.md | **no** | 0.760 | 0.021 | no (no) | ❌ |
| 12 | How do I stop a pipeline execution that is currently in progress? | pipelines-stop.md | **no** | 0.709 | 0.003 | no (no) | ❌ |
| 13 | How do I configure a Kubernetes ingress controller on Amazon EKS? | — | — | 0.689 | 0.008 | no (yes) | ❌ |
| 14 | How do I train and deploy a machine learning model with Amazon SageMaker? | — | — | 0.713 | 0.002 | no (yes) | ❌ |
| 15 | How do I reset the root user password for my AWS account? | — | — | 0.781 | 0.063 | no (yes) | ❌ |

**Hit@4 (on-corpus):** 5/12
**Escalation accuracy:** 12/15

### Failures — what was actually retrieved

- **#2** What phases can I define in a CodeBuild buildspec file?
  - create-project-console.md
  - change-project-console.md
  - getting-started-cli-create-build-spec.md
  - getting-started-create-build-spec-console.md
- **#4** How do I trigger a CodeBuild build automatically when I push to a GitHub branch?
  - sample-bitbucket-pull-request.md
  - sample-github-pull-request.md
  - trigger-create.md
- **#8** How do I add a manual approval step to my CodePipeline pipeline?
  - how-to-create-pipeline-add.md
  - how-to-create-pipeline-add-test.md
  - how-to-create-pipeline.md
  - how-to-create-pipeline-console.md
- **#9** How do I make my pipeline start automatically when the source repo changes?
  - how-to-create-pipeline.md
  - how-to-create-pipeline-cli.md
  - sample-elastic-beanstalk.md
  - change-project-console.md
- **#10** What are stages, actions, and transitions in a CodePipeline pipeline?
  - how-to-create-pipeline-add.md
  - sample-elastic-beanstalk.md
  - how-to-create-pipeline-add-test.md
  - how-to-create-pipeline-console.md
- **#11** How do I retry a failed action in a CodePipeline stage?
  - retry-build.md
  - how-to-create-pipeline-add-test.md
  - how-to-create-pipeline-add.md
  - how-to-create-pipeline-console.md
- **#12** How do I stop a pipeline execution that is currently in progress?
  - stop-batch-build.md
  - stop-build.md
  - session-manager.md
  - run-build-cli-auto-stop.md
- **#13** How do I configure a Kubernetes ingress controller on Amazon EKS?
  - sample-efs.md
  - planning.md
- **#14** How do I train and deploy a machine learning model with Amazon SageMaker?
  - sample-codedeploy.md
  - sample-elastic-beanstalk.md
  - getting-started-create-build-project-console.md
  - serverless-applications.md
- **#15** How do I reset the root user password for my AWS account?
  - auth-and-access-control.md
  - access-tokens.md
  - index.md
