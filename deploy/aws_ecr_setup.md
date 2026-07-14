# AWS setup for the GitHub Actions → ECR push (task 6.3)

One-time setup so [`.github/workflows/build-push-ecr.yml`](../.github/workflows/build-push-ecr.yml)
can push images. Five steps:

1. Create the ECR repository.
2. Register GitHub as an OIDC identity provider in IAM (once per AWS account).
3. Create an IAM role that only this GitHub repo can assume.
4. Attach a least-privilege ECR-push policy to that role.
5. Set three GitHub repo variables and run the workflow.

**How the auth works (the concept):** when the workflow runs, GitHub gives the
job a signed OIDC token that states *which repo and branch* it is running for.
The job presents that token to AWS STS (`AssumeRoleWithWebIdentity`); AWS
verifies the signature against GitHub's identity provider (step 2) and checks
the claims against the role's trust policy (step 3). If they match, STS returns
credentials that live ~1 hour. **No AWS access key ever exists in GitHub** —
nothing to leak, rotate, or revoke.

## Prerequisites

- An AWS account, and the AWS CLI v2 installed and configured
  (`aws configure`) with a user/profile that can administer IAM and ECR.
- Your 12-digit account ID:

  ```powershell
  aws sts get-caller-identity --query Account --output text
  ```

Throughout, replace:

| Placeholder      | Value                                        |
| ---------------- | -------------------------------------------- |
| `<ACCOUNT_ID>`   | the 12-digit ID from the command above       |
| `us-east-1`      | your region, if different (match `.env` / `AWS_REGION`) |

The GitHub repo (`Rownak/aws-mlops-support-agent`) and ECR repo name
(`aws-mlops-support-agent`) are already filled in below.

## Step 1 — Create the ECR repository

```powershell
aws ecr create-repository `
  --repository-name aws-mlops-support-agent `
  --region us-east-1 `
  --image-scanning-configuration scanOnPush=true
```

`scanOnPush` = free basic CVE scan of each pushed image. Note the
`repositoryUri` in the output — the image lands there.

*Console alternative:* ECR → Repositories → Create repository → name
`aws-mlops-support-agent`, enable scan on push.

## Step 2 — Register GitHub's OIDC provider in IAM

Once per AWS account (skip if it already exists — check with
`aws iam list-open-id-connect-providers`):

```powershell
aws iam create-open-id-connect-provider `
  --url https://token.actions.githubusercontent.com `
  --client-id-list sts.amazonaws.com `
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

(The thumbprint is required by the API but AWS has ignored it for GitHub since
2023 — it trusts GitHub's root CA directly. Any well-formed value works; this
is the historical GitHub one.)

*Console alternative:* IAM → Identity providers → Add provider → OpenID
Connect → provider URL `https://token.actions.githubusercontent.com`,
audience `sts.amazonaws.com`.

## Step 3 — Create the IAM role GitHub will assume

The trust policy is the security boundary: the `sub` condition means **only
workflows in `Rownak/aws-mlops-support-agent` running on branch `master`** can
assume this role. A fork, another repo, or another branch gets denied by STS.

Save as `trust-policy.json` (anywhere local, don't commit — it contains your
account ID, which is not secret but also not needed in the repo):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::<ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
          "token.actions.githubusercontent.com:sub": "repo:Rownak/aws-mlops-support-agent:ref:refs/heads/master"
        }
      }
    }
  ]
}
```

```powershell
aws iam create-role `
  --role-name github-actions-ecr-push `
  --assume-role-policy-document file://trust-policy.json `
  --description "Lets GitHub Actions in aws-mlops-support-agent push images to ECR"
```

Note the role `Arn` in the output
(`arn:aws:iam::<ACCOUNT_ID>:role/github-actions-ecr-push`) — you'll paste it
into GitHub in step 5.

## Step 4 — Attach the least-privilege ECR-push policy

Two statements: `GetAuthorizationToken` must be `Resource: "*"` (that's how
the ECR API works — it issues one docker-login token per registry), but every
push/pull action is scoped to the **one repository ARN**. If the role ever
leaked, it couldn't touch any other ECR repo, let alone other services.

Save as `ecr-push-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "EcrLogin",
      "Effect": "Allow",
      "Action": "ecr:GetAuthorizationToken",
      "Resource": "*"
    },
    {
      "Sid": "PushPullThisRepoOnly",
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:PutImage",
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer"
      ],
      "Resource": "arn:aws:ecr:us-east-1:<ACCOUNT_ID>:repository/aws-mlops-support-agent"
    }
  ]
}
```

```powershell
aws iam put-role-policy `
  --role-name github-actions-ecr-push `
  --policy-name ecr-push-this-repo `
  --policy-document file://ecr-push-policy.json
```

## Step 5 — Configure GitHub and run

GitHub repo → **Settings → Secrets and variables → Actions → Variables tab**
(variables, not secrets — none of these values is sensitive):

| Variable         | Value                                                    |
| ---------------- | -------------------------------------------------------- |
| `AWS_ROLE_ARN`   | `arn:aws:iam::<ACCOUNT_ID>:role/github-actions-ecr-push` |
| `AWS_REGION`     | `us-east-1`                                              |
| `ECR_REPOSITORY` | `aws-mlops-support-agent`                                |

Or with the GitHub CLI:

```powershell
gh variable set AWS_ROLE_ARN --body "arn:aws:iam::<ACCOUNT_ID>:role/github-actions-ecr-push"
gh variable set AWS_REGION --body "us-east-1"
gh variable set ECR_REPOSITORY --body "aws-mlops-support-agent"
```

Then either push a commit to `master` or trigger manually: GitHub →
**Actions → "Build and push to ECR" → Run workflow**.

Verify the image arrived:

```powershell
aws ecr describe-images `
  --repository-name aws-mlops-support-agent `
  --region us-east-1 `
  --query "imageDetails[].{tags:imageTags,pushed:imagePushedAt}"
```

Expect two tags on the newest image: the commit SHA and `latest`.

## Troubleshooting

- **`Not authorized to perform sts:AssumeRoleWithWebIdentity`** — the trust
  policy didn't match. Check the `sub` string character-for-character: owner
  and repo name are case-sensitive, and the branch must be `master` (a run
  from any other branch is *supposed* to fail). Also confirm the workflow has
  `permissions: id-token: write`.
- **`Could not load credentials from any providers`** — usually the
  `AWS_ROLE_ARN` variable is unset/typo'd (an empty `role-to-assume` makes
  the action fall back to nonexistent static credentials).
- **`repository ... does not exist`** — region mismatch: the ECR repo was
  created in a different region than the `AWS_REGION` variable.
- **Push denied after successful login** — the policy's repository ARN
  (region/account/name) doesn't match the repo you created in step 1.
