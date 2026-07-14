# AWS setup for the ECS Fargate deploy (task 6.4)

Deploys the image pushed in 6.3 as a public demo: one Fargate task with a
public IP (no load balancer — cheapest; the IP changes on each task restart).
Secrets come from AWS Secrets Manager. Jira is dry-run twice over: the
`DRY_RUN=true` env var, and `demo_config()` forcing it in code.

Steps:

1. Store the API keys in Secrets Manager.
2. Create the task **execution role** (pull image, write logs, read that secret).
3. Create the CloudWatch log group.
4. Register the task definition.
5. Create the cluster + security group.
6. Create the service and find its public IP.
7. Verify the demo + logs. Then scale to 0 when done.

**Roles, the concept:** the *execution role* is used by ECS infrastructure
**before your code runs** — to pull from ECR, create log streams, and fetch
the secret it injects as env vars. A *task role* would be what the app itself
uses to call AWS at runtime — our app only talks to OpenAI/Pinecone/Jira over
HTTPS, so it gets **no task role at all**. A compromised container has zero
AWS permissions.

Prereq: 6.3 completed (image in ECR). All commands assume PowerShell, region
`us-east-1`, account `327573816970`, and are run from `deploy\aws_setup\`
(gitignored working dir for files with real values).

## Step 1 — Store the API keys in Secrets Manager

One secret holding a JSON object; the task definition picks individual keys
out of it. Save as `demo-secret.json` (in `deploy\aws_setup\` — **never
commit**), with your real values from `.env`:

```json
{
  "OPENAI_API_KEY": "sk-...",
  "PINECONE_API_KEY": "pcsk_..."
}
```

```powershell
aws secretsmanager create-secret `
  --name aws-mlops-support-agent/demo `
  --description "API keys for the aws-mlops-support-agent public demo" `
  --secret-string file://demo-secret.json
```

**Copy the `ARN` from the output** — it ends in a random 6-character suffix
(e.g. `...secret:aws-mlops-support-agent/demo-Ab12Cd`), so it can't be
guessed; it's `<SECRET_ARN>` in steps 2 and 4. Then delete
`demo-secret.json` — Secrets Manager is its home now.

(Cost: ~$0.40/month per secret.)

## Step 2 — Create the task execution role

Trust policy: ECS tasks (not GitHub this time) get to assume it. Save as
`ecs-trust-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "ecs-tasks.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

```powershell
aws iam create-role `
  --role-name ecsTaskExecutionRole-mlops-agent `
  --assume-role-policy-document file://ecs-trust-policy.json `
  --description "Pulls the demo image, writes logs, reads the demo secret"

# AWS-managed policy: ECR pull + CloudWatch logs
aws iam attach-role-policy `
  --role-name ecsTaskExecutionRole-mlops-agent `
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

Plus read access to **exactly one secret**. Save as `secret-read-policy.json`
(replace `<SECRET_ARN>` with the full ARN from step 1 — no angle brackets):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "secretsmanager:GetSecretValue",
      "Resource": "<SECRET_ARN>"
    }
  ]
}
```

```powershell
aws iam put-role-policy `
  --role-name ecsTaskExecutionRole-mlops-agent `
  --policy-name read-demo-secret `
  --policy-document file://secret-read-policy.json
```

## Step 3 — Create the CloudWatch log group

ECS won't create it on demand; a missing group fails the task at start.
14-day retention keeps demo logs from accruing cost forever:

```powershell
aws logs create-log-group --log-group-name /ecs/aws-mlops-support-agent
aws logs put-retention-policy `
  --log-group-name /ecs/aws-mlops-support-agent `
  --retention-in-days 14
```

## Step 4 — Register the task definition

Copy the committed template [`deploy/task-definition.json`](task-definition.json)
into `deploy\aws_setup\`, and replace **both** `<SECRET_ARN>` placeholders
with the real ARN from step 1 (keep the `:OPENAI_API_KEY::` /
`:PINECONE_API_KEY::` suffixes — that's the "pick this JSON key out of the
secret" syntax; the trailing `::` means default version).

What the template says, briefly: Fargate, 0.5 vCPU / 1 GB, x86_64 (what the
GitHub runner builds), `latest` image from ECR, port 8501, `DRY_RUN=true` in
plain env, the two API keys injected from Secrets Manager at container start
(never visible in console/logs), health check on `/_stcore/health` (ECS
ignores the Dockerfile HEALTHCHECK), logs to the group from step 3.

```powershell
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

Output shows `revision: 1`. Re-registering after any edit bumps the revision;
the service update in step 6 picks up the newest by default.

## Step 5 — Cluster + security group

```powershell
aws ecs create-cluster --cluster-name mlops-demo
```

Security group in the default VPC, allowing inbound 8501 from anywhere
(it's a public demo; the app has no auth — scale to 0 when not demoing):

```powershell
$vpc = (aws ec2 describe-vpcs --filters Name=is-default,Values=true --query "Vpcs[0].VpcId" --output text)
$sg = (aws ec2 create-security-group `
  --group-name mlops-agent-demo `
  --description "Public 8501 for the mlops support agent demo" `
  --vpc-id $vpc --query GroupId --output text)
aws ec2 authorize-security-group-ingress `
  --group-id $sg --protocol tcp --port 8501 --cidr 0.0.0.0/0
$sg   # note it down
```

Grab two default-VPC subnets (Fargate wants at least one; two spreads AZs):

```powershell
aws ec2 describe-subnets --filters Name=vpc-id,Values=$vpc `
  --query "Subnets[0:2].SubnetId" --output text
```

## Step 6 — Create the service and find the public IP

Replace the two subnet IDs and `$sg` value:

```powershell
aws ecs create-service `
  --cluster mlops-demo `
  --service-name agent `
  --task-definition aws-mlops-support-agent `
  --desired-count 1 `
  --launch-type FARGATE `
  --network-configuration "awsvpcConfiguration={subnets=[subnet-XXXX,subnet-YYYY],securityGroups=[$sg],assignPublicIp=ENABLED}"
```

Wait ~1–2 minutes (image pull + health check start period), then chase
task → network interface → public IP:

```powershell
$task = (aws ecs list-tasks --cluster mlops-demo --service-name agent --query "taskArns[0]" --output text)
$eni = (aws ecs describe-tasks --cluster mlops-demo --tasks $task `
  --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" --output text)
aws ec2 describe-network-interfaces --network-interface-ids $eni `
  --query "NetworkInterfaces[0].Association.PublicIp" --output text
```

Open **`http://<that-ip>:8501`** (http, not https — no TLS without a load
balancer; fine for a demo).

## Step 7 — Verify

1. Ask an on-corpus question (e.g. "How do I set environment variables in a
   CodeBuild buildspec?") → cited answer appears.
2. Click **Open a ticket** → the UI must show "**Demo mode** — ticket drafted
   but NOT sent to Jira (dry-run)". That's the 6.4 acceptance check: Jira
   stays dry-run in public.
3. Logs landed in CloudWatch as JSON lines (the 5.2 payoff):

   ```powershell
   aws logs tail /ecs/aws-mlops-support-agent --since 15m
   ```

4. Task shows `HEALTHY`:

   ```powershell
   aws ecs describe-tasks --cluster mlops-demo --tasks $task --query "tasks[0].healthStatus"
   ```

## Cost control — scale to zero when not demoing

Running: ~$0.025/hour (0.5 vCPU + 1 GB, us-east-1) ≈ $18/month if left on.

```powershell
# Stop (keeps all setup; $0 compute while stopped)
aws ecs update-service --cluster mlops-demo --service agent --desired-count 0

# Restart for a demo (new task = NEW public IP — re-run the step 6 IP chase)
aws ecs update-service --cluster mlops-demo --service agent --desired-count 1
```

Full teardown (reverse order): delete service (`--force`), cluster, security
group, task definition (deregister), log group, role (detach managed +
delete inline first), secret (`--force-delete-without-recovery`).

## Troubleshooting

- **Task stops immediately, service keeps retrying** — read the reason:
  `aws ecs describe-tasks --cluster mlops-demo --tasks <arn> --query "tasks[0].{stopped:stoppedReason,containers:containers[0].reason}"`
  - `ResourceInitializationError ... secretsmanager` → execution role can't
    read the secret: ARN typo in `secret-read-policy.json` or the task def's
    `valueFrom` (the ARN must include the random suffix).
  - `CannotPullContainerError` → image/tag missing in ECR, or region mismatch.
- **Task runs but page never loads** — security group inbound rule isn't
  port 8501, or you used `https://`.
- **Container starts then dies with a config error in logs** — a required
  env var didn't arrive; check the JSON key names inside the secret match
  `OPENAI_API_KEY` / `PINECONE_API_KEY` exactly.
- **`Missing required env vars`** in CloudWatch → same as above.

## Later upgrades (out of scope for 6.4)

- **Stable URL**: put an ALB in front (~$17/month idle) or a small
  CloudFront/route53 setup; needed only if the README should carry a
  permanently clickable link.
- **CI deploy**: extend the 6.3 workflow with
  `aws ecs update-service --force-new-deployment` after push, and pin the
  image tag to the commit SHA instead of `latest`.
