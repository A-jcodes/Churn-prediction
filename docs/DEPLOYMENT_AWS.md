# Deploying the Churn API on AWS

Two paths, in order of recommendation:

1. **App Runner** (used here): fully managed, HTTPS out of the box, scales to zero-ish cost at portfolio traffic levels, no VPC or load balancer to configure. The right choice for a single container serving a model.
2. **ECS Fargate**: more moving parts (cluster, task definition, ALB, security groups) but the standard answer when the service needs VPC integration, sidecars, or fine-grained scaling. Sketched at the end.

Both start the same way: build the image and push it to ECR.

## Prerequisites

- AWS account with the AWS CLI v2 installed and configured (`aws configure`)
- Docker running locally
- The trained model present at `models/churn_model.joblib` (run `python -m src.train` first; the Dockerfile copies the artifact into the image)

Set two variables used throughout (pick your region):

```bash
export AWS_REGION=ca-central-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
```

## Step 1: Push the image to ECR

```bash
# one-time: create the repository
aws ecr create-repository \
  --repository-name telco-churn-api \
  --region $AWS_REGION \
  --image-scanning-configuration scanOnPush=true

# authenticate Docker to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# build for x86_64 (App Runner requirement; matters if you build on ARM)
docker build --platform linux/amd64 -t telco-churn-api .

# tag and push
docker tag telco-churn-api:latest \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/telco-churn-api:latest
docker push \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/telco-churn-api:latest
```

## Step 2: Create the App Runner service

App Runner needs an IAM role allowing it to pull from ECR. One-time setup:

```bash
cat > apprunner-trust.json <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "build.apprunner.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
JSON

aws iam create-role \
  --role-name AppRunnerECRAccessRole \
  --assume-role-policy-document file://apprunner-trust.json

aws iam attach-role-policy \
  --role-name AppRunnerECRAccessRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
```

Then create the service:

```bash
aws apprunner create-service \
  --service-name telco-churn-api \
  --region $AWS_REGION \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "'$AWS_ACCOUNT_ID'.dkr.ecr.'$AWS_REGION'.amazonaws.com/telco-churn-api:latest",
      "ImageRepositoryType": "ECR",
      "ImageConfiguration": {"Port": "8000"}
    },
    "AuthenticationConfiguration": {
      "AccessRoleArn": "arn:aws:iam::'$AWS_ACCOUNT_ID':role/AppRunnerECRAccessRole"
    },
    "AutoDeploymentsEnabled": true
  }' \
  --instance-configuration '{"Cpu": "1024", "Memory": "2048"}' \
  --health-check-configuration '{"Protocol": "HTTP", "Path": "/health"}'
```

`AutoDeploymentsEnabled: true` means every future `docker push` to the `:latest` tag redeploys automatically, which is the simplest possible CD loop.

## Step 3: Verify

```bash
aws apprunner list-services --region $AWS_REGION \
  --query "ServiceSummaryList[?ServiceName=='telco-churn-api'].ServiceUrl" --output text
```

Then hit the returned URL:

```bash
curl https://<service-url>/health
curl -X POST https://<service-url>/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":"No","Partner":"Yes","Dependents":"No",
       "tenure":2,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic",
       "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No",
       "StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month",
       "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
       "MonthlyCharges":95.7,"TotalCharges":191.4}'
```

FastAPI's interactive docs are live at `https://<service-url>/docs`, which makes a great screenshot for the README.

## Cost control (important for a portfolio project)

- App Runner at 1 vCPU / 2 GB with near-zero traffic costs a few dollars per month while running; **pause the service when not demoing it** (`aws apprunner pause-service`) and cost drops to pennies for provisioned memory only.
- ECR storage for one small image is cents per month.
- Set a billing alarm before creating anything: CloudWatch billing alert at 5 CAD is a five-minute setup that prevents surprises.
- Delete everything when done: `aws apprunner delete-service`, `aws ecr delete-repository --force`.

## The ECS Fargate alternative (when to bother)

Choose Fargate over App Runner when the service must sit inside a VPC with private resources (an RDS feature store, internal-only access), needs sidecar containers, or requires custom scaling policies. The shape of the work: create an ECS cluster, register a task definition pointing at the same ECR image (port 8000, awsvpc network mode, 0.5 vCPU / 1 GB is plenty), create a Fargate service behind an Application Load Balancer with a target group health check on `/health`, and open the ALB security group on 443. It is roughly an afternoon of configuration versus ten minutes for App Runner, for capabilities this project does not yet need. Documenting that trade-off is itself the point: right-sizing infrastructure is the skill.

## Production hardening (documented as next steps)

- **Auth:** the endpoint is currently public. Add an API key check via FastAPI dependency, or front with API Gateway.
- **Observability:** App Runner ships stdout to CloudWatch Logs automatically; add structured JSON logging of prediction inputs/outputs (minus PII) for drift monitoring.
- **CI/CD:** GitHub Actions workflow on merge to main: run pytest, build and push the image, let App Runner auto-deploy. OIDC federation avoids storing AWS keys in GitHub secrets.
- **Model updates:** retraining produces a new image (model baked in at build time). This keeps serving immutable and rollback trivial: redeploy the previous image tag.
