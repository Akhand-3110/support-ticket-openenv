# Support Ticket OpenEnv

## Description
Simulates a real-world customer support system where an AI agent must classify and respond to tickets.

## Observation Space
- ticket: string

## Action Space
- category: billing / tech / general
- action: refund / troubleshoot / escalate / ignore
- response: string

## Tasks
- Easy: Simple issues
- Medium: Payment issues
- Hard: Multi-step reasoning

## Run Locally
docker build -t support-env .
docker run -p 7860:7860 support-env