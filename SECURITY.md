# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow responsible disclosure practices.

### How to Report

1. **DO NOT** create a public GitHub issue for security vulnerabilities
2. Email security concerns to: [security@example.com] (or contact maintainers directly)
3. Include the following in your report:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Resolution Timeline**: Depends on severity
  - Critical: 24-72 hours
  - High: 1-2 weeks
  - Medium: 2-4 weeks
  - Low: Next release cycle

### After Reporting

1. We will confirm receipt of your report
2. We will investigate and determine the impact
3. We will develop and test a fix
4. We will release the fix and credit you (unless you prefer anonymity)
5. We will publish a security advisory

## Security Best Practices

### For Deployment

#### Environment Variables

```bash
# ❌ Never commit secrets
API_SECRET_KEY=super-secret-key

# ✅ Use environment variables
export API_SECRET_KEY=$(openssl rand -hex 32)
```

#### Configuration

```python
# ✅ Use secure defaults
class Settings(BaseSettings):
    # Always require API key in production
    api_key: str = Field(..., min_length=32)
    
    # Disable debug in production
    debug: bool = Field(default=False)
    
    # Use HTTPS
    api_url: str = Field(default="https://api.example.com")
```

#### Docker Security

```dockerfile
# ✅ Use non-root user
RUN adduser --disabled-password --gecos "" appuser
USER appuser

# ✅ Use specific image tags
FROM python:3.11.7-slim-bookworm

# ✅ Don't expose unnecessary ports
EXPOSE 8000
```

### For Development

#### Dependencies

```bash
# Regularly update dependencies
pip install --upgrade pip
pip install pip-audit

# Audit for vulnerabilities
pip-audit

# Pin exact versions in production
pip freeze > requirements.lock
```

#### Secrets Management

```python
# ✅ Use environment variables
import os
api_key = os.environ["API_KEY"]

# ✅ Use secret managers in production
# AWS Secrets Manager, HashiCorp Vault, etc.

# ❌ Never hardcode secrets
api_key = "hardcoded-secret"  # BAD!
```

#### Input Validation

```python
from pydantic import BaseModel, Field, validator

class PredictionRequest(BaseModel):
    """Validated prediction request."""
    
    customer_id: str = Field(..., min_length=1, max_length=50)
    tenure: float = Field(..., ge=0, le=100)
    
    @validator("customer_id")
    def validate_customer_id(cls, v: str) -> str:
        # Prevent injection attacks
        if not v.isalnum():
            raise ValueError("Customer ID must be alphanumeric")
        return v
```

### For API Usage

#### Rate Limiting

The API implements rate limiting to prevent abuse:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640000000
```

Respect rate limits and implement exponential backoff:

```python
import time
from tenacity import retry, wait_exponential

@retry(wait=wait_exponential(multiplier=1, min=4, max=60))
def call_api():
    response = requests.post(API_URL, json=data)
    if response.status_code == 429:
        raise Exception("Rate limited")
    return response
```

#### Authentication

```python
# ✅ Use API keys in headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "X-Request-ID": str(uuid.uuid4()),
}

# ❌ Don't put secrets in URLs
requests.get(f"{API_URL}?api_key={api_key}")  # BAD!
```

## Security Features

### Implemented

- [x] Input validation with Pydantic
- [x] Rate limiting (configurable)
- [x] Request ID tracing
- [x] Structured logging (no sensitive data)
- [x] Non-root Docker user
- [x] Health check endpoints
- [x] CORS configuration

### Planned

- [ ] OAuth2/JWT authentication
- [ ] API key rotation
- [ ] Audit logging
- [ ] Encryption at rest
- [ ] mTLS for internal services

## Vulnerability Disclosure Timeline

| Date | Event |
|------|-------|
| Day 0 | Vulnerability reported |
| Day 2 | Acknowledgment sent |
| Day 7 | Initial assessment complete |
| Day 14-30 | Fix developed and tested |
| Day 30 | Fix released |
| Day 37 | Public disclosure |

## Security Contacts

- Primary: Repository maintainers
- Backup: Create private security advisory on GitHub

## Acknowledgments

We thank the following individuals for responsibly disclosing security issues:

- (No vulnerabilities reported yet)

---

*This security policy is based on industry best practices and may be updated as our security posture evolves.*
