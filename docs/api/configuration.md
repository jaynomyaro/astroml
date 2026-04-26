# Configuration and Deployment Documentation

## Overview

This guide covers configuration management, deployment strategies, and operational best practices for AstroML in production environments.

## Table of Contents

1. [Configuration Management](#configuration-management)
2. [Environment Setup](#environment-setup)
3. [Database Configuration](#database-configuration)
4. [Stellar Network Configuration](#stellar-network-configuration)
5. [Model Configuration](#model-configuration)
6. [Deployment Options](#deployment-options)
7. [Docker Deployment](#docker-deployment)
8. [Kubernetes Deployment](#kubernetes-deployment)
9. [Monitoring and Logging](#monitoring-and-logging)
10. [Security Configuration](#security-configuration)

## Configuration Management

### Configuration Structure

AstroML uses a hierarchical configuration system:

```
config/
├── database.yaml          # Database connection settings
├── stellar.yaml           # Stellar network settings
├── models.yaml            # Model configuration
├── ingestion.yaml         # Data ingestion settings
├── training.yaml           # Training configuration
├── deployment.yaml         # Deployment settings
└── logging.yaml           # Logging configuration
```

### Environment-Based Configuration

Configuration files support environment-specific overrides:

```yaml
# config/base.yaml
database:
  host: localhost
  port: 5432
  
# config/development.yaml
database:
  name: astroml_dev
  user: dev_user
  
# config/production.yaml
database:
  name: astroml_prod
  user: prod_user
  ssl_mode: require
```

### Configuration Loading

```python
from astroml.config import ConfigLoader
import os

def load_configuration():
    """Load configuration based on environment."""
    
    env = os.getenv("ASTROML_ENV", "development")
    
    loader = ConfigLoader()
    config = loader.load_config(
        base_files=["config/base.yaml"],
        env_file=f"config/{env}.yaml",
        overrides=get_env_overrides()
    )
    
    return config

def get_env_overrides():
    """Get configuration overrides from environment variables."""
    overrides = {}
    
    # Database overrides
    if os.getenv("DATABASE_URL"):
        overrides["database"] = {
            "url": os.getenv("DATABASE_URL")
        }
    
    # Stellar network overrides
    if os.getenv("STELLAR_NETWORK"):
        overrides["stellar"] = {
            "network": os.getenv("STELLAR_NETWORK")
        }
    
    return overrides

config = load_configuration()
```

## Environment Setup

### Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up database
createdb astroml_dev
psql astroml_dev < schema.sql

# Configure environment
export ASTROML_ENV=development
export DATABASE_URL="postgresql://user:pass@localhost/astroml_dev"
export STELLAR_NETWORK="testnet"
```

### Production Environment

```bash
# System requirements
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib python3-dev python3-pip

# Create production user
sudo useradd -m astroml
sudo -u astroml bash

# Install dependencies
sudo -u astroml bash -c "
python3 -m venv /opt/astroml/venv
source /opt/astroml/venv/bin/activate
pip install -r requirements.txt
"

# Production database
sudo -u postgres createdb astroml_prod
sudo -u postgres createuser astroml_prod
sudo -u postgres psql -c "ALTER USER astroml_prod PASSWORD 'secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE astroml_prod TO astroml_prod;"
```

## Database Configuration

### PostgreSQL Configuration

```yaml
# config/database.yaml
database:
  # Connection settings
  host: ${DB_HOST:localhost}
  port: ${DB_PORT:5432}
  name: ${DB_NAME:astroml}
  user: ${DB_USER:astroml}
  password: ${DB_PASSWORD}
  url: ${DATABASE_URL}
  
  # Connection pool
  pool_size: ${DB_POOL_SIZE:10}
  max_overflow: ${DB_MAX_OVERFLOW:20}
  pool_timeout: ${DB_POOL_TIMEOUT:30}
  pool_recycle: ${DB_POOL_RECYCLE:3600}
  
  # SSL settings
  ssl_mode: ${DB_SSL_MODE:prefer}
  ssl_cert: ${DB_SSL_CERT}
  ssl_key: ${DB_SSL_KEY}
  ssl_ca: ${DB_SSL_CA}
  
  # Performance settings
  statement_timeout: ${DB_STATEMENT_TIMEOUT:30000}
  query_timeout: ${DB_QUERY_TIMEOUT:30000}
  
  # Retry settings
  max_retries: ${DB_MAX_RETRIES:3}
  retry_delay: ${DB_RETRY_DELAY:1}
```

### Database Schema

```sql
-- schema.sql
-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_ledgers_sequence ON ledgers(sequence);
CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions(created_at);
CREATE INDEX IF NOT EXISTS idx_accounts_account_id ON accounts(account_id);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_node_id ON graph_nodes(node_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_source_target ON graph_edges(source, target);

-- Partition large tables by time
CREATE TABLE transactions_partitioned (
    LIKE transactions INCLUDING ALL
) PARTITION BY RANGE (created_at);

-- Create partitions
CREATE TABLE transactions_2024_q1 PARTITION OF transactions_partitioned
FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

CREATE TABLE transactions_2024_q2 PARTITION OF transactions_partitioned
FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');
```

### Database Migration

```python
# migrations/migration_manager.py
from alembic import command
from astroml.config import get_config

class MigrationManager:
    """Manage database migrations."""
    
    def __init__(self):
        self.config = get_config()
    
    def upgrade_database(self, revision="head"):
        """Upgrade database to latest revision."""
        command.upgrade(revision, f"postgresql://{self.config.database.url}")
    
    def downgrade_database(self, revision):
        """Downgrade database to specific revision."""
        command.downgrade(revision, f"postgresql://{self.config.database.url}")
    
    def create_migration(self, message):
        """Create new migration."""
        command.revision(message=message)

# Usage
manager = MigrationManager()
manager.upgrade_database()
```

## Stellar Network Configuration

### Network Settings

```yaml
# config/stellar.yaml
stellar:
  # Network selection
  network: ${STELLAR_NETWORK:testnet}
  
  # Horizon API endpoints
  horizon_urls:
    mainnet: "https://horizon.stellar.org"
    testnet: "https://horizon-testnet.stellar.org"
    futurenet: "https://horizon-futurenet.stellar.org"
  
  # Rate limiting
  rate_limit:
    requests_per_second: ${STELLAR_RPS:100}
    burst_size: ${STELLAR_BURST:200}
    retry_attempts: ${STELLAR_RETRIES:3}
    retry_delay: ${STELLAR_RETRY_DELAY:1}
  
  # Data ingestion
  ingestion:
    batch_size: ${STELLAR_BATCH_SIZE:100}
    max_concurrent: ${STELLAR_MAX_CONCURRENT:10}
    timeout: ${STELLAR_TIMEOUT:30}
    
  # Supported assets
  supported_assets:
    - code: "XLM"
      issuer: null
      type: "native"
    - code: "USDC"
      issuer: "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335XOP3IA2M65BZDCCXN2YRC2TH"
      type: "credit_alphanum4"
    - code: "EURT"
      issuer: "GAP5LEFEQO2NDFVZAQ3JGMGH2X23Y3SUXFMXZLKEGHI4QJ5MTIGY"
      type: "credit_alphanum4"
```

### Network Validation

```python
# stellar/network_validator.py
from stellar_sdk import Server
from astroml.config import get_config

class NetworkValidator:
    """Validate Stellar network configuration."""
    
    def __init__(self):
        self.config = get_config()
    
    def validate_connection(self):
        """Validate connection to Stellar Horizon."""
        try:
            server = Server(horizon_url=self.config.stellar.horizon_urls[self.config.stellar.network])
            
            # Test basic connectivity
            root = server.root()
            print(f"Connected to {self.config.stellar.network}")
            print(f"Horizon version: {root['horizon_version']}")
            print(f"Stellar core version: {root['core_version']}")
            
            # Test ledger endpoint
            latest_ledger = server.ledgers().limit(1).call()['_embedded']['records'][0]
            print(f"Latest ledger: {latest_ledger['sequence']}")
            
            return True
            
        except Exception as e:
            print(f"Failed to connect to Stellar: {e}")
            return False
    
    def validate_assets(self):
        """Validate supported asset configuration."""
        server = Server(horizon_url=self.config.stellar.horizon_urls[self.config.stellar.network])
        
        for asset in self.config.stellar.supported_assets:
            try:
                if asset['type'] == 'native':
                    continue  # XLM is always valid
                
                # Validate asset exists
                asset_response = server.assets().for_code(asset['code']).limit(1).call()
                
                if not asset_response['_embedded']['records']:
                    print(f"Warning: Asset {asset['code']} not found on network")
                else:
                    print(f"✓ Asset {asset['code']} validated")
                    
            except Exception as e:
                print(f"Error validating asset {asset['code']}: {e}")

# Usage
validator = NetworkValidator()
validator.validate_connection()
validator.validate_assets()
```

## Model Configuration

### Model Settings

```yaml
# config/models.yaml
models:
  # Default model configurations
  gcn:
    input_dim: ${GCN_INPUT_DIM:64}
    hidden_dims: ${GCN_HIDDEN_DIMS:[128, 64]}
    output_dim: ${GCN_OUTPUT_DIM:2}
    dropout: ${GCN_DROPOUT:0.5}
    learning_rate: ${GCN_LEARNING_RATE:0.001}
    weight_decay: ${GCN_WEIGHT_DECAY:1e-5}
    
  temporal_gcn:
    input_dim: ${TEMPORAL_GCN_INPUT_DIM:64}
    hidden_dims: ${TEMPORAL_GCN_HIDDEN_DIMS:[128, 64]}
    output_dim: ${TEMPORAL_GCN_OUTPUT_DIM:2}
    temporal_dim: ${TEMPORAL_GCN_TEMPORAL_DIM:32}
    dropout: ${TEMPORAL_GCN_DROPOUT:0.5}
    learning_rate: ${TEMPORAL_GCN_LEARNING_RATE:0.001}
    time_encoding: ${TEMPORAL_GCN_TIME_ENCODING:sinusoidal}
    
  anomaly_detector:
    method: ${ANOMALY_METHOD:autoencoder}
    threshold: ${ANOMALY_THRESHOLD:0.95}
    model_type: ${ANOMALY_MODEL_TYPE:gcn}
    feature_dim: ${ANOMALY_FEATURE_DIM:64}
    hidden_dims: ${ANOMALY_HIDDEN_DIMS:[128, 64]}
    learning_rate: ${ANOMALY_LEARNING_RATE:0.001}
    
  # Model registry
  registry:
    default_model: ${DEFAULT_MODEL:gcn}
    auto_save: ${MODEL_AUTO_SAVE:true}
    model_dir: ${MODEL_DIR:./models}
    checkpoint_interval: ${CHECKPOINT_INTERVAL:10}
```

### Model Registry

```python
# models/model_registry.py
import torch
import importlib
from astroml.config import get_config

class ModelRegistry:
    """Registry for managing trained models."""
    
    def __init__(self):
        self.config = get_config()
        self.models = {}
        self.load_models()
    
    def register_model(self, name, model, metadata=None):
        """Register a model in the registry."""
        self.models[name] = {
            "model": model,
            "metadata": metadata or {},
            "created_at": datetime.now()
        }
        
        if self.config.models.auto_save:
            self.save_model(name, model)
    
    def get_model(self, name):
        """Get a model from the registry."""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found in registry")
        
        return self.models[name]["model"]
    
    def save_model(self, name, model):
        """Save a model to disk."""
        import os
        os.makedirs(self.config.models.model_dir, exist_ok=True)
        
        model_path = os.path.join(self.config.models.model_dir, f"{name}.pkl")
        torch.save(model.state_dict(), model_path)
    
    def load_model(self, name, model_class):
        """Load a model from disk."""
        import os
        model_path = os.path.join(self.config.models.model_dir, f"{name}.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model instance
        if name == "gcn":
            model = model_class(**self.config.models.gcn.__dict__)
        elif name == "temporal_gcn":
            model = model_class(**self.config.models.temporal_gcn.__dict__)
        else:
            raise ValueError(f"Unknown model type: {name}")
        
        # Load state dict
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        return model

# Usage
registry = ModelRegistry()
model = registry.get_model("gcn")
```

## Deployment Options

### Local Development

```bash
# Development server
python -m astroml.api.server --env development --port 8000

# With custom config
python -m astroml.api.server --config config/local.yaml --port 8000

# Background process
nohup python -m astroml.api.server --env production > logs/api.log 2>&1 &
```

### Cloud Deployment Options

#### AWS EC2

```yaml
# deployment/aws/ec2-user-data.sh
#!/bin/bash
# User data script for EC2 instance

# Update system
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip postgresql-client

# Install AstroML
sudo -u ec2-user bash -c "
cd /home/ec2-user
git clone https://github.com/Traqora/astroml.git
cd astroml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
"

# Set up environment
cat > /home/ec2-user/.env << EOF
ASTROML_ENV=production
DATABASE_URL=postgresql://user:pass@localhost/astroml
STELLAR_NETWORK=mainnet
EOF

# Start services
sudo systemctl enable astroml
sudo systemctl start astroml
```

#### Google Cloud Platform

```yaml
# deployment/gcp/app.yaml
runtime: python39
entrypoint: gunicorn -b :$PORT astroml.api.server:app
env_variables:
  ASTROML_ENV: production
  DATABASE_URL: ${DATABASE_URL}
  STELLAR_NETWORK: mainnet

resources:
  cpu: 2
  memory_gb: 4
  disk_size_gb: 20

automatic_scaling:
  min_num_instances: 1
  max_num_instances: 10
  cpu_utilization:
    target: 70
```

#### Azure Container Instances

```yaml
# deployment/azure/container-instance.yaml
apiVersion: 2021-03-01
type: Microsoft.ContainerInstance/containerGroups
name: astroml-ci
location: eastus
properties:
  containers:
  - name: astroml
    properties:
      image: astroml:latest
      ports:
      - port: 8000
      environmentVariables:
      - name: ASTROML_ENV
        value: production
      - name: DATABASE_URL
        value: $(DATABASE_URL)
      resources:
        requests:
          cpu: 2.0
          memoryInGb: 4.0
```

## Docker Deployment

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY astroml/ ./astroml/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create non-root user
RUN useradd -m -u 1000 astroml
USER astroml

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import astroml; print('OK')" || exit 1

# Expose port
EXPOSE 8000

# Start command
CMD ["python", "-m", "astroml.api.server"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  astroml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ASTROML_ENV=production
      - DATABASE_URL=postgresql://astroml:password@postgres:5432/astroml
      - STELLAR_NETWORK=mainnet
    depends_on:
      - postgres
      - redis
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=astroml
      - POSTGRES_USER=astroml
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - astroml-api
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### Multi-Stage Dockerfile

```dockerfile
# Dockerfile.multi-stage
# Build stage
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y gcc g++

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application
COPY astroml/ ./astroml/
COPY config/ ./config/

# Create user
RUN useradd -m -u 1000 astroml
USER astroml

EXPOSE 8000
CMD ["python", "-m", "astroml.api.server"]
```

## Kubernetes Deployment

### Namespace and ConfigMaps

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: astroml
---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: astroml-config
  namespace: astroml
data:
  database.yaml: |
    host: postgres
    port: 5432
    name: astroml
    user: astroml
  stellar.yaml: |
    network: mainnet
    horizon_url: https://horizon.stellar.org
```

### Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: astroml-api
  namespace: astroml
spec:
  replicas: 3
  selector:
    matchLabels:
      app: astroml-api
  template:
    metadata:
      labels:
        app: astroml-api
    spec:
      containers:
      - name: astroml-api
        image: astroml:latest
        ports:
        - containerPort: 8000
        env:
        - name: ASTROML_ENV
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: astroml-secrets
              key: database-url
        - name: STELLAR_NETWORK
          value: "mainnet"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: models
          mountPath: /app/models
      volumes:
      - name: config
        configMap:
          name: astroml-config
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
```

### Service and Ingress

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: astroml-api-service
  namespace: astroml
spec:
  selector:
    app: astroml-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: astroml-ingress
  namespace: astroml
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.astroml.io
    secretName: astroml-tls
  rules:
  - host: api.astroml.io
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: astroml-api-service
            port:
              number: 80
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: astroml-hpa
  namespace: astroml
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: astroml-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Monitoring and Logging

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'astroml'
    static_configs:
      - targets: ['astroml-api-service:8000']
    metrics_path: /metrics
    scrape_interval: 5s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Alert Rules

```yaml
# monitoring/alert_rules.yml
groups:
  - name: astroml
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: ModelAccuracyDrop
        expr: astroml_model_accuracy < 0.8
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Model accuracy dropped below threshold"
          description: "Model accuracy is {{ $value }}"

      - alert: IngestionBacklog
        expr: astroml_ingestion_backlog_size > 10000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Ingestion backlog detected"
          description: "Backlog size is {{ $value }} ledgers"
```

### Logging Configuration

```yaml
# config/logging.yaml
logging:
  version: 1
  disable_existing_loggers: false
  
  formatters:
    default:
      format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    json:
      format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
      class: pythonjsonlogger.jsonlogger.JsonFormatter
    
    handlers:
      console:
        class: logging.StreamHandler
        level: INFO
        formatter: default
        stream: ext://sys.stdout
      
      file:
        class: logging.handlers.RotatingFileHandler
        level: INFO
        formatter: json
        filename: logs/astroml.log
        maxBytes: 10485760  # 10MB
        backupCount: 5
      
      sentry:
        class: sentry_sdk.integrations.logging.SentryHandler
        level: ERROR
        dsn: ${SENTRY_DSN}
    
    loggers:
      astroml:
        level: INFO
        handlers: [console, file, sentry]
        propagate: false
      
      uvicorn:
        level: INFO
        handlers: [console, file]
        propagate: false
      
      sqlalchemy:
        level: WARNING
        handlers: [console, file]
        propagate: false
```

## Security Configuration

### Security Headers

```python
# security/middleware.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

def setup_security_middleware(app: FastAPI):
    """Setup security middleware for FastAPI app."""
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://api.astroml.io"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    # Trusted hosts
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["api.astroml.io", "*.astroml.io"]
    )
    
    # HTTPS redirect
    if os.getenv("FORCE_HTTPS", "false").lower() == "true":
        app.add_middleware(HTTPSRedirectMiddleware)
    
    # Security headers
    @app.middleware("http")
    async def add_security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response
```

### API Rate Limiting

```python
# security/rate_limiter.py
from slowapi import Limiter, _rate
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

@app.get("/api/v1/ingest")
@limiter.limit("100/minute")
async def ingest_data():
    """Rate limited ingestion endpoint."""
    pass

@app.get("/api/v1/predict")
@limiter.limit("1000/minute")
async def predict():
    """Higher rate limit for predictions."""
    pass

# Custom rate limiter for authenticated users
@app.get("/api/v1/admin")
@limiter.limit("10/minute")
async def admin_endpoint():
    """Strict rate limit for admin endpoints."""
    pass
```

### Environment Variables Security

```bash
# .env.production
# Database
DATABASE_URL=postgresql://user:secure_password@localhost/astroml_prod
DB_SSL_MODE=require

# Stellar
STELLAR_NETWORK=mainnet
STELLAR_WEBHOOK_SECRET=your_webhook_secret

# API Security
API_SECRET_KEY=your_very_secure_secret_key_here
JWT_SECRET_KEY=your_jwt_secret_key_here
ENCRYPTION_KEY=your_32_character_encryption_key

# Monitoring
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id

# External Services
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
```

---

This comprehensive configuration and deployment guide provides everything needed to deploy AstroML in production environments, from local development to cloud deployments with proper monitoring, logging, and security configurations.
