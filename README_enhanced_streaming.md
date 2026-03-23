# Enhanced Stellar Ingestion Service

A robust ingestion service using the Stellar SDK to stream effects and operations with comprehensive error handling for rate limits and connection drops.

## Features

### 🚀 Robust Error Handling
- **Rate Limiting**: Adaptive backoff with exponential increase and configurable limits
- **Connection Drops**: Automatic detection and recovery with health monitoring
- **Retry Logic**: Configurable retry attempts with exponential backoff
- **Graceful Shutdown**: Clean shutdown on SIGINT/SIGTERM

### 📈 Performance & Monitoring
- **Batch Processing**: Configurable batch sizes for optimal throughput
- **Health Monitoring**: Continuous connection health checks
- **Statistics Tracking**: Real-time metrics and performance data
- **State Persistence**: Automatic cursor saving for resume capability

### 🔧 Flexible Configuration
- **Multi-Horizon Support**: Stream from multiple Horizon instances simultaneously
- **Multiple Stream Types**: Effects and operations streaming
- **Custom Timeouts**: Configurable connection and stream timeouts
- **Environment-based Config**: Easy deployment configuration

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Key dependencies for enhanced streaming:
# - stellar-sdk>=9.0.0
# - tenacity>=8.4.0
# - aiohttp>=3.9
```

## Quick Start

### Single Stream

```bash
# Stream effects from testnet
python -m astroml.ingestion.enhanced_cli --stream-type effects --horizon testnet

# Stream operations from mainnet with custom cursor
python -m astroml.ingestion.enhanced_cli --stream-type operations --horizon mainnet --cursor 123456789

# Resume from saved state
python -m astroml.ingestion.enhanced_cli --stream-type effects --resume
```

### Multi-Horizon Streaming

```bash
# Stream from both testnet and mainnet
python -m astroml.ingestion.enhanced_cli --multi --horizons testnet,mainnet --streams effects,operations

# Custom Horizon URLs
python -m astroml.ingestion.enhanced_cli --multi --horizons "https://horizon-testnet.stellar.org,https://custom-horizon.example.com" --streams effects
```

## Configuration

### Command Line Options

```bash
# Basic configuration
--stream-type [effects|operations]     # Stream type (default: effects)
--horizon [testnet|mainnet]           # Horizon network (default: testnet)
--cursor CURSOR                       # Starting cursor
--resume                              # Resume from saved state

# Error handling
--max-retries INT                     # Max retry attempts (default: 5)
--base-retry-delay FLOAT              # Base retry delay in seconds (default: 1.0)
--max-retry-delay FLOAT               # Max retry delay in seconds (default: 60.0)
--rate-limit-backoff FLOAT            # Rate limit backoff (default: 5.0)

# Performance tuning
--batch-size INT                      # Batch size (default: 100)
--batch-timeout FLOAT                 # Batch timeout (default: 5.0)
--connection-timeout FLOAT           # Connection timeout (default: 30.0)
--stream-timeout FLOAT                # Stream timeout (default: 60.0)

# Monitoring
--health-check-interval FLOAT         # Health check interval (default: 30.0)
--log-level [DEBUG|INFO|WARNING|ERROR] # Logging level (default: INFO)
```

### Programmatic Usage

```python
import asyncio
from astroml.ingestion.enhanced_stream import EnhancedStellarStream, EnhancedStreamConfig

async def main():
    config = EnhancedStreamConfig(
        horizon_url="https://horizon-testnet.stellar.org",
        stream_type="effects",
        max_retries=5,
        batch_size=100,
        rate_limit_backoff=5.0
    )
    
    async with EnhancedStellarStream(config) as stream:
        await stream.run()
        
        # Get statistics
        stats = stream.get_stats()
        print(f"Processed {stats['processed_count']} records")

asyncio.run(main())
```

## Architecture

### Core Components

#### EnhancedStellarStream
- Main streaming client using Stellar SDK
- Handles effects and operations streaming
- Implements retry logic and error recovery

#### RateLimitTracker
- Tracks request rates and rate limit events
- Implements adaptive backoff strategies
- Prevents overwhelming the Horizon API

#### ConnectionHealthMonitor
- Monitors connection health and detects drops
- Tracks consecutive failures
- Triggers recovery mechanisms

#### StreamService
- High-level service for managing multiple streams
- Provides graceful shutdown and monitoring
- Handles state persistence

### Error Handling Strategy

1. **Rate Limits**: 
   - Detect HTTP 429 responses
   - Calculate backoff based on `Retry-After` header
   - Implement exponential increase for repeated limits

2. **Connection Drops**:
   - Detect timeouts and connection errors
   - Implement health checks with configurable intervals
   - Use exponential backoff for reconnection

3. **Server Errors**:
   - Retry on 5xx errors with exponential backoff
   - Stop after configured max retries
   - Log detailed error information

4. **Client Errors**:
   - Fail fast on 4xx errors (except 429)
   - Validate configuration before starting
   - Provide clear error messages

### Database Schema

The enhanced service adds support for the `effects` table:

```sql
CREATE TABLE effects (
    id BIGINT PRIMARY KEY,
    account VARCHAR(56) NOT NULL,
    type VARCHAR(32) NOT NULL,
    amount NUMERIC,
    asset_code VARCHAR(12),
    asset_issuer VARCHAR(56),
    destination_account VARCHAR(56),
    created_at TIMESTAMP NOT NULL,
    details JSONB
);

-- Indexes for performance
CREATE INDEX ix_effects_account_created_at ON effects(account, created_at);
CREATE INDEX ix_effects_type_created_at ON effects(type, created_at);
CREATE INDEX ix_effects_destination_created_at ON effects(destination_account, created_at);
```

## Monitoring & Statistics

### Real-time Metrics

```python
stats = stream.get_stats()
print(f"Cursor: {stats['cursor']}")
print(f"Processed: {stats['processed_count']}")
print(f"Request Rate: {stats['request_rate']:.2f} req/s")
print(f"Healthy: {stats['is_healthy']}")
print(f"Failures: {stats['consecutive_failures']}")
print(f"Current Backoff: {stats['current_backoff']:.1f}s")
```

### Health Checks

The service performs regular health checks:

- **Server Responsiveness**: Periodic `root` endpoint calls
- **Connection Freshness**: Track last successful request
- **Error Rate**: Monitor consecutive failures
- **Rate Limit Status**: Track recent rate limit events

## Testing

```bash
# Run tests
python -m pytest astroml/ingestion/tests/test_enhanced_stream.py -v

# Run with coverage
python -m pytest astroml/ingestion/tests/ --cov=astroml.ingestion.enhanced_stream
```

## Deployment

### Environment Variables

```bash
export ASTROML_HORIZON_URL="https://horizon-testnet.stellar.org"
export ASTROML_STREAM_ENDPOINT="/effects"
export ASTROML_STREAM_CURSOR="123456789"
export ASTROML_LOG_LEVEL="INFO"
```

### Docker Example

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY astroml/ ./astroml/
CMD ["python", "-m", "astroml.ingestion.enhanced_cli", "--stream-type", "effects"]
```

### Systemd Service

```ini
[Unit]
Description=Enhanced Stellar Ingestion Service
After=network.target

[Service]
Type=simple
User=astroml
WorkingDirectory=/opt/astroml
ExecStart=/opt/astroml/venv/bin/python -m astroml.ingestion.enhanced_cli --stream-type effects --log-level INFO
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Troubleshooting

### Common Issues

1. **Rate Limiting**: 
   - Increase `--rate-limit-backoff`
   - Reduce `--batch-size`
   - Use multiple Horizon instances

2. **Connection Drops**:
   - Increase `--connection-timeout`
   - Reduce `--health-check-interval`
   - Check network connectivity

3. **Memory Usage**:
   - Reduce `--batch-size`
   - Monitor database connections
   - Check for memory leaks

### Debug Mode

```bash
# Enable debug logging
python -m astroml.ingestion.enhanced_cli --log-level DEBUG --stream-type effects

# Monitor specific components
python -m astroml.ingestion.enhanced_cli --log-level DEBUG --health-check-interval 5.0
```

## Performance Tips

1. **Batch Size**: Start with 100, adjust based on network conditions
2. **Timeouts**: Set connection timeout to 30s, stream timeout to 60s
3. **Rate Limiting**: Use 5.0s backoff for testnet, 10.0s for mainnet
4. **Health Checks**: 30s intervals balance monitoring and overhead

## Comparison with Original Stream

| Feature | Original | Enhanced |
|---------|----------|----------|
| SDK | aiohttp-sse-client | Stellar SDK |
| Rate Limiting | Basic | Adaptive with backoff |
| Connection Health | None | Comprehensive monitoring |
| Retry Logic | Simple | Exponential with max limits |
| Multi-Horizon | No | Yes |
| Statistics | Basic | Detailed metrics |
| State Persistence | File-based | Enhanced state manager |
| Testing | Limited | Comprehensive test suite |

## License

This enhanced streaming service maintains the same license as the original AstroML project.
