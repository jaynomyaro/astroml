# Audit Trail Documentation

## Overview

The AstroML audit trail system provides comprehensive logging of sensitive API operations for security audits and compliance (issues #332, #535).

## Features

### Enhanced Audit Logging (Issue #535)

- **Request Parameter Logging**: Captures and sanitizes request parameters
- **Sensitive Data Redaction**: Automatically redacts sensitive fields (passwords, tokens, API keys)
- **API Key Tracking**: Tracks which API key was used for each request
- **IP Address Logging**: Captures client IP addresses with proxy support
- **User-Agent Logging**: Records client user-agent strings
- **Tamper-Resistant**: Append-only database storage prevents modification
- **90-Day Retention**: Automatic cleanup of logs older than 90 days

### Logged Information

Each audit log entry includes:

- **Timestamp**: ISO 8601 format (UTC)
- **User Identity**: User ID and username
- **Authentication Type**: How the user authenticated (api_key, session, etc.)
- **API Key ID**: Which API key was used (if applicable)
- **Endpoint and Method**: Request path and HTTP method
- **Request Parameters**: Sanitized query and body parameters
- **Response Status**: HTTP status code
- **IP Address**: Client IP address (with proxy support)
- **User-Agent**: Client user-agent string

### Sensitive Fields Redacted

The following fields are automatically redacted from audit logs:

- `password`
- `token`
- `api_key`
- `secret`
- `credit_card`
- `ssn`
- `social_security`
- `auth`
- `authorization`

## API Endpoints

### Search Audit Logs

```http
GET /api/v1/audit/logs
```

**Query Parameters:**
- `user_id`: Filter by user ID
- `action`: Filter by action (create, update, delete, login, logout)
- `resource_type`: Filter by resource type
- `resource_id`: Filter by resource ID
- `start_date`: Filter by start date
- `end_date`: Filter by end date
- `limit`: Maximum results (default: 100, max: 1000)
- `offset`: Pagination offset

**Required Scope:** `audit:read`

### Export Audit Logs

```http
GET /api/v1/audit/export
```

**Query Parameters:**
- `user_id`: Filter by user ID
- `action`: Filter by action
- `resource_type`: Filter by resource type
- `start_date`: Filter by start date
- `end_date`: Filter by end date

**Required Scope:** `audit:export`

### Rotate Audit Logs

```http
POST /api/v1/audit/rotate
```

Manually trigger deletion of logs older than retention period.

**Required Scope:** `audit:admin`

### Get Audit Statistics

```http
GET /api/v1/audit/stats
```

Returns:
- Total log count
- Retention period
- Maximum records

**Required Scope:** `audit:read`

## Access Control

### Scopes

- `audit:read`: Read audit logs and statistics
- `audit:export`: Export audit logs
- `audit:admin`: Rotate logs and administrative functions

### Privacy Considerations

**Data Minimization:**
- Only sensitive operations are logged
- Sensitive fields are automatically redacted
- Request parameters are sanitized before storage

**Access Restrictions:**
- Audit log access requires specific scopes
- All access is logged in the audit trail itself
- Export functionality requires elevated permissions

**Retention Policy:**
- Logs are retained for 90 days by default
- Automatic cleanup removes old logs
- Manual rotation available for immediate cleanup

**Data Protection:**
- Logs stored in append-only database table
- No direct modification of existing log entries
- IP addresses and user-agents captured for security analysis

## Configuration

### Environment Variables

```bash
# Audit retention period in days (default: 90)
AUDIT_RETENTION_DAYS=90

# Maximum audit log records (default: 1000000)
AUDIT_MAX_RECORDS=1000000
```

### Middleware Configuration

The audit middleware is automatically enabled for sensitive operations:

- POST, PUT, PATCH, DELETE requests
- Authentication endpoints (login, logout)
- User management endpoints
- API key management endpoints

## Security Best Practices

1. **Regular Review**: Review audit logs regularly for suspicious activity
2. **Access Monitoring**: Monitor who accesses audit logs
3. **Retention Compliance**: Ensure retention policy meets compliance requirements
4. **Export Security**: Securely handle exported audit data
5. **Alerting**: Set up alerts for unusual patterns in audit logs

## Example Usage

### Search for User Activity

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "https://api.example.com/api/v1/audit/logs?user_id=123&start_date=2024-01-01"
```

### Export Logs for Compliance

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "https://api.example.com/api/v1/audit/export?start_date=2024-01-01&end_date=2024-01-31" \
  -o audit_export.json
```

### Get Statistics

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "https://api.example.com/api/v1/audit/stats"
```

## Troubleshooting

### Missing Audit Logs

1. Check if the operation type is logged (only sensitive operations)
2. Verify middleware is properly configured
3. Check database connection for audit logger

### Sensitive Data in Logs

1. Ensure sensitive field names match the redaction list
2. Check custom parameter sanitization logic
3. Verify request body parsing is working

### Performance Impact

1. Audit logging adds minimal overhead (<5ms per request)
2. Database writes are asynchronous
3. Failed audit logging does not block requests

## Compliance

This audit trail system helps meet compliance requirements for:

- **SOC 2**: Access logging and monitoring
- **PCI DSS**: Access control and audit trails
- **GDPR**: Data access logging
- **HIPAA**: Audit controls for PHI access
- **ISO 27001**: Access logging and review

## Future Enhancements

- Real-time alerting on suspicious patterns
- Machine learning anomaly detection on audit logs
- Immutable log storage (WORM)
- Blockchain-based audit trail verification
- Advanced search and analytics
