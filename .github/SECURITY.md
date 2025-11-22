# Security Policy

## Supported Versions

Currently, we are in early development. Security updates will be provided for:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take the security of EEG-RAG seriously. If you discover a security vulnerability, please follow these steps:

### 1. Do Not Disclose Publicly

Please **do not** create a public GitHub issue for security vulnerabilities.

### 2. Contact Us Privately

Report security vulnerabilities through GitHub's Security Advisory feature:

1. Navigate to the repository's Security tab
2. Click "Report a vulnerability"
3. Fill out the vulnerability report form

Alternatively, you can email security concerns to the project maintainers.

### 3. Provide Details

Include as much information as possible:

- Type of vulnerability
- Affected component/module
- Steps to reproduce
- Potential impact
- Suggested fix (if you have one)

### 4. Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: 1-7 days
  - High: 7-14 days
  - Medium: 14-30 days
  - Low: Best effort

## Security Best Practices

### For Users

1. **API Keys**: Never commit API keys or secrets to version control
2. **Environment Variables**: Use `.env` files (not committed to git)
3. **Docker Security**: Don't run containers as root
4. **Dependencies**: Keep dependencies updated
5. **Access Control**: Limit access to production systems

### For Developers

1. **Code Review**: All PRs must be reviewed before merging
2. **Dependency Scanning**: Use automated tools to check dependencies
3. **Input Validation**: Validate all user inputs
4. **Error Handling**: Don't expose sensitive information in errors
5. **Logging**: Don't log sensitive data (API keys, passwords, PII)

## Known Security Considerations

### Data Privacy

- EEG-RAG processes scientific literature, which may contain sensitive information
- Ensure compliance with data protection regulations when using with proprietary data
- Be cautious when sharing query logs or cached results

### API Security

- OpenAI API keys should be stored securely in environment variables
- Implement rate limiting to prevent abuse
- Use HTTPS for all external communications

### Docker Considerations

- Virtual environments should be created inside Docker containers, not in the root directory
- Use official base images from trusted sources
- Regularly update container images
- Don't expose unnecessary ports

### Neo4j & Redis

- Use authentication for database connections
- Don't use default passwords in production
- Restrict network access to database ports
- Encrypt data in transit

## Vulnerability Disclosure Policy

If we receive a security vulnerability report, we will:

1. Confirm receipt within 48 hours
2. Investigate and validate the vulnerability
3. Develop and test a fix
4. Release a security patch
5. Credit the reporter (if desired)
6. Publish a security advisory

## Security Updates

Security updates will be announced through:

- GitHub Security Advisories
- Release notes
- CHANGELOG.md

Subscribe to repository notifications to stay informed.

## Responsible Disclosure

We kindly ask security researchers to:

- Give us reasonable time to fix vulnerabilities before public disclosure
- Make a good faith effort to avoid privacy violations, data destruction, and service interruption
- Not exploit vulnerabilities beyond what's necessary to demonstrate the issue

We commit to:

- Respond promptly to vulnerability reports
- Keep reporters informed of progress
- Credit researchers who report vulnerabilities responsibly
- Not take legal action against researchers who follow this policy

## Questions?

If you have questions about this security policy, please open a discussion on GitHub or contact the maintainers.

Thank you for helping keep EEG-RAG and our users safe!
