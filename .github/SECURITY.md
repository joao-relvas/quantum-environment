# Security Policy

## Supported Versions
Only the versions listed below receive security updates.

| Version | Supported |
|--------|-----------|
| main    | ✔️ |
| dev     | ❌ (development only) |

If you are using any fork, custom build, or older tag, you are responsible for your own security updates.

---

## Reporting a Vulnerability

If you discover a security vulnerability, **do NOT create a public issue**.

Instead, report it privately:´
    - email: relvasjv@gmail.com

Please include:
- a clear description of the issue  
- steps to reproduce  
- potential impact  
- suggested fix (if any)

---

## Security Best Practices for Contributors

- Never commit secrets, tokens, passwords, API keys, or credentials.  
- Do not bypass code review for changes that affect:
  - authentication
  - authorization
  - encryption
  - secrets handling
  - key management
- Run `pre-commit` checks before every PR.
- Use GitHub’s Secret Scanning and Dependabot alerts.

---

## Security Hardening Notes

This project enforces:
- CI pipeline verification on every PR  
- Required reviews before merging to `main`  
- Automatic dependency scanning  
- No direct commits to `main`  
- Automated builds in isolated environments  

---
