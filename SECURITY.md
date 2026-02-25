# Security Policy

## Supported Versions

Only the latest stable release receives security fixes. Older versions will not receive backported patches.

| Version | Supported |
| ------- | --------- |
| 1.0.x (latest) | Yes |
| < 1.0.0 | No |

If you are running an older release, please update to the current stable release before reporting an issue, as it may already be resolved.

---

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

If you believe you have found a security vulnerability in PaintFE, please report it privately. Disclosing security issues publicly before a fix is available puts other users at risk.

**How to report:**

Open a [GitHub Security Advisory](https://github.com/kylejckson/paintfe/security/advisories/new) on this repository. This channel is private and only visible to the repository maintainers until a fix is published.

Please include as much of the following as you can:

- A clear description of the vulnerability and what it allows an attacker to do
- The component or feature affected (file parser, scripting engine, AI integration, etc.)
- Steps to reproduce, including any test files, scripts, or inputs required
- The version of PaintFE you tested against
- Your operating system and architecture
- Any suggested mitigations or patches you may have in mind

You will receive an acknowledgment within **5 business days**. If you do not hear back, feel free to follow up by mentioning it in the advisory thread.

---

## Disclosure Process

PaintFE follows a coordinated disclosure model.

1. You report privately via the Security Advisory channel.
2. The maintainer triages the report and confirms whether it is a valid security issue.
3. A fix is developed and tested, targeting a patch release.
4. A release is published and a GitHub Security Advisory is made public at the same time.
5. Credit is given in the advisory and release notes unless you request to remain anonymous.

The target timeline from confirmed report to public disclosure is **30 days** for most issues. Complex or widespread issues may require more time and will be communicated to the reporter.

---

## Scope

The following are considered in scope for security reports:

- **File parsing vulnerabilities** -- malformed or crafted image files (JPEG, PNG, TIFF, WebP, BMP, GIF, APNG, RAW formats) that cause crashes, memory corruption, or arbitrary code execution when opened in PaintFE
- **Project file vulnerabilities** -- crafted `.pfe` project files that cause unsafe deserialization, out-of-bounds memory access, or code execution when loaded
- **Scripting sandbox escapes** -- Rhai scripts that escape the sandboxed execution environment, access the filesystem, execute system commands, or exfiltrate data beyond what the API intentionally exposes
- **AI / ONNX integration** -- path traversal or arbitrary library loading vulnerabilities in the ONNX Runtime integration path
- **Clipboard handling** -- malformed clipboard payloads that cause memory corruption or unexpected behavior when pasting into the editor
- **GPU pipeline** -- shader or buffer handling bugs that could be triggered by crafted inputs to cause out-of-bounds GPU memory access or driver-level crashes
- **CLI batch mode** -- glob expansion or script execution in headless mode that allows unintended filesystem access or privilege escalation

---

## Out of Scope

The following are not considered security vulnerabilities for the purpose of this policy:

- Crashes caused by legitimate but very large input files (running out of memory is expected behavior)
- Bug reports about missing features or incorrect output from filters and adjustments
- Issues in third-party dependencies that do not affect PaintFE specifically -- please report those upstream
- Security issues in external tools such as ONNX Runtime models supplied by the user (PaintFE does not bundle any models)
- Social engineering or phishing attacks unrelated to the software itself
- Issues that require physical access to a logged-in machine already controlled by an attacker

---

## Security Considerations for Users

### ONNX Runtime and AI Models

The AI background removal feature uses a user-supplied ONNX Runtime library (`onnxruntime.dll` or `libonnxruntime.so`) and a user-supplied model file. PaintFE loads the runtime library dynamically at the path you specify in Preferences. **Only point PaintFE at ONNX Runtime libraries and model files you obtained from trusted sources.** A malicious dynamic library placed at that path would be executed with the same privileges as PaintFE.

### Rhai Scripting

The built-in script editor executes Rhai scripts in a sandboxed environment with operation limits, call depth limits, and no direct filesystem or network access. However, the sandbox restricts the scripting API, not the underlying process. Do not run scripts obtained from untrusted sources. The Edit > Preferences panel does not execute scripts automatically; they must be explicitly triggered by the user.

### Opening Untrusted Files

PaintFE uses several third-party image decoding libraries. While these libraries are widely used and actively maintained, no image parser is immune to bugs. Exercise the same caution you would with any application when opening image files from unknown sources.

### Portable Binary

PaintFE ships as a single portable binary with no installer. Verify the SHA-256 checksum of the binary against the value published on the GitHub Release page before running a newly downloaded copy.

---

## Credit

Researchers who responsibly disclose valid security issues will be credited by name (or handle, or anonymously on request) in the GitHub Security Advisory and in the release notes for the patch that resolves the issue.

Thank you for helping keep PaintFE and its users secure.
