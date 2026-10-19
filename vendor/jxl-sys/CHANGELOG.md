# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.5+libjxl-0.11.1](https://github.com/Zetier/jxl-sys/compare/v0.1.4+libjxl-0.11.1...v0.1.5+libjxl-0.11.1) - 2025-08-01

### Fixed

- use raw string interpolation in regex pattern for version validation

### Other

- *(deps)* update bindgen requirement from 0.71 to 0.72
# Changelog

All notable changes to this project will be documented in this file.

## [unreleased]

## [0.1.11+libjxl-0.11.2](https://github.com/Zetier/jxl-sys/compare/v0.1.10+libjxl-0.11.2...v0.1.11+libjxl-0.11.2) - 2026-07-29

### Other

- use full LTO for libjxl
- *(deps)* bump actions/checkout from 6 to 7

## [0.1.10+libjxl-0.11.2](https://github.com/Zetier/jxl-sys/compare/v0.1.9+libjxl-0.11.2...v0.1.10+libjxl-0.11.2) - 2026-03-19

### Added

- use Ninja if available
- use Ninja if available

### Other

- *(deps)* bump actions/create-github-app-token from 2 to 3
- Use GitHub App token for release-plz workflows
- Remove redundant checkout from rust-common-setup action
- *(deps)* bump actions/checkout from 4 to 6

## [0.1.9+libjxl-0.11.2](https://github.com/Zetier/jxl-sys/compare/v0.1.7+libjxl-0.11.1...v0.1.9+libjxl-0.11.2) - 2026-02-13

### Other

- Add feature flag for lto; add build flags to support lto
- Bump version of libjxl

## [0.1.7+libjxl-0.11.1](https://github.com/Zetier/jxl-sys/compare/v0.1.6+libjxl-0.11.1...v0.1.7+libjxl-0.11.1) - 2026-01-28

### Fixed

- avoid MSVCRT mismatch by forcing Release

### Other

- release v0.1.6+libjxl-0.11.1

## [0.1.6+libjxl-0.11.1](https://github.com/Zetier/jxl-sys/compare/v0.1.5+libjxl-0.11.1...v0.1.6+libjxl-0.11.1) - 2026-01-26

### Fixed

- compilation issues
- format + toolchain specify
- msvc support

### Other

- release v0.1.6+libjxl-0.11.1

## [0.1.4+libjxl-0.11.1](https://github.com/Zetier/jxl-sys/compare/v0.1.3+libjxl-0.11.1...v0.1.4+libjxl-0.11.1) - 2025-05-14

### Fixed

- different directives to link to c++ standard library

## [0.1.3+libjxl-0.11.1](https://github.com/Zetier/jxl-sys/compare/v0.1.2+libjxl-0.11.1...v0.1.3+libjxl-0.11.1) - 2025-05-12

### Fixed

- search lib64 for static libraries

## [0.1.2+libjxl-0.11.1](https://github.com/Zetier/jxl-sys/compare/v0.1.1+libjxl-0.11.1...v0.1.2+libjxl-0.11.1) - 2025-05-05

### Fixed

- build.rs now links against more libraries

## [0.1.1+libjxl-0.11.1](https://github.com/Zetier/jxl-sys/compare/v0.1.0+libjxl-0.11.1...v0.1.1+libjxl-0.11.1) - 2025-04-30

### Fixed

- allow release-plz to release "dirty"
- sort excludes
- use submodules, scope package with excludes

## [0.1.0+libjxl-0.11.1](https://github.com/Zetier/jxl-sys/releases/tag/v0.1.0+libjxl-0.11.1) - 2025-04-29

### Added

- more clarity in building instructions
