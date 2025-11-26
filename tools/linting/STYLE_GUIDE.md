# QUANTUM PLATFORM - CODING STYLE GUIDE

## 1. Language + Versions
    - Primary language: Python 3.12+.
    - Required typing: mypy strict mode.
    - All modules must run 'black', 'isort' and 'flake8'.
    - Probably future implementation in C++.

## 2. Formatting
    - Variables must be named with lowercase letters and '_' between words.
    - Tab is 4 spaces.
    - Code formatting is enforced by:
        - black (line length: 88).
        - isort (profile: black).


## 3. Imports
    - Absolute imports only.
    - Standard library → third-party → internal modules.

## 4. Typing
    - All funcions must include full type hints.
    - Use `typing` and `typing_extensions` features (TypedDict, Protocol, etc.)
    - No `Any` unless justified in a docstring.

## 5. Logging
    - Use only structured logging via the Python `logging` library.
    - No print() calls in production modules.

## 6. Testing Conventions
    - All tests go in 'tests/' with 'pytest'.
    - Test filenames: 'test_<module>.py'
    - Every PR must include tests for new logic.

## 7. Documentation
    - Every module requires a top-level docstring.
    - Public functions/classes must have docstrings (Google style).

## 8. Commit messages
    - Format: `<type>: <short description>`
    - Types: feat, fix, docs, refactor, test, chore.

## 9. CI Linting
    - The CI pipeline will reject:
        - formatting violations
        - missing type hints
        - unused imports
        - unreachable code


