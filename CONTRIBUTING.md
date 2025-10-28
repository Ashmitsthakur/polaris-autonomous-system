# Contributing to Polaris Autonomous System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- ROS2 (Humble or later)
- Git

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/valid_monke/polaris-autonomous-system.git
cd polaris-autonomous-system

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## 🧪 Running Tests

Before submitting any changes, ensure all tests pass:

```bash
# Run unit tests
python tests/unit_tests.py

# Run validation framework
python main.py --validate

# Run EKF validation
python scripts/validate_ekf.py
```

## 📝 Code Style

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep lines under 100 characters when possible

## 🔧 Development Workflow

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/my-new-feature`
3. **Make your changes**
4. **Run tests**: Ensure all tests pass
5. **Commit your changes**: `git commit -am 'Add some feature'`
6. **Push to the branch**: `git push origin feature/my-new-feature`
7. **Submit a pull request**

## 📊 Testing Guidelines

### Unit Tests
- Add unit tests for all new functions
- Place tests in `tests/unit_tests.py`
- Ensure test coverage for edge cases

### Integration Tests
- Test complete pipeline workflows
- Validate with real sensor data when possible

### Performance Tests
- Ensure changes don't degrade performance
- Target: 30 Hz processing rate maintained

## 🐛 Bug Reports

When reporting bugs, include:
- Python version
- ROS2 distribution
- Steps to reproduce
- Expected vs actual behavior
- Error messages and logs

## 💡 Feature Requests

When requesting features:
- Describe the use case
- Explain why existing features don't meet the need
- Consider implementation complexity

## 📚 Documentation

- Update README.md if adding user-facing features
- Add docstrings to all new code
- Update relevant documentation in `docs/`

## 🤝 Code Review Process

All submissions require review. The review process:
1. Automated tests must pass
2. Code style must be consistent
3. Documentation must be updated
4. At least one maintainer approval required

## 📄 License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## 🙏 Questions?

Feel free to open an issue for any questions or clarifications.

