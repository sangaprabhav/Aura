# Contributing to Aura - MedGemma Curriculum Learning

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a new branch for your feature/fix
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

1. **Environment Setup**
   ```bash
   git clone https://github.com/sangaprabhav/Aura.git
   cd Aura
   ./setup.sh
   source venv/bin/activate
   ```

2. **Run Tests**
   ```bash
   python test_setup.py
   ```

3. **Verify Installation**
   ```bash
   python -c "import torch; print('PyTorch:', torch.__version__)"
   python -c "from transformers import AutoProcessor; print('Transformers available')"
   ```

## Types of Contributions

### 🐛 Bug Fixes
- Fix training issues
- Resolve memory problems
- Correct data loading errors

### ✨ New Features
- Additional evaluation metrics
- Support for new model architectures
- Enhanced data preprocessing
- New training strategies

### 📚 Documentation
- Improve README
- Add code comments
- Create tutorials
- Fix typos

### 🔧 Performance Improvements
- Memory optimization
- Training speed improvements
- GPU utilization enhancements

## Coding Standards

### Python Code Style
- Follow PEP 8
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and small

### Example Function Documentation
```python
def process_medical_image(image_path: str, target_size: tuple) -> torch.Tensor:
    """
    Process a medical image for model input.
    
    Args:
        image_path: Path to the medical image file
        target_size: Tuple of (height, width) for resizing
        
    Returns:
        torch.Tensor: Preprocessed image tensor
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be processed
    """
```

## Testing Guidelines

### Before Submitting
1. **Run the test suite**
   ```bash
   python test_setup.py
   ```

2. **Test your changes locally**
   ```bash
   # For small changes, test with reduced dataset
   python run_curriculum_training.py --stage caption
   ```

3. **Check for memory leaks**
   - Monitor GPU memory usage
   - Ensure proper cleanup after training

### Writing Tests
- Add tests for new functions
- Test edge cases and error conditions
- Include integration tests for training pipeline

## Pull Request Process

### Before Creating a PR
1. **Sync with upstream**
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-branch
   git rebase main
   ```

2. **Run pre-commit checks**
   - Code formatting
   - Lint checks
   - Test suite

### PR Description Template
```markdown
## Summary
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Ran test_setup.py successfully
- [ ] Tested training pipeline
- [ ] Verified memory usage

## Checklist
- [ ] Code follows project style guidelines
- [ ] Added/updated documentation
- [ ] Added tests for new functionality
- [ ] All tests pass locally
```

## Specific Areas for Contribution

### High Priority
1. **Memory Optimization**
   - Reduce GPU memory usage
   - Implement gradient accumulation improvements
   - Add support for smaller GPUs

2. **Evaluation Metrics**
   - Add ROUGE scores
   - Implement medical-specific metrics
   - Create evaluation visualizations

3. **Data Processing**
   - Support for additional image formats
   - Enhanced data augmentation
   - Better error handling for corrupted images

### Medium Priority
1. **Model Support**
   - Support for other vision-language models
   - Multi-GPU training
   - Mixed precision improvements

2. **Configuration**
   - YAML configuration files
   - Hyperparameter tuning utilities
   - Training resumption improvements

### Documentation
1. **Tutorials**
   - Step-by-step training guide
   - Troubleshooting guide
   - Performance optimization tips

2. **Code Documentation**
   - Function docstrings
   - Module documentation
   - Architecture explanations

## Issue Reporting

### Bug Reports
Use the bug report template and include:
- Complete error messages
- System information
- Steps to reproduce
- Expected vs actual behavior

### Feature Requests
Use the feature request template and include:
- Clear problem description
- Proposed solution
- Alternative approaches considered
- Implementation ideas

## Code Review Process

### What We Look For
1. **Correctness**: Does the code work as intended?
2. **Performance**: Are there any performance regressions?
3. **Maintainability**: Is the code easy to understand and modify?
4. **Testing**: Are there adequate tests?
5. **Documentation**: Is the code properly documented?

### Review Timeline
- Initial review: Within 3-5 days
- Follow-up reviews: Within 1-2 days
- Merge timeline: Depends on complexity and review feedback

## Community Guidelines

### Be Respectful
- Use inclusive language
- Be constructive in feedback
- Help newcomers get started

### Communication
- Use GitHub issues for bugs and features
- Join discussions in a constructive manner
- Ask questions if you're unsure

## Getting Help

### Resources
- **README.md**: Project overview and setup
- **Issues**: Search existing issues first
- **Discussions**: For general questions and ideas

### Contact
- Create an issue for bugs or features
- Tag maintainers for urgent issues
- Use appropriate labels for categorization

## Recognition

Contributors will be:
- Listed in the contributors section
- Mentioned in release notes for significant contributions
- Invited to be maintainers for consistent, high-quality contributions

Thank you for contributing to the advancement of medical AI research!