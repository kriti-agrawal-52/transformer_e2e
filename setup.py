# setup.py - Python Package Configuration File
# ===============================================
# This file tells Python how to install your project as a package.
# When someone runs "pip install ." in your project directory, 
# Python reads this file to understand:
# - What your package is called
# - What version it is  
# - What dependencies it needs
# - How to install it
# - What command-line tools it provides

# Import setuptools - the modern Python packaging library
from setuptools import setup, find_packages

# Read the README.md file to use as the long description on PyPI
# This way, your PyPI page will show your README content
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements.txt file and parse it into a list
# This automatically includes all your dependencies in the package
with open("requirements.txt", "r", encoding="utf-8") as fh:
    # Create a list of requirements, filtering out empty lines and comments
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# The main setup() function - this defines your package
setup(
    # Package metadata
    name="transformer-e2e",  # The name people will use to "pip install transformer-e2e"
    version="0.1.0",  # Semantic versioning: MAJOR.MINOR.PATCH
    author="Kriti Agrawal",  # Your name
    author_email="kritiagrawal5297@gmail.com",  # Your email (update this!)
    
    # Short description (appears in search results)
    description="End-to-end Transformer MLOps Pipeline for training, fine-tuning, and deployment",
    
    # Long description (appears on the package page)
    long_description=long_description,
    long_description_content_type="text/markdown",  # Tell PyPI it's markdown
    
    # Where to find your code (update this with your actual GitHub URL!)
    url="hhttps://github.com/kriti-agrawal-52/transformer_e2e",
    
    # Automatically find all Python packages in your project
    # This looks for directories with __init__.py files
    packages=find_packages(),
    
    # Classifiers help people find your package on PyPI
    # These are standardized categories that describe your project
    classifiers=[
        "Development Status :: 3 - Alpha",  # Still in early development
        "Intended Audience :: Developers",  # Who should use this
        "License :: OSI Approved :: MIT License",  # Open source license
        "Operating System :: OS Independent",  # Works on any OS
        "Programming Language :: Python :: 3",  # Written in Python 3
        "Programming Language :: Python :: 3.11",  # Specific Python version
        "Topic :: Scientific/Engineering :: Artificial Intelligence",  # AI/ML category
    ],
    
    # Minimum Python version required
    python_requires=">=3.11",
    
    # Dependencies that pip will automatically install
    # These come from your requirements.txt file
    install_requires=requirements,
    
    # Entry points create command-line tools
    # After installation, users can run "transformer-train" and "transformer-generate"
    entry_points={
        "console_scripts": [
            # "command-name=module.submodule:function"
            "transformer-train=scripts.train:main",  # Creates "transformer-train" command
            "transformer-generate=scripts.generate:main",  # Creates "transformer-generate" command
        ],
    },
) 