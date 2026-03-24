"""Setup configuration for opencontext Python SDK."""

from setuptools import find_packages, setup

setup(
    name="opencontext",
    version="0.1.0",
    description="AI context management database and SDK",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="openContext Contributors",
    url="https://github.com/jjh886/openContext",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "server": ["fastapi>=0.100.0", "uvicorn[standard]>=0.23.0"],
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.20.0"],
        "all": [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.23.0",
            "openai>=1.0.0",
            "anthropic>=0.20.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
