"""
TurboVec Enhanced — Setup Configuration
"""

from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).parent

# Read README for long description
README = ROOT / "README.md"
LONG_DESCRIPTION = README.read_text(encoding="utf-8") if README.exists() else ""

# Read requirements
REQS = (ROOT / "requirements.txt").read_text(encoding="utf-8").strip().splitlines()

setup(
    name="turbovec-enhanced",
    version="1.0.0",
    author="TurboVec Enhanced Contributors",
    description="GPU-accelerated vector search, hybrid retrieval, and RAG pipeline",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/RyanCodrai/turbovec",
    packages=find_packages(exclude=["tests*", "docs*"]),
    python_requires=">=3.12",
    install_requires=REQS,
    extras_require={
        "gpu": ["faiss-gpu>=1.7.4"],
        "dev": ["pytest>=8.2.0", "pytest-asyncio>=0.23.0", "pytest-mock>=3.14.0", "black>=24.0.0", "ruff>=0.4.0"],
    },
    entry_points={
        "console_scripts": [
            "turbovec-enhanced=agent.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Indexing",
    ],
    keywords="vector search hnsw hybrid retrieval rag embedding approximate-nearest-neighbor",
    project_urls={
        "Documentation": "https://github.com/RyanCodrai/turbovec#readme",
        "Source": "https://github.com/RyanCodrai/turbovec",
        "Tracker": "https://github.com/RyanCodrai/turbovec/issues",
    },
)
