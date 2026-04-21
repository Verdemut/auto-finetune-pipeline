from setuptools import setup, find_packages
from pathlib import Path

readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

setup(
    name="auto-finetune-pipeline",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Professional Neural Network Training Interface for Stable Diffusion Fine-tuning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/auto-finetune-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "diffusers>=0.25.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "gradio>=4.0.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "psutil>=5.9.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "cuda": ["torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118"],
        "xformers": ["xformers>=0.0.22"],
        "dev": ["pytest>=7.0", "black>=23.0", "flake8>=6.0"],
    },
    entry_points={
        "console_scripts": [
            "auto-finetune=main:main",
            "auto-finetune-webui=webui_platform:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)