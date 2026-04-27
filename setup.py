from setuptools import setup, find_packages

setup(
    name="gpt-finetune",
    version="1.0.0",
    description="Fine-tune GPT / LLaMA models on custom text data",
    author="Your Name",
    author_email="you@example.com",
    url="https://github.com/yourusername/gpt-finetune",
    license="MIT",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.15.0",
        "accelerate>=0.25.0",
        "peft>=0.6.0",
        "evaluate>=0.4.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
        "quantization": ["bitsandbytes>=0.41.0"],
        "metrics": ["rouge-score", "bert-score", "sacrebleu"],
        "notebooks": ["jupyter", "matplotlib", "seaborn", "plotly"],
    },
    entry_points={
        "console_scripts": [
            "gpt-train=train:main",
            "gpt-generate=generate:main",
            "gpt-evaluate=evaluate:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
