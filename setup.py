from setuptools import setup, find_packages

setup(
    name="tf_perturb",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "tqdm",
        "transformer_lens",
        "datasets",
    ],
    author="",
    author_email="",
    description="A package for mechanistic interpretability of adversarially perturbed transformers",
    long_description=open("README.md", "r").read() if open("README.md", "r") else "",
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
