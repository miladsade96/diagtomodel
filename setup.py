import setuptools


with open("README.md", "r", encoding="uft-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="diag2model",
    version="0.3.1",
    author="Milad Sadeghi DM",
    author_email="EverLookNeverSee@ProtonMail.ch",
    description="Implementations of Artificial Neural Networks Based on their Diagrams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EverLookNeverSee/diag2model",
    project_urls={
        "Bug Tracker": "https://github.com/EverLookNeverSee/diag2model/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={""},
    packages=setuptools.find_packages(where=""),
    python_requires=">=3.9"
)
