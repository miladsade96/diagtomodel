import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="diagtomodel",
    version="0.3.3",
    author="Milad Sadeghi DM",
    author_email="EverLookNeverSee@ProtonMail.ch",
    description="Implementations of Artificial Neural Networks Based on their Diagrams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EverLookNeverSee/diag2model",
    keywords=[
        "Python",
        "Machine Learning",
        "Computer Vision",
        "Deep Learning",
        "Neural Networks",
    ],
    install_requires=[
        "tensorflow>=2.5.0",
        "tensorflow-gpu>=2.5.0"
    ],
    project_urls={
        "GitHub Pages": "https://everlookneversee.github.io/diag2model/",
        "Bug Tracker": "https://github.com/EverLookNeverSee/diag2model/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9"
)
