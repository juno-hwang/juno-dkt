import setuptools

with open("pypiREADME.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="juno-dkt", # Replace with your own username
    version="0.8.12",
    author="Juno Hwang",
    author_email="wnsdh10@snu.ac.kr",
    description="Scikit-learn style implementation of Deep Knowledge Tracing models based on pytorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/juno-hwang/juno-dkt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy','pandas','scikit-learn','tqdm','networkx'],
    python_requires='>=3.6',
)