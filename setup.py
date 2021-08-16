import setuptools


setuptools.setup(
    name="juno-dkt", # Replace with your own username
    version="0.8.9",
    author="Juno Hwang",
    author_email="wnsdh10@snu.ac.kr",
    description="Scikit-learn style implementation of Deep Knowledge Tracing models based on pytorch.",
    url="https://github.com/juno-hwang/juno-dkt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy','pandas','scikit-learn','tqdm'],
    python_requires='>=3.6',
)