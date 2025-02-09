import os
import re
from setuptools import setup, find_packages


def get_version():
    version_file = os.path.join(
        os.path.dirname(__file__), "src", "bayescurvefit", "__init__.py"
    )
    with open(version_file, "r") as f:
        for line in f:
            match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', line)
            if match:
                return match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="bayescurvefit",
    version=get_version(),
    description=(
        "BayesCurveFit: Enhancing Biological Data Analysis with Robust Curve "
        "Fitting and FDR Detection"
    ),
    author="Niu Du",
    author_email="ndu0328@gmail.com",
    license="Apache-2.0",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.26,<2.0.0",
        "pandas>=2.0",
        "matplotlib",
        "seaborn",
        "scipy>=1.7.1,<1.15.0",
        "tqdm",
        "scikit-learn>=1.5.0",
        "emcee>=3.1.6",
    ],
    extras_require={
        "dev": [
            "pytest",
            "flake8>=3.9.2",
            "black>=22.3.0",
            "isort>=5.10.1",
            "mypy",
            "pre-commit",
            "autoflake",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
