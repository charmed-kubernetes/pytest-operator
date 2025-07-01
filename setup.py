#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from setuptools import setup, find_packages

setup(
    author="Cory Johns",
    author_email="cory.johns@canonical.com",
    description="Fixtures for Operators",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Framework :: Pytest",
        "Programming Language :: Python",
        "Topic :: Software Development :: Testing",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    license="MIT license",
    include_package_data=True,
    keywords=["pytest", "py.test", "operators", "yaml", "asyncio"],
    name="pytest-operator",
    packages=find_packages(include=["pytest_operator"]),
    package_data={"pytest_operator": ["py.typed"]},
    url="https://github.com/charmed-kubernetes/pytest-operator",
    version="0.42.1",
    zip_safe=True,
    install_requires=[
        "ipdb",
        "pytest",
        "pytest-asyncio<0.23",
        "pyyaml",
        "juju",
        "jinja2",
    ],
    entry_points={
        "pytest11": [
            "pytest-operator = pytest_operator.plugin",
        ]
    },
)
