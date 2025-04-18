from setuptools import setup, find_packages

_deps = [
    "torch",
    "matplotlib"
]
_dev_deps = [
    "pytest",
    "black~=24.4.2",
    "isort~=5.13.2",
    "mypy~=1.10.0",
    "ruff~=0.4.8",
    "flake8~=7.0.0"
]

setup(
    name="PTTP",
    version="1.0",
    author="Kyle Sayers",
    description="",
    install_requires=_deps,
    extras_require={"dev": _dev_deps},
    package_dir={"": "src"},
    packages=find_packages("src", include=["pttp"], exclude=["*.__pycache__.*"]),
)
