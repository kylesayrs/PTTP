from setuptools import setup, find_packages

_deps = ["torch", "matplotlib"]


setup(
    name="PTTP",
    version="1.0",
    author="Kyle Sayers",
    description="",
    install_requires=_deps,
    extras_require={"dev": ["pytest"]},
    package_dir={"": "src"},
    packages=find_packages("src", include=["pttp"], exclude=["*.__pycache__.*"]),
)
