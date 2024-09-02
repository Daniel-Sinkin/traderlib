from setuptools import find_packages, setup

# Define the package name and other metadata
setup(
    name="traderlib",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A C++ Python extension",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/traderlib",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    package_data={
        "traderlib": ["*.so"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
