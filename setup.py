from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    # Basic package information
    name="diffusion_nl",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),

    # Packaging options
    zip_safe=False,
    include_package_data=True,

    # Package dependencies
    install_requires=[
        # List your dependencies here
        # e.g.
        # 'numpy>=1.18.0',
        # 'pandas>=1.0.0',
    ],

    # Metadata for PyPI
    author="Niklas Hoepner",
    author_email="nhopner@gmail.com",
    description="Text2Video models via Diffusion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    keywords="text2video",

    # Other configurations
    python_requires=">=3.10",
    setup_requires=["wheel"],
)