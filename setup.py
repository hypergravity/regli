import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='regli',
    version='0.0.2',
    author='Bo Zhang',
    author_email='bozhang@nao.cas.cn',
    description='REgular Grid Linear Interpolator.',  # short description
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://github.com/hypergravity/regli',
    packages=setuptools.find_packages(),
    license='MIT',
    classifiers=[
        "Development Status :: 6 - Mature",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics"],
    package_dir={'regli': 'regli'},
    include_package_data=True,
    requires=['numpy', 'scipy']
)
