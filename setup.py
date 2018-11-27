from distutils.core import setup

if __name__ == '__main__':
    setup(
        name='regli',
        version='0.0.2',
        author='Bo Zhang',
        author_email='bzhang@mpia.de',
        # py_modules=['bopy','spec','core'],
        description='REgular Grid Linear Interpolator.',  # short description
        license='MIT',
        # install_requires=['numpy>=1.7','scipy','matplotlib','nose'],
        url='http://github.com/hypergravity/regli',
        classifiers=[
            "Development Status :: 6 - Mature",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.6",
            "Topic :: Scientific/Engineering :: Astronomy",
            "Topic :: Scientific/Engineering :: Physics"],
        package_dir={'regli': 'regli'},
        packages=['regli', ],
        #package_data={"ruby": ["data/*", "script/*"]},
        include_package_data=True,
        requires=['numpy', 'scipy', 'matplotlib']
    )
