# see: https://packaging.python.org/tutorials/packaging-projects/
from setuptools import setup, find_packages

with open('README.md') as fp:
    long_description = fp.read()

setup(
    name = 'pykuwahara',
    version = '0.2',
    author = 'J. Melka',
    author_email="yoch.melka@gmail.com",
    description = 'Kuwahara filter in python',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url = 'https://github.com/yoch/pykuwahara',
    license = 'GPL3',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Artistic Software',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=['numpy', 'opencv-contrib-python'],
    #python_requires=">=3.6",
)
