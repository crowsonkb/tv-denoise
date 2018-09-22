from setuptools import setup

setup(
    name='tv-denoise',
    version='0.1',
    description='Total variation denoising for images.',
    long_description=open('README.rst').read(),
    url='https://www.example.com/',
    author='Katherine Crowson',
    author_email='crowsonkb@gmail.com',
    # license='MIT',
    packages=['tv_denoise'],
    install_requires=['dataclasses>=0.6;python_version<"3.7"',
                      'numpy>=1.14.3',
                      'pillow>=5.1.0'],
    entry_points={
        'console_scripts': ['tv_denoise=tv_denoise.cli:main'],
    },
)
