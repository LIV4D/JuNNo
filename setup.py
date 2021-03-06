from setuptools import setup, find_packages

setup(
    name='JuNNo',
    version='1.0',
    package_dir={'': 'lib'},
    packages=find_packages('lib'),
    url='',
    license='',
    long_description=open('README.md').read(),
    author='Gabriel Lepetit-Aimon',
    author_email='',
    include_package_data=True,
    description='',
    requires=['ipywidgets(>=7.4.2)',
              'sympy(>=1.3)',
              'tqdm(>=4.29.1)',
              'numpy(>=1.15.4)',
              'traitlets(>=4.3.2)',
              'six(>=1.12.0)',
              'scipy(>=1.1.0)',
              'pandas(>=0.23.4)',
              'psutil(>=5.4.8)',
              'ipython(>=7.2.0)',
              'scikit_learn(>=0.20.2)',
              'tables(>=3.4.4)'],
)
