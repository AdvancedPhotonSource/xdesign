Before making a pull request, please follow these steps:

1. If you are adding a new feature, [open an issue](https://github.com/tomography/xdesign/issues) explaining what you want to add.
2. Use [pycodestyle](https://pypi.python.org/pypi/pycodestyle) or something similar to check for PEP8 compliance.
3. Create new tests for your new code and pass the existing tests by calling [nosetests](http://nose.readthedocs.io/en/latest/index.html) on the tests directory. Remember to target both python 2.7 and 3.x.
4. Document your code with [reST](http://www.sphinx-doc.org/en/1.5.1/rest.html). We use `Sphinx` to generate our documentation.
