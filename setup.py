import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='textgrid_editor',
    version='1.0.0',
    author='hellolzc',
    url='https://github.com/hellolzc/textgrid_editor',
    author_email='me@hellolzc.cn',
    description="Praat TextGrid editor.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    # test_suite='tests',
)
