# 1. Update package version in setup.py

# 2. Build package
python setup.py sdist bdist_wheel

# 3. Upload to pip (pip install twine)
twine upload dist/*
