# `name` is the name of the package as used for `pip install package`
name = "package_name"
# `path` is the name of the package for `import package`
path = name.lower().replace("-", "_").replace(" ", "_")
# Your version number should follow https://python.org/dev/peps/pep-0440 and
# https://semver.org
version = "0.0.0"
author = "Brian Pugh"
author_email = "bnp117@gmail.com"
description = ""  # One-liner
url = "https://github.com/BrianPugh/package_name"
license = "apache-2.0"
