#!/bin/bash
# Managing only top-level dependencies ensures cleaner project files and allows
# Poetry to automatically resolve and lock transitive dependencies, promoting
# reproducibility and avoiding conflicts.

# List of package names and their corresponding import names
PACKAGE_NAMES=(
  "grpcio" "tensorboard-plugin-wit" "absl-py" "astunparse" "cachetools" "certifi"
  "charset-normalizer" "coloredlogs" "flatbuffers" "gast" "google-auth"
  "google-auth-oauthlib" "google-pasta" "humanfriendly" "idna" "importlib-metadata"
  "keras" "keras-preprocessing" "libclang" "markdown" "numpy" "oauthlib" "opt-einsum"
  "protobuf" "pyasn1" "pyasn1-modules" "pyparsing" "requests" "requests-oauthlib"
  "rsa" "six" "tensorboard" "tensorboard-data-server" "tensorflow-estimator"
  "termcolor" "tqdm" "typing-extensions" "urllib3" "werkzeug" "wrapt" "zipp" "h5py"
  "tensorflow" "torch" "pytest" "pytest-mock" "packaging" "black" "isort" "flake8"
  "pylint" "autoflake" "flake8-isort" "flake8-docstrings" "flake8-bugbear"
  "flake8-comprehensions" "flake8-black"
)

IMPORT_NAMES=(
  "grpc" "tensorboard_plugin_wit" "absl" "astunparse" "cachetools" "certifi"
  "charset_normalizer" "coloredlogs" "flatbuffers" "gast" "google.auth"
  "google_auth_oauthlib" "google_pasta" "humanfriendly" "idna" "importlib_metadata"
  "keras" "keras_preprocessing" "clang" "markdown" "numpy" "oauthlib" "opt_einsum"
  "google.protobuf" "pyasn1" "pyasn1_modules" "pyparsing" "requests" "requests_oauthlib"
  "rsa" "six" "tensorboard" "tensorboard_data_server" "tensorflow_estimator"
  "termcolor" "tqdm" "typing_extensions" "urllib3" "werkzeug" "wrapt" "zipp" "h5py"
  "tensorflow" "torch" "pytest" "pytest_mock" "packaging" "black" "isort" "flake8"
  "pylint" "autoflake" "flake8_isort" "flake8_docstrings" "flake8_bugbear"
  "flake8_comprehensions" "flake8_black"
)

# List to store unused packages
unused_packages=()

# Iterate over each package-import pair
for i in "${!PACKAGE_NAMES[@]}"; do
  package_name="${PACKAGE_NAMES[$i]}"
  import_name="${IMPORT_NAMES[$i]}"

  # Search for the import name in all .py and .ipynb files
  found=$(grep -r -m 1 -E "import ${import_name}|from ${import_name}" --include=\*.py --include=\*.ipynb .)

  # If not found, add to unused_packages list
  if [[ -z "$found" ]]; then
    unused_packages+=("$package_name")
  fi
done

# Return the unused packages list
if [ ${#unused_packages[@]} -eq 0 ]; then
  echo "All packages are used in the codebase."
else
  echo "The following packages are not used and can potentially be removed:"
  for pkg in "${unused_packages[@]}"; do
    echo "$pkg"
  done
fi

# for new bash, we can use
# declare -A PACKAGE_IMPORT_MAP=(
#     ["grpcio"]="grpc"
#     ["tensorboard-plugin-wit"]="tensorboard_plugin_wit"
#     ["absl-py"]="absl"
#     ["astunparse"]="astunparse"
#     ["cachetools"]="cachetools"
#     ["certifi"]="certifi"
#     ["charset-normalizer"]="charset_normalizer"
#     ["coloredlogs"]="coloredlogs"
#     ["flatbuffers"]="flatbuffers"
#     ["gast"]="gast"
#     ["google-auth"]="google.auth"
#     ["google-auth-oauthlib"]="google_auth_oauthlib"
#     ["google-pasta"]="google_pasta"
#     ["humanfriendly"]="humanfriendly"
#     ["idna"]="idna"
#     ["importlib-metadata"]="importlib_metadata"
#     ["keras"]="keras"
#     ["keras-preprocessing"]="keras_preprocessing"
#     ["libclang"]="clang"
#     ["markdown"]="markdown"
#     ["numpy"]="numpy"
#     ["oauthlib"]="oauthlib"
#     ["opt-einsum"]="opt_einsum"
#     ["protobuf"]="google.protobuf"
#     ["pyasn1"]="pyasn1"
#     ["pyasn1-modules"]="pyasn1_modules"
#     ["pyparsing"]="pyparsing"
#     ["requests"]="requests"
#     ["requests-oauthlib"]="requests_oauthlib"
#     ["rsa"]="rsa"
#     ["six"]="six"
#     ["tensorboard"]="tensorboard"
#     ["tensorboard-data-server"]="tensorboard_data_server"
#     ["tensorflow-estimator"]="tensorflow_estimator"
#     ["termcolor"]="termcolor"
#     ["tqdm"]="tqdm"
#     ["typing-extensions"]="typing_extensions"
#     ["urllib3"]="urllib3"
#     ["werkzeug"]="werkzeug"
#     ["wrapt"]="wrapt"
#     ["zipp"]="zipp"
#     ["h5py"]="h5py"
#     ["tensorflow"]="tensorflow"
#     ["torch"]="torch"
#     ["pytest"]="pytest"
#     ["pytest-mock"]="pytest_mock"
#     ["packaging"]="packaging"
#     ["black"]="black"
#     ["isort"]="isort"
#     ["flake8"]="flake8"
#     ["pylint"]="pylint"
#     ["autoflake"]="autoflake"
#     ["flake8-isort"]="flake8_isort"
#     ["flake8-docstrings"]="flake8_docstrings"
#     ["flake8-bugbear"]="flake8_bugbear"
#     ["flake8-comprehensions"]="flake8_comprehensions"
#     ["flake8-black"]="flake8_black"
# )
