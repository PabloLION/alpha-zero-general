#!/bin/bash

# Set dry run to true or false
DRY_RUN=false # Change to 'false' to actually rename files

# Function to convert CamelCase to snake_case
to_snake_case() {
  # Match fully uppercase acronyms
  if [[ "$1" =~ ^[A-Z]+$ ]]; then
    echo "$1" | tr '[:upper:]' '[:lower:]'
  else
    # Split on two consecutive uppercase letters
    echo "$1" | sed -r 's/([A-Z])([A-Z][a-z])/\1_\2/g' | sed -r 's/([a-z0-9])([A-Z])/\1_\2/g' | tr '[:upper:]' '[:lower:]'
  fi
}

# Function to output as a table
print_table() {
  printf "%-60s | %-60s\n" "Original File" "Proposed Rename"
  printf "%-60s | %-60s\n" "--------------------------------------------------" "--------------------------------------------------"
}

# Find all Python files (excluding pycache and non-Python files)
print_table
find ./alpha_zero_general -type f -name "*.py" ! -path "*/__pycache__/*" | while read -r file; do
  # Get the base name of the file (without path and extension)
  filename=$(basename "$file" .py)

  # Skip __init__.py files
  if [[ "$filename" == "__init__" ]]; then
    continue
  fi

  # Convert filename to snake_case
  new_filename=$(to_snake_case "$filename")

  # Get the directory of the file
  dir=$(dirname "$file")

  # Check if a rename is necessary
  if [[ "$filename" != "$new_filename" ]]; then
    if [[ "$DRY_RUN" == "true" ]]; then
      printf "%-60s | %-60s\n" "$file" "$dir/$new_filename.py"
    else
      mv "$file" "$dir/$new_filename.py"
      echo "Renamed: $file -> $dir/$new_filename.py"
    fi
  fi
done
