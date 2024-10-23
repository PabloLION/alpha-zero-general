import glob
import subprocess


def run_autoflake():
    print(
        "Running autoflake to clean up unused imports, variables, and duplicate keys..."
    )
    # Expand the **/*.py pattern using glob
    python_files = glob.glob("**/*.py", recursive=True)

    # Run autoflake only if there are Python files
    if python_files:
        subprocess.run(
            [
                "autoflake",
                "--in-place",
                "--remove-unused-variables",
                "--remove-all-unused-imports",
                "--remove-duplicate-keys",
                "--expand-star-imports",
                "--ignore-pass-after-docstring",
            ]
            + python_files,
            check=True,
        )
    else:
        print("No Python files found for autoflake.")


def run_black():
    print("Running black...")
    subprocess.run(["black", "."], check=True)


def run_isort():
    print("Running isort...")
    subprocess.run(["isort", "."], check=True)


def run_flake8():
    print("Running flake8...")
    subprocess.run(["flake8", "."], check=True)


def run_pylint():
    print("Running pylint...")
    subprocess.run(["pylint", "."], check=True)


def run_lint():
    run_autoflake()
    run_black()
    run_isort()
    # disable flake8 and pylint for now
    # run_flake8()
    # run_pylint()


if __name__ == "__main__":
    run_lint()
