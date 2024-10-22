import subprocess


def run_lint():
    subprocess.run(["black", "."], check=True)
    subprocess.run(["isort", "."], check=True)


if __name__ == "__main__":
    run_lint()
