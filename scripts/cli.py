import subprocess


def run_formatters():
    for tool in ["isort .", "black .", "mdformat ."]:
        print(f"running `{tool}`")
        subprocess.run(tool, shell=True)


def run_verification():
    for tool in ["flake8 wildfireGP/ tests/", "codespell wildfireGP/ tests/"]:
        print(f"running `{tool}`")
        subprocess.run(tool, shell=True)
