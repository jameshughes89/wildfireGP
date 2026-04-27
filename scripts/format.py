import subprocess


def main():
    subprocess.run(["mdformat", "."], check=True)
    subprocess.run(["isort", "."], check=True)
    subprocess.run(["black", "."], check=True)
1