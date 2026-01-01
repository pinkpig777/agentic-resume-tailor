import subprocess
import sys


def main() -> int:
    """Run pytest with color enabled.

    Returns:
        Process exit code.
    """
    result = subprocess.run([sys.executable, "-m", "pytest", "--color=yes"], check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
