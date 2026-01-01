import sys
import unittest

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


def main() -> int:
    """Run unit tests with a colored summary.

    Returns:
        Process exit code.
    """
    suite = unittest.defaultTestLoader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print(f"{GREEN}RESULT: PASS{RESET}")
        return 0
    print(f"{RED}RESULT: FAIL{RESET}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
