# Fixture for tools/lint/run_vulture.sh -- `dead_function` must fire (unused);
# `used_function` must not (it is called below).
def dead_function():
    return 1


def used_function():
    return 2


used_function()
