// Fixture for tools/lint/host_warnings.py -- must fire (-Wunused-parameter).
int bad(int used, int unused_param) {
  return used;
}

// Must NOT fire: every parameter is used.
int ok(int a, int b) {
  return a + b;
}
