def get_gradient_at_b(x, y, m, b):
  diff = 0
  # N is the number of points
  N = len(x)
  for i in range(0, len(x)):
    y_val = y[i]
    x_val = x[i]
    diff += (y_val - ((m * x_val) + b))
  # Define b_gradient
  b_gradient = -2/N * diff
  return b_gradient
