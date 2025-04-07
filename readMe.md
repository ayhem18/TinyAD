# TinyAD: A Tiny Automatic Differentiation Tool

## Overview

This is a Python implementation of an Automatic Differentiation (AD) tool with potential extensions to C++. TinyAD provides a lightweight and flexible framework for computing derivatives of arbitrary functions through the computational graph approach.

## Key Features

- **Computational Graph**: Builds a dynamic computational graph that tracks operations between variables
- **Automatic Gradient Computation**: Implements both forward and backward (reverse-mode) automatic differentiation
- **Composable Operations**: Supports elementary mathematical operations that can be combined to form complex expressions
- **Extensible Architecture**: Built with a clean, object-oriented design that makes it easy to add new operations

## Implementation Details

### Core Components

- **Variable Types**:
  - `Var`: Abstract base class for all variables in the computation graph
  - `ElementaryVar`: Represents input variables with assignable values
  - `ConstantVar`: Represents fixed numerical constants

- **Operations**:
  - Binary operations (Add, Mult, Sub, Div)
  - Extensible to support unary operations and more complex functions

### Differentiation Approach

TinyAD implements reverse-mode automatic differentiation, which efficiently computes gradients by tracking the computational graph and propagating derivatives backward from the output to inputs.

## Usage Examples

***PS: Soon, I will overload the basic operators to make the usage even more intuitive***

TinyAD allows you to define variables, build expressions, compute values, and calculate gradients:

```
x = ElementaryVar("x", 2.0)

y = ElementaryVar("y", 3.0)

# Build expression: z = x * y + x
z = Add(Mult(x, y), x)

# Compute the value
result = z.compute()  # Evaluates to 8.0

# Compute gradients
z.backward()
# x.grad = 4.0 (derivative of z with respect to x)
# y.grad = 2.0 (derivative of z with respect to y)
```


## Testing

The project includes comprehensive unit tests for all operations, including:
- Basic forward computation tests
- Gradient computation tests for individual operations
- Complex expression tests combining multiple operations
- Numerical stability tests

## Development

This project is under active development. Future plans include:
- Adding more mathematical operations (exponential, logarithmic, trigonometric)
- Optimizing performance for large computational graphs
- Extending to C++ implementation for performance-critical applications
- Adding vectorized operations support

## License

This project is available under the MIT License.
