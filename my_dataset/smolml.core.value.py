import math

"""
/////////////
/// VALUE ///
/////////////

Neural network library for automatic differentiation and gradient computation.
Based on amazing work by karpathy (https://github.com/karpathy/micrograd).
"""
class Value:
    """
    Stores a single scalar value and its gradient for automatic differentiation.
    Implements backpropagation through a computational graph of Value objects.
    """
    def __init__(self, data, _children=(), _op=""):
        """
        Initializes a Value with data and optional gradient computation information.
        Stores children nodes and operation for building computational graph.
        """
        self.data = data
        self.grad = 0  # Derivative of final output with respect to this value
        self._backward = lambda: None  # Function to compute gradients
        self._prev = set(_children)  # Child nodes in computational graph  
        self._op = _op  # Operation that created this value

    def __add__(self, other):
        """
        Adds two Values and sets up gradient computation for backpropagation.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        """
        Multiplies two Values and sets up gradient computation for backpropagation.
        Uses product rule for derivatives.
        """ 
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad  
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        """
        Raises Value to a power and sets up gradient computation.
        Currently only supports int/float powers.
        """
        other = other.data if isinstance(other, Value) else other
        assert isinstance(other, (int, float)), "only supporting int/float/value powers for now"
        out = Value(self.data**other, (self,), f"**{other}")
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __abs__(self):
        """
        Computes absolute value and sets up gradient computation.
        Derivative is 1 for positive values, -1 for negative values.
        """
        out = Value(abs(self.data), (self,), "abs")
        def _backward():
            self.grad += (1 if self.data >= 0 else -1) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        """
        Implements division with special handling for division by zero.
        Returns 0 when dividing by 0 or very small numbers.
        """
        other = other if isinstance(other, Value) else Value(other)
        if abs(other.data) < 1e-10:  # Division by zero or very small number
            out = Value(0.0, (self, other), "/")
            def _backward():
                self.grad += 0.0
                other.grad += 0.0
            out._backward = _backward
            return out
        return self * other**-1

    def __rtruediv__(self, other):
        """
        Implements reverse division (other / self) with zero handling.
        """
        if abs(self.data) < 1e-10:  # Division by zero or very small number
            out = Value(0.0, (self,), "r/")
            def _backward():
                self.grad += 0.0
            out._backward = _backward
            return out
        return other * self**-1

    def exp(self):
        """
        Computes exponential (e^x) and sets up gradient computation.
        Derivative of e^x is e^x.
        """
        x = self.data
        out = Value(math.exp(x), (self,), "exp")
        def _backward():
            self.grad += math.exp(x) * out.grad
        out._backward = _backward
        return out

    def log(self):
        """
        Computes natural logarithm (ln) and sets up gradient computation.
        Derivative of ln(x) is 1/x.
        """
        assert self.data > 0, "log is only defined for positive numbers"
        x = self.data
        out = Value(math.log(x), (self,), "log")
        def _backward():
            self.grad += (1.0 / x) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        """
        Applies ReLU activation function and sets up gradient computation.
        Derivative is 1 for positive values, 0 for negative values.
        """
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        """
        Applies tanh activation function and sets up gradient computation.
        Derivative is 1 - tanh²(x).
        """
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        """
        Performs backpropagation to compute gradients in the computational graph.
        Uses topological sort to process nodes in correct order.
        """
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    # Operation overloads for convenience
    def __neg__(self): return self * -1  # -self
    def __radd__(self, other): return self + other  # other + self
    def __sub__(self, other): return self + (-other)  # self - other
    def __rsub__(self, other): return other + (-self)  # other - self
    def __rmul__(self, other): return self * other  # other * self

    # Comparison and representation methods
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    def __eq__(self, other):
        if isinstance(other, Value): return self.data == other.data
        return self.data == other
    def __ne__(self, other): return not self.__eq__(other)
    def __lt__(self, other):
        if isinstance(other, Value): return self.data < other.data
        return self.data < other
    def __le__(self, other):
        if isinstance(other, Value): return self.data <= other.data
        return self.data <= other
    def __gt__(self, other):
        if isinstance(other, Value): return self.data > other.data
        return self.data > other
    def __ge__(self, other):
        if isinstance(other, Value): return self.data >= other.data
        return self.data >= other
    def __hash__(self): return id(self)