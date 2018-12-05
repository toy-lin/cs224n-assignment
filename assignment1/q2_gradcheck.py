#!/usr/bin/env python

import numpy as np
import random
from q1_softmax import softmax
from q2_sigmoid import sigmoid,sigmoid_grad

# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """ Gradient check for a function f.

    Arguments:
    f -- a function that takes a single argument and outputs the
         cost and its gradients
    x -- the point (numpy array) to check the gradient at
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4        # Do not change this!

    # Iterate over all indexes ix in x to check the gradient.
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        # Try modifying x[ix] with h defined above to compute numerical
        # gradients (numgrad).

        # Use the centered difference of the gradient.
        # It has smaller asymptotic error than forward / backward difference
        # methods. If you are curious, check out here:
        # https://math.stackexchange.com/questions/2326181/when-to-use-forward-or-central-difference-approximations

        # Make sure you call random.setstate(rndstate)
        # before calling f(x) each time. This will make it possible
        # to test cost functions with built in randomness later.

        ### YOUR CODE HERE:
        random.setstate(rndstate)
        x[ix] += h
        fx_a,_ = f(x)

        random.setstate(rndstate)
        x[ix] -= 2*h
        fx_s,_ = f(x)

        x[ix] += h
        numgrad = (fx_a - fx_s) / (h*2)

        ### END YOUR CODE

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (
                grad[ix], numgrad)
            return

        it.iternext() # Step to next dimension

    print "Gradient check passed!"


def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print "Running sanity checks..."
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test
    print ""


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    a = np.array([1.0])
    b = np.array([2.0])
    params = np.concatenate((a.flatten(),b.flatten()))
    quad = lambda x : (x[0]**2+x[1]**3,np.array((2*x[0],3*(x[1]**2))))
    gradcheck_naive(quad,params)

    labels = np.zeros([5,5])
    for i in range(labels.shape[0]):
        labels[i,i] = 1
    
    print "check sigmoid ..."
    check_sigmoid = lambda x: (np.sum(sigmoid(x)),sigmoid_grad(sigmoid(x)))
    gradcheck_naive(check_sigmoid,np.random.rand(5,5))

    print "check softmax cross entropy"
    check_softmax = lambda x: (np.sum(-np.log(softmax(x)[labels==1]))/labels.shape[0],softmax_ce_grad(softmax(x),labels))
    gradcheck_naive(check_softmax,np.random.rand(5,5))
    ### END YOUR CODE

def softmax_ce_grad(outputs,labels):
    grad_ce = np.sum(-labels/outputs/labels.shape[0],1,keepdims=True)

    grad_softmax = -np.expand_dims(outputs[labels==1],1) * outputs
    grad_softmax[labels==1] = outputs[labels==1]*(1-outputs[labels==1])
    return grad_ce * grad_softmax

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
