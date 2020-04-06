package m

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
)

type Activator interface {
	Activate(i, j int, sum float64) float64
	Deactivate(m mat.Matrix) mat.Matrix
	fmt.Stringer
}

var ActivatorLookup = map[string]Activator{
	"sigmoid": Sigmoid{},
}

type Sigmoid struct{}

func (s Sigmoid) Activate(i, j int, sum float64) float64 {
	return 1.0 / (1.0 + math.Exp(-sum))
}

func (s Sigmoid) Deactivate(matrix mat.Matrix) mat.Matrix {
	rows, _ := matrix.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return multiply(matrix, subtract(ones, matrix))
}

func (s Sigmoid) String() string {
	return "sigmoid"
}
