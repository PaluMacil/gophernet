package m

import (
	"encoding/csv"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
	"os"
	"path"
	"strconv"
	"strings"
	"time"
)

type Config struct {
	Name         string
	InputNum     int
	HiddenNum    int
	OutputNum    int
	LayerNum     int
	Epochs       int
	TargetLabels []string
	Activator    Activator
	LearningRate float64
}

type Normalizer struct {
}

func NewNetwork(c Config) Network {
	totalWeights := c.LayerNum - 1
	net := Network{
		config:       c,
		weights:      make([]mat.Matrix, totalWeights),
		layers:       make([]mat.Matrix, c.LayerNum),
		weightedSums: make([]mat.Matrix, totalWeights),
		errors:       make([]mat.Matrix, c.LayerNum),
	}
	lastWeightIndex := len(net.weights) - 1
	for i := 0; i <= lastWeightIndex; i++ {
		if i == 0 {
			net.weights[i] = mat.NewDense(
				net.config.HiddenNum,
				net.config.InputNum,
				randomArray(net.config.InputNum*net.config.HiddenNum, float64(net.config.InputNum)),
			)
		} else if i == lastWeightIndex {
			net.weights[i] = mat.NewDense(
				net.config.OutputNum,
				net.config.HiddenNum,
				randomArray(net.config.HiddenNum*net.config.OutputNum, float64(net.config.HiddenNum)),
			)
		} else {
			net.weights[i] = mat.NewDense(
				net.config.HiddenNum,
				net.config.HiddenNum,
				randomArray(net.config.HiddenNum*net.config.HiddenNum, float64(net.config.HiddenNum)),
			)
		}
	}

	return net
}

type Network struct {
	trainingStart int64
	trainingEnd   int64
	weights       []mat.Matrix
	layers        []mat.Matrix
	weightedSums  []mat.Matrix
	errors        []mat.Matrix
	normalizers   []Normalizer
	config        Config
}

func (net Network) lastIndex() int {
	return len(net.layers) - 1
}

func (net *Network) Train(lines Lines) error {
	net.trainingStart = time.Now().Unix()
	for i := 1; i <= net.config.Epochs; i++ {
		for _, line := range lines {
			net.trainOne(line.Inputs, line.Targets)
		}
		fmt.Printf("Epoch %d of %d complete\n", i, net.config.Epochs)
	}
	net.trainingEnd = time.Now().Unix()
	err := net.save()
	if err != nil {
		return fmt.Errorf("saving weights: %w", err)
	}
	fmt.Printf("Training took %d seconds\n", net.trainingEnd-net.trainingStart)

	return nil
}

func (net *Network) trainOne(inputData []float64, targetData []float64) {
	rand.Seed(time.Now().UTC().UnixNano())
	net.feedForward(inputData)
	finalOutputs := net.layers[net.lastIndex()]

	net.backpropagate(targetData, finalOutputs)
}

func (net *Network) backpropagate(targetData []float64, finalOutputs mat.Matrix) {
	for i := net.lastIndex(); i > 0; i-- {
		// find errors
		if i == net.lastIndex() {
			// network error
			targets := mat.NewDense(len(targetData), 1, targetData)
			net.errors[len(net.errors)-1] = subtract(targets, finalOutputs)
		} else {
			// calculate hidden errors
			net.errors[i] = dot(net.weights[1].T(), net.errors[2])
			//original: net.errors[i] = dot(net.weights[1].T(), net.errors[2])
			//maybe? net.errors[i] = dot(net.weights[i-1].T(), net.errors[i])
		}
		net.weights[i-1] = add(net.weights[i-1],
			scale(net.config.LearningRate,
				dot(multiply(net.errors[i], net.config.Activator.Deactivate(net.layers[i])),
					net.layers[i-1].T()))).(*mat.Dense)
	}
}

func (net *Network) feedForward(inputData []float64) {
	// first layer
	net.layers[0] = mat.NewDense(len(inputData), 1, inputData)
	net.weightedSums[0] = dot(net.weights[0], net.layers[0])
	for i, _ := range net.layers {
		if i == 0 {
			continue
		}
		net.layers[i] = apply(net.config.Activator.Activate, net.weightedSums[i-1])
		// don't get weighted sums if final output
		if i != len(net.layers)-1 {
			net.weightedSums[i] = dot(net.weights[i], net.layers[i])
		}
	}
}

func (net Network) Predict(inputData []float64) string {
	// feedforward
	net.layers[0] = mat.NewDense(len(inputData), 1, inputData)
	net.weightedSums[0] = dot(net.weights[0], net.layers[0])
	net.layers[1] = apply(net.config.Activator.Activate, net.weightedSums[0])
	net.weightedSums[1] = dot(net.weights[1], net.layers[1])
	net.layers[2] = apply(net.config.Activator.Activate, net.weightedSums[1])

	bestOutputIndex := 0
	highest := 0.0
	outputs := net.layers[net.lastIndex()]
	for i := 0; i < net.config.OutputNum; i++ {
		if outputs.At(i, 0) > highest {
			bestOutputIndex = i
			highest = outputs.At(i, 0)
		}
	}

	return net.labelFor(bestOutputIndex)
}

// Analyze tests the network against the test set and outputs the accuracy as well as writing to a log
func (net Network) Analyze() error {
	var needsHeaders bool
	logFilename := path.Join("data", "out", "analysis.csv")
	if _, err := os.Stat(logFilename); os.IsNotExist(err) {
		needsHeaders = true
	}
	file, err := os.OpenFile(logFilename,
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	w := csv.NewWriter(file)
	if needsHeaders {
		err = w.Write([]string{
			"Name", "Inputs", "Hiddens", "Outputs", "Layers", "Epochs", "Target Labels", "LR", "End Time", "SecondsToTrain", "Accuracy",
		})
		if err != nil {
			return fmt.Errorf("writing csv headers: %w", err)
		}
		w.Flush()
	}
	record := make([]string, 11)
	record[0] = net.config.Name
	record[1] = strconv.Itoa(net.config.InputNum)
	record[2] = strconv.Itoa(net.config.HiddenNum)
	record[3] = strconv.Itoa(net.config.OutputNum)
	record[4] = strconv.Itoa(net.config.LayerNum)
	record[5] = strconv.Itoa(net.config.Epochs)
	record[6] = strings.Join(net.config.TargetLabels, ", ")
	record[7] = strconv.FormatFloat(net.config.LearningRate, 'f', 4, 32)
	record[8] = strconv.Itoa(int(net.trainingEnd))
	record[9] = strconv.Itoa(int(net.trainingEnd - net.trainingStart))
	accuracy, err := net.test()
	if err != nil {
		return fmt.Errorf("testing network: %w", err)
	}
	record[10] = strconv.FormatFloat(accuracy, 'f', 5, 32)
	fmt.Printf("Accuracy %.2f%%\n", accuracy)
	err = w.Write(record)
	if err := w.Error(); err != nil {
		return fmt.Errorf("error writing csv: %s", err.Error())
	}
	w.Flush()

	return nil
}

func (net Network) test() (float64, error) {
	var correct float64
	var total float64
	filename := path.Join("data", "test", net.config.Name+".data")
	file, err := os.Open(filename)
	if err != nil {
		return 0, fmt.Errorf("opening test file: %w", err)
	}
	defer file.Close()
	lines, err := GetLines(file, net.config.InputNum, net.config.OutputNum)
	if err != nil {
		return 0, fmt.Errorf("getting lines: %w", err)
	}
	total = float64(len(lines))
	for _, line := range lines {
		prediction := net.Predict(line.Inputs)
		var actual string
		for i, t := range line.Targets {
			if int(t+math.Copysign(0.5, t)) == 1 {
				actual = net.labelFor(i)
				break
			}
		}
		if actual == prediction {
			correct++
		}
	}

	percent := 100 * (correct / total)

	return percent, nil
}

func (net Network) labelFor(index int) string {
	return net.config.TargetLabels[index]
}

func (net Network) save() error {
	fmt.Printf("saving layer weight files for %s, run #%d\n", net.config.Name, net.trainingEnd)
	for i := 0; i < len(net.weights); i++ {
		filename := fmt.Sprintf("%s-%d-%d.wgt", net.config.Name, net.trainingEnd, i)
		wf, err := os.Create(path.Join("data", "out", filename))
		if err != nil {
			return err
		}
		d := net.weights[i].(*mat.Dense)
		_, err = d.MarshalBinaryTo(wf)
		if err != nil {
			return fmt.Errorf("marshalling weights: %w\n", err)
		}
		err = wf.Close()
		if err != nil {
			return err
		}
	}

	return nil
}
