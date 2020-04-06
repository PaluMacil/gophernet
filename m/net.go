package m

import (
	"encoding/csv"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"io"
	"math"
	"os"
	"path"
	"path/filepath"
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

func newPredictionNetwork(weights []mat.Matrix, activator Activator) Network {
	return Network{
		config:       Config{Activator: activator},
		weights:      weights,
		layers:       make([]mat.Matrix, len(weights)+1),
		weightedSums: make([]mat.Matrix, len(weights)),
	}
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
	config        Config
}

func (net Network) lastIndex() int {
	return len(net.layers) - 1
}

func (net Network) testFilepath() string {
	return path.Join("data", "test", net.config.Name+".data")
}

func (net Network) testExists() bool {
	info, err := os.Stat(net.testFilepath())
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
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
			net.errors[i] = dot(net.weights[i].T(), net.errors[i+1])
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
	for i := range net.layers {
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
	net.feedForward(inputData)

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

var outPath = path.Join("data", "out")
var analysisFilepath = path.Join(outPath, "analysis.csv")

// Analyze tests the network against the test set and outputs the accuracy as well as writing to a log
func (net Network) Analyze() error {
	var needsHeaders bool
	if _, err := os.Stat(analysisFilepath); os.IsNotExist(err) {
		needsHeaders = true
	}
	file, err := os.OpenFile(analysisFilepath,
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	w := csv.NewWriter(file)
	if needsHeaders {
		err = w.Write([]string{
			"Name", "Activator", "Inputs", "Hiddens", "Outputs", "Layers", "Epochs", "Target Labels", "LR", "End Time", "SecondsToTrain", "Accuracy",
		})
		if err != nil {
			return fmt.Errorf("writing csv headers: %w", err)
		}
		w.Flush()
	}
	record := make([]string, 12)
	record[0] = net.config.Name
	record[1] = net.config.Activator.String()
	record[2] = strconv.Itoa(net.config.InputNum)
	record[3] = strconv.Itoa(net.config.HiddenNum)
	record[4] = strconv.Itoa(net.config.OutputNum)
	record[5] = strconv.Itoa(net.config.LayerNum)
	record[6] = strconv.Itoa(net.config.Epochs)
	record[7] = strings.Join(net.config.TargetLabels, ", ")
	record[8] = strconv.FormatFloat(net.config.LearningRate, 'f', 4, 32)
	record[9] = strconv.Itoa(int(net.trainingEnd))
	record[10] = strconv.Itoa(int(net.trainingEnd - net.trainingStart))
	if net.testExists() {
		accuracy, err := net.test()
		if err != nil {
			return fmt.Errorf("testing network: %w", err)
		}
		record[11] = strconv.FormatFloat(accuracy, 'f', 5, 32)
		fmt.Printf("Accuracy %.2f%%\n", accuracy)
	} else {
		record[11] = "?"
		fmt.Printf("Accuracy: (no test file at %s)", net.testFilepath())
	}
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
	file, err := os.Open(net.testFilepath())
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
		f, err := os.Create(path.Join("data", "out", filename))
		if err != nil {
			return err
		}
		d := net.weights[i].(*mat.Dense)
		_, err = d.MarshalBinaryTo(f)
		if err != nil {
			return fmt.Errorf("marshalling weights: %w\n", err)
		}
		err = f.Close()
		if err != nil {
			return err
		}
	}

	return nil
}

func load(name, endingTime string, activator Activator) (Network, error) {
	sep := string(os.PathSeparator)
	pattern := fmt.Sprintf(".%s%s%s%s-%s-*.wgt", sep, outPath, sep, name, endingTime)
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return Network{}, fmt.Errorf("matching pattern %s: %w", pattern, err)
	}
	weights := make([]mat.Dense, len(matches))
	for _, m := range matches {
		splits := strings.Split(m, "-")
		layerString := strings.Split(splits[2], ".")[0]
		layerIndex, err := strconv.Atoi(layerString)
		if err != nil {
			return Network{}, fmt.Errorf("converting layer portion of filename to a number: %w", err)
		}
		f, err := os.Open(m)
		if err != nil {
			return Network{}, fmt.Errorf("opening file for layer %s: %w", layerString, err)
		}
		weights[layerIndex].Reset()
		_, err = weights[layerIndex].UnmarshalBinaryFrom(f)
		if err != nil {
			return Network{}, fmt.Errorf("unmarshalling layer %s: %w", layerString, err)
		}
		err = f.Close()
		if err != nil {
			return Network{}, fmt.Errorf("closing file for layer %s: %w", layerString, err)
		}
	}
	matrices := make([]mat.Matrix, len(weights))
	for i := range weights {
		matrices[i] = &weights[i]
	}

	return newPredictionNetwork(matrices, activator), nil
}

const csvRecords = 12

// bestRun takes a dataset name and returns the best run epoch and activator
func bestRun(name string) (string, Activator, error) {
	file, err := os.Open(analysisFilepath)
	if err != nil {
		return "", nil, fmt.Errorf("opening analysis csv file: %w", err)
	}
	r := csv.NewReader(file)
	// set to negative because if all accuracies for this data set were not measured, they failed parse of accuracy will
	// parse as the zero value of a float (0), which allows us to use the untested run until we get test data
	highestAccuracy := -1.
	var bestEndingTime string
	var activator Activator
	i := 0
	// Iterate through the records
	for {
		// Read each record from csv
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", nil, fmt.Errorf("reading record: %w", err)
		}
		if len(record) != csvRecords {
			if i == 0 {
				return "", nil, fmt.Errorf("there are %d analysis csv headers, expected %d", len(record), csvRecords)
			} else {
				return "", nil, fmt.Errorf("there are %d analysis csv values in record %d, expected %d", len(record), i, csvRecords)
			}
		}
		// record[0] is name
		// record[1] is activator
		// record[9] is time ending (epoch time)
		// record[11] is Accuracy
		if record[0] != name {
			continue
		}
		accuracy, _ := strconv.ParseFloat(record[11], 64)
		if accuracy > highestAccuracy {
			highestAccuracy = accuracy
			bestEndingTime = record[9]
			var ok bool
			activator, ok = ActivatorLookup[record[1]]
			if !ok {
				return "", nil, fmt.Errorf("invalid activator: %s", record[1])
			}
		}
		i++
	}

	return bestEndingTime, activator, nil
}

func BestNetworkFor(name string) (Network, error) {
	epoch, activator, err := bestRun(name)
	if err != nil {
		return Network{}, fmt.Errorf("getting best epoch for %s: %w", name, err)
	}
	net, err := load(name, epoch, activator)
	if err != nil {
		return Network{}, fmt.Errorf("loading network")
	}

	return net, nil
}
