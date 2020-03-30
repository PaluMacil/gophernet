package main

import (
	"flag"
	"fmt"
	"gophernet/m"
	"os"
)

func main() {
	flagNumInputs := flag.Int("input", 64, "input controls the number of input nodes")
	flagNumHidden := flag.Int("hidden", 30, "output controls the number of hidden nodes")
	flagNumOutput := flag.Int("output", 10, "output controls the number of output nodes")
	flagNumLayers := flag.Int("layers", 3, "layers controls the total number of layers to use (3 means one hidden)")
	flagNumEpochs := flag.Int("epochs", 6, "number of epochs")
	flagActivator := flag.String("activator", "sigmoid", "activator is the activation function to use (default is sigmoid)")
	flagNormalize := flag.Bool("normalize", false, "normalize controls whether each input will be normalized across the training data")
	flagLearningRate := flag.Float64("rate", .1, "rate is the learning rate")
	//TODO: see below
	//targetLabels := flag.String("labels", "0,1,2,3,4,5,6,7,8,9", "labels are name to call each output")
	flag.Parse()

	if len(flag.Args()) < 2 {
		fmt.Println("a command and dataset must be specified")
		os.Exit(1)
	}

	if *flagNumLayers < 3 {
		fmt.Println("cannot have fewer than three layers")
		os.Exit(1)
	}
	activator, ok := m.ActivatorLookup[*flagActivator]
	if !ok {
		fmt.Println("invalid activator")
		os.Exit(1)
	}
	config := m.Config{
		Name:         flag.Args()[1],
		InputNum:     *flagNumInputs,
		HiddenNum:    *flagNumHidden,
		OutputNum:    *flagNumOutput,
		LayerNum:     *flagNumLayers,
		Epochs:       *flagNumEpochs,
		Activator:    activator,
		Normalize:    *flagNormalize,
		LearningRate: *flagLearningRate,
	}

	switch flag.Args()[0] {
	case "train":
		train(config)
	case "predict":

	}
	train(config)
}

func train(config m.Config) {
	filename := config.Name + ".data"
	file, err := os.Open(filename)
	if err != nil {
		fmt.Printf("opening input file: %s", err.Error())
		os.Exit(1)
	}
	defer file.Close()

	network := m.NewNetwork(config)
	lines, err := m.GetLines(file, config.InputNum, config.OutputNum)
	if err != nil {
		fmt.Printf("couldn't get lines from file: %s\n", err.Error())
		os.Exit(1)
	}
	err = network.Train(lines)
	if err != nil {
		fmt.Printf("training network: %s\n", err.Error())
		os.Exit(1)
	}

	fmt.Println("Training complete")
}
