package main

import (
	"flag"
	"fmt"
	"github.com/PaluMacil/gophernet/m"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("a command and dataset must be specified")
		os.Exit(1)
	}
	subCommand := os.Args[1]
	networkName := os.Args[2]

	switch subCommand {
	case "train":
		// seed rand with pseudo random values
		rand.Seed(time.Now().UTC().UnixNano())
		// parse training flags
		trainFlags := flag.NewFlagSet("train", flag.ContinueOnError)
		flagNumInputs := trainFlags.Int("input", 64, "input controls the number of input nodes")
		flagNumHidden := trainFlags.Int("hidden", 30, "output controls the number of hidden nodes")
		flagNumOutput := trainFlags.Int("output", 10, "output controls the number of output nodes")
		flagNumLayers := trainFlags.Int("layers", 3, "layers controls the total number of layers to use (3 means one hidden)")
		flagNumEpochs := trainFlags.Int("epochs", 6, "number of epochs")
		flagActivator := trainFlags.String("activator", "sigmoid", "activator is the activation function to use (default is sigmoid)")
		flagLearningRate := trainFlags.Float64("rate", .05, "rate is the learning rate")
		flagTargetLabels := trainFlags.String("labels", "0,1,2,3,4,5,6,7,8,9", "labels are name to call each output")
		err := trainFlags.Parse(os.Args[3:])
		if err != nil {
			fmt.Printf("parsing train flags: %s\n", err.Error())
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

		labelSplits := strings.Split(*flagTargetLabels, ",")
		if len(labelSplits) != *flagNumOutput {
			fmt.Printf("expected %d target labels, got %d\n", *flagNumOutput, len(labelSplits))
			os.Exit(1)
		}

		config := m.Config{
			Name:         networkName,
			InputNum:     *flagNumInputs,
			HiddenNum:    *flagNumHidden,
			OutputNum:    *flagNumOutput,
			LayerNum:     *flagNumLayers,
			Epochs:       *flagNumEpochs,
			TargetLabels: labelSplits,
			Activator:    activator,
			LearningRate: *flagLearningRate,
		}

		train(config)
	case "predict":
		predictFlags := flag.NewFlagSet("predict", flag.ContinueOnError)
		flagQuery := predictFlags.String("query", "0,1,0,0", "labels are name to call each output")
		err := predictFlags.Parse(os.Args[3:])
		if err != nil {
			fmt.Printf("parsing train flags: %s\n", err.Error())
			os.Exit(1)
		}
		queryStrings := strings.Split(*flagQuery, ",")
		query := make([]float64, len(queryStrings))
		for i, s := range queryStrings {
			num, err := strconv.ParseFloat(s, 64)
			if err != nil {
				fmt.Printf("parsing input: %s\n", err.Error())
			}
			query[i] = num
		}

		network, err := m.BestNetworkFor(networkName)
		if err != nil {
			fmt.Printf("predicting %s: %s\n", queryStrings, err.Error())
			os.Exit(1)
		}

		prediction := network.Predict(query)
		fmt.Println("Prediction:", prediction)
	}
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
	err = network.Analyze()
	if err != nil {
		fmt.Printf("doing analysis of network: %s\n", err.Error())
		os.Exit(1)
	}
	fmt.Println("Training complete")
}
