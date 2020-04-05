package main

import (
	"fmt"
	"os"
	"strings"
)

// This purpose of this script is to pre-process the data specific to the digits data. It splits
// the target value into 10 (0 to 9). It will write the original input filename with the number 2
// on the end. After manual examination you can replace the file and never run this again.
func main() {
	if len(os.Args) != 2 {
		fmt.Printf("command requires input filename")
		os.Exit(1)
	}
	filename := os.Args[1]
	file, err := os.Open(filename)
	if err != nil {
		fmt.Printf("opening input file: %s", err)
		os.Exit(1)
	}
	defer file.Close()

	var newLines Lines
	// preparation is hardcoded to look for this being either fishing or digits
	baseFilename := strings.Split(filename, ".")[0]
	if baseFilename == "digits" || baseFilename == "digits-test" {
		newLines, err = prepareDigits(file)
		if err != nil {
			fmt.Printf("preparing digits: %s", err)
			os.Exit(1)
		}
	} else {
		newLines, err = prepareFishing(file)
		if err != nil {
			fmt.Printf("preparing fishing: %s", err)
			os.Exit(1)
		}
	}

	newFilename := filename + ".2"
	err = newLines.WriteTo(newFilename)
	if err != nil {
		fmt.Printf("writing output file: %s", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %d lines to %s", len(newLines), newFilename)
}
