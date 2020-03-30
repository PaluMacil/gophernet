package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
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

	originalLines, err := GetLines(file)
	newLines := make(Lines, len(originalLines))
	for i, o := range originalLines {
		targetList := make([]int, 10)
		targetList[o.target] = 1
		var targetString strings.Builder
		for i, t := range targetList {
			if i != 0 {
				targetString.WriteString(" ")
			}
			targetString.WriteString(strconv.Itoa(t))
		}
		newLines[i] = fmt.Sprintf("%s %s", originalLines[i].inputs, targetString.String())
	}
	newFilename := filename + ".2"
	err = newLines.WriteTo(newFilename)
	if err != nil {
		fmt.Printf("writing output file: %s", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %d lines to %s", len(newLines), newFilename)
}

type Line struct {
	inputs string
	target int
}

type Lines []string

func (l Lines) WriteTo(filename string) error {
	file, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("failed creating file: %w", err)
	}
	defer file.Close()

	w := bufio.NewWriter(file)

	for _, line := range l {
		_, err = w.WriteString(line + "\n")
		if err != nil {
			return fmt.Errorf("writing string: %w", err)
		}
	}

	err = w.Flush()
	if err != nil {
		return fmt.Errorf("flushing buffer: %w", err)
	}

	return nil
}

func GetLines(reader io.Reader) ([]Line, error) {
	scanner := bufio.NewScanner(reader)
	var lines []Line
	for scanner.Scan() {
		splits := strings.Split(scanner.Text(), " ")
		input := strings.Join(splits[:64], " ")
		target, err := strconv.Atoi(splits[64])
		if err != nil {
			return lines, fmt.Errorf("parsing target: %w", err)
		}
		line := Line{
			inputs: input,
			target: target,
		}
		lines = append(lines, line)
	}
	return lines, nil
}
