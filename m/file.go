package m

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
	"strings"
)

type Line struct {
	Inputs  []float64
	Targets []float64
}
type Lines []Line

func GetLines(reader io.Reader, inputNum, outputNum int) (Lines, error) {
	scanner := bufio.NewScanner(reader)
	var lines Lines
	var lineNum int
	for scanner.Scan() {
		lineNum++
		splits := strings.Split(scanner.Text(), " ")
		if len(splits) != inputNum+outputNum {
			return lines, errInvalidLine{
				lineNum:  lineNum,
				splits:   len(splits),
				expected: inputNum + outputNum,
			}
		}
		inputs := make([]float64, inputNum)
		targets := make([]float64, outputNum)
		for i, split := range splits {
			if i < inputNum {
				num, err := strconv.ParseFloat(split, 64)
				if err != nil {
					return lines, fmt.Errorf("parsing input: %w", err)
				}
				inputs[i] = num
			} else {
				num, err := strconv.ParseFloat(split, 64)
				if err != nil {
					return lines, fmt.Errorf("parsing target: %w", err)
				}
				targets[i-inputNum] = num
			}
		}
		line := Line{
			Inputs:  inputs,
			Targets: targets,
		}
		lines = append(lines, line)
	}
	return lines, nil
}

type errInvalidLine struct {
	lineNum  int
	splits   int
	expected int
}

func (e errInvalidLine) Error() string {
	return fmt.Sprintf("at line %d, expected %d values, got %d",
		e.lineNum, e.expected, e.splits)
}
