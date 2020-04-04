package main

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
	"strings"
)

func prepareFishing(r io.Reader) (Lines, error) {
	var lines Lines
	var lineNo int
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		lineNo++
		line := make([]string, 6)
		splits := strings.Split(scanner.Text(), ",")
		if len(splits) != 5 {
			return nil, fmt.Errorf("incorrect number of items in line %d", lineNo)
		}
		switch splits[0] {
		case "Strong":
			normInput := normalize(1, 2, 1)
			line[0] = strconv.FormatFloat(normInput, 'f', -1, 32)
		case "Weak":
			normInput := normalize(1, 2, 2)
			line[0] = strconv.FormatFloat(normInput, 'f', -1, 32)
		default:
			return nil, fmt.Errorf("invalid value for Wind on line %d", lineNo)
		}
		switch splits[1] {
		case "Warm":
			normInput := normalize(1, 3, 1)
			line[1] = strconv.FormatFloat(normInput, 'f', -1, 32)
		case "Moderate":
			normInput := normalize(1, 3, 2)
			line[1] = strconv.FormatFloat(normInput, 'f', -1, 32)
		case "Cold":
			normInput := normalize(1, 3, 3)
			line[1] = strconv.FormatFloat(normInput, 'f', -1, 32)
		default:
			return nil, fmt.Errorf("invalid value for Water on line %d", lineNo)
		}
		switch splits[2] {
		case "Warm":
			normInput := normalize(1, 2, 1)
			line[2] = strconv.FormatFloat(normInput, 'f', -1, 32)
		case "Cool":
			normInput := normalize(1, 2, 2)
			line[2] = strconv.FormatFloat(normInput, 'f', -1, 32)
		default:
			return nil, fmt.Errorf("invalid value for Air on line %d", lineNo)
		}
		switch splits[3] {
		case "Sunny":
			normInput := normalize(1, 3, 1)
			line[3] = strconv.FormatFloat(normInput, 'f', -1, 32)
		case "Cloudy":
			normInput := normalize(1, 3, 2)
			line[3] = strconv.FormatFloat(normInput, 'f', -1, 32)
		case "Rainy":
			normInput := normalize(1, 3, 3)
			line[3] = strconv.FormatFloat(normInput, 'f', -1, 32)
		default:
			return nil, fmt.Errorf("invalid value for Forecast on line %d", lineNo)
		}
		switch splits[4] {
		case "Yes":
			line[4] = "1"
			line[5] = "0"
		case "No":
			line[4] = "0"
			line[5] = "1"
		default:
			return nil, fmt.Errorf("invalid value for Target on line %d", lineNo)
		}

		lines = append(lines, strings.Join(line, " "))
	}

	return lines, nil
}
