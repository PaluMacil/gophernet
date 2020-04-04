package main

import (
	"fmt"
	"github.com/PaluMacil/gophernet/m"
	"io"
	"math"
	"strconv"
	"strings"
)

func prepareDigits(r io.Reader) (Lines, error) {
	originalLines, err := m.GetLines(r, 64, 1)
	if err != nil {
		return Lines{}, fmt.Errorf("getting lines: %w", err)
	}
	newLines := make(Lines, len(originalLines))
	for i, o := range originalLines {
		var inputString strings.Builder
		for _, in := range originalLines[i].Inputs {
			normInput := normalize(0, 16, in)
			inString := strconv.FormatFloat(normInput, 'f', -1, 32)
			inputString.WriteString(inString)
			inputString.WriteString(" ")
		}

		targetList := make([]int, 10)
		// round target and convert to int
		initialTarget := int(o.Targets[0] + math.Copysign(0.5, o.Targets[0]))
		// this target needs to be split into nine 0s and one 1 at the index
		targetList[initialTarget] = 1
		var targetString strings.Builder
		for i, t := range targetList {
			if i != 0 {
				targetString.WriteString(" ")
			}
			targetString.WriteString(strconv.Itoa(t))
		}
		newLines[i] = fmt.Sprintf("%s%s", inputString.String(), targetString.String())
	}
	return newLines, nil
}
