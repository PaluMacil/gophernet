package main

import (
	"bufio"
	"fmt"
	"os"
)

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

func normalize(min, max, input float64) float64 {
	numerator := input - min
	rng := max - min
	return numerator / rng
}

func normalize2(min, max, input float64) float64 {
	avg := (min + max) / 2
	rng := (max - min) / 2
	return (input - avg) / rng
}
