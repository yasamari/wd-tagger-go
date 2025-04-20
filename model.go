package wdtagger

import (
	"encoding/csv"
	"os"

	ort "github.com/yalue/onnxruntime_go"
)

func openModel(modelPath string) (*ort.DynamicAdvancedSession, int, int, error) {
	inputInfo, outputInfo, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		return nil, 0, 0, err
	}

	opts, err := setupSessionOptions()
	if err != nil {
		return nil, 0, 0, err
	}
	s, err := ort.NewDynamicAdvancedSession(modelPath, []string{inputInfo[0].Name}, []string{outputInfo[0].Name}, opts)

	return s, int(inputInfo[0].Dimensions[1]), int(outputInfo[0].Dimensions[1]), err
}

func openTags(tagsPath string) ([][]string, error) {
	f, err := os.Open(tagsPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		return nil, err
	}

	return records[1:], nil
}

func setupSessionOptions() (*ort.SessionOptions, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}

	cudaOpts, err := ort.NewCUDAProviderOptions()
	if err == nil {
		err = opts.AppendExecutionProviderCUDA(cudaOpts)
		if err == nil {
			return opts, nil
		}
	}

	return opts, nil
}
