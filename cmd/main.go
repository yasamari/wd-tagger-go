package main

import (
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"os"
	"path/filepath"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
	wdtagger "github.com/yasamari/wd-tagger-go"
	_ "golang.org/x/image/webp"
)

var (
	inputDir           = flag.String("input", "images", "タグをつける画像が含まれるディレクトリのパス")
	batchSize          = flag.Int("batch-size", 16, "バッチサイズ")
	generalThreshold   = flag.Float64("general-threshold", 0.35, "一般タグのしきい値")
	characterThreshold = flag.Float64("character-threshold", 0.85, "キャラクタータグのしきい値")
	generalMCut        = flag.Bool("general-mcut", false, "一般タグでMCutを使用するかどうか")
	characterMCut      = flag.Bool("character-mcut", false, "キャラクタータグでMCutを使用するかどうか")
	model              = flag.String("model", string(wdtagger.WdSwinV2TaggerV3), "モデルのリポジトリ")
	libraryPath        = flag.String("library-path", "onnxruntime-linux-x64-gpu-1.21.0/lib/libonnxruntime.so", "onnxruntimeのパス")
)

func main() {
	flag.Parse()
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ort.SetSharedLibraryPath(*libraryPath)

	err := ort.InitializeEnvironment()
	if err != nil {
		return err
	}

	tagger, err := wdtagger.NewWdTagger(wdtagger.TaggerRepo(*model))
	if err != nil {
		return err
	}
	defer tagger.Destroy()

	entries, err := os.ReadDir(*inputDir)
	if err != nil {
		return err
	}

	var images []image.Image
	var fileNames []string

	predict := func() error {
		result, err := tagger.Predict(images, float32(*generalThreshold), float32(*characterThreshold), *generalMCut, *characterMCut)
		if err != nil {
			return err
		}

		for i, tags := range result {
			path := filepath.Join(*inputDir, fmt.Sprintf("%s.txt", fileNames[i]))
			if err := writeTags(path, tags); err != nil {
				return err
			}
		}
		return nil
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		filePath := filepath.Join(*inputDir, entry.Name())
		f, err := os.Open(filePath)
		if err != nil {
			return err
		}
		img, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			if err == image.ErrFormat {
				continue
			}
			return err
		}
		images = append(images, img)
		fileNames = append(fileNames, entry.Name())

		if len(images) >= *batchSize {
			if err := predict(); err != nil {
				return err
			}
			images = []image.Image{}
			fileNames = []string{}
		}
	}
	if len(images) > 0 {
		if err := predict(); err != nil {
			return err
		}
	}
	return nil
}

func writeTags(path string, tags *wdtagger.Tags) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	t := []string{tags.Rating}
	t = append(t, tags.CharacterTags...)
	t = append(t, tags.GeneralTags...)

	_, err = f.WriteString(strings.Join(t, ","))
	if err != nil {
		return err
	}
	return nil
}
