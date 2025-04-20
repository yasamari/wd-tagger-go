package wdtagger

import (
	"image"
	"image/color"
	"sort"
	"sync"

	"slices"

	"github.com/gomlx/go-huggingface/hub"
	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/image/draw"
)

type WdTagger struct {
	session       *ort.DynamicAdvancedSession
	size, classes int
	tags          [][]string
}

type Tags struct {
	GeneralTags   []string
	CharacterTags []string
	Rating        string
}

type TaggerRepo string

const (
	WdConvnextTaggerV3   TaggerRepo = "SmilingWolf/wd-convnext-tagger-v3"
	WdEVA02LargeTaggerV3 TaggerRepo = "SmilingWolf/wd-eva02-large-tagger-v3"
	WdSwinV2TaggerV3     TaggerRepo = "SmilingWolf/wd-swinv2-tagger-v3"
	WdConvnextTaggerV2   TaggerRepo = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
	WdConvnextV2TaggerV2 TaggerRepo = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
	WdMoatTaggerV2       TaggerRepo = "SmilingWolf/wd-v1-4-moat-tagger-v2"
	WdSwinv2TaggerV2     TaggerRepo = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
	WdVitTaggerV2        TaggerRepo = "SmilingWolf/wd-v1-4-vit-tagger-v2"
	WdVitLargeTaggerV3   TaggerRepo = "SmilingWolf/wd-vit-large-tagger-v3"
	WdVitTaggerV3        TaggerRepo = "SmilingWolf/wd-vit-tagger-v3"

	IdolSankakuEVA02LargeTaggerV1 TaggerRepo = "deepghs/idolsankaku-eva02-large-tagger-v1"
	IdolSankakuSwinV2TaggerV1     TaggerRepo = "deepghs/idolsankaku-swinv2-tagger-v1"
)

func NewWdTagger(repoName TaggerRepo) (*WdTagger, error) {
	repo := hub.New(string(repoName))
	paths, err := repo.DownloadFiles("model.onnx", "selected_tags.csv")
	if err != nil {
		return nil, err
	}
	modelPath := paths[0]
	tagsPath := paths[1]

	session, size, classes, err := openModel(modelPath)
	if err != nil {
		return nil, err
	}
	tags, err := openTags(tagsPath)
	if err != nil {
		return nil, err
	}

	return &WdTagger{
		session: session,
		size:    size,
		classes: classes,
		tags:    tags,
	}, nil
}

func (t *WdTagger) Destroy() error {
	return t.session.Destroy()
}

func (t *WdTagger) Predict(
	images []image.Image,
	generalThreshold,
	characterThreshold float32,
	generalMCut,
	characterMCut bool,
) ([]*Tags, error) {
	var wg sync.WaitGroup
	var mu sync.Mutex
	preprocessed := map[int][]float32{}
	for idx, img := range images {
		wg.Add(1)
		go func() {
			defer wg.Done()
			processed := t.preprocess(img)

			mu.Lock()
			defer mu.Unlock()
			preprocessed[idx] = processed
		}()
	}
	wg.Wait()
	input := make([]float32, 0, len(images)*t.size*t.size*3)
	for idx := range len(preprocessed) {
		input = append(input, preprocessed[idx]...)
	}

	batch := int64(len(images))

	inputTensor, err := ort.NewTensor(ort.NewShape(batch, int64(t.size), int64(t.size), 3), input)
	if err != nil {
		return nil, err
	}
	defer inputTensor.Destroy()

	outputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(batch, int64(t.classes)))
	if err != nil {
		return nil, err
	}
	defer outputTensor.Destroy()

	err = t.session.Run([]ort.ArbitraryTensor{inputTensor}, []ort.ArbitraryTensor{outputTensor})
	if err != nil {
		return nil, err
	}

	output := outputTensor.GetData()

	resultMap := map[int]*Tags{}
	for idx := range int(batch) {
		scores := output[:t.classes]
		wg.Add(1)
		go func() {
			defer wg.Done()
			tags := t.postprocess(scores, generalThreshold, characterThreshold, generalMCut, characterMCut)
			mu.Lock()
			defer mu.Unlock()
			resultMap[idx] = tags
		}()
		output = output[t.classes:]
	}
	wg.Wait()

	result := make([]*Tags, 0, len(images))
	for idx := range len(images) {
		result = append(result, resultMap[idx])
	}
	return result, nil
}

func (t *WdTagger) preprocess(img image.Image) []float32 {
	bicubic := &draw.Kernel{
		Support: 2,
		At: func(t float64) float64 {
			if t < 0 {
				t = -t
			}
			if t < 1 {
				return (1.5*t-2.5)*t*t + 1
			}
			if t < 2 {
				return ((-0.5*t+2.5)*t-4)*t + 2
			}
			return 0
		},
	}

	srcBounds := img.Bounds()

	srcW := srcBounds.Dx()
	srcH := srcBounds.Dy()

	maxDim := max(srcW, srcH)
	canvas := image.NewRGBA(image.Rect(0, 0, maxDim, maxDim))
	draw.Draw(canvas, canvas.Bounds(), &image.Uniform{color.White}, image.Point{}, draw.Src)

	offsetX := (maxDim - srcW) / 2
	offsetY := (maxDim - srcH) / 2
	draw.Draw(canvas, image.Rect(offsetX, offsetY, offsetX+srcW, offsetY+srcH), img, image.Point{}, draw.Over)

	resized := image.NewRGBA(image.Rect(0, 0, t.size, t.size))
	bicubic.Scale(resized, resized.Bounds(), canvas, canvas.Bounds(), draw.Over, nil)

	result := make([]float32, 0, t.size*t.size*3)
	for y := range t.size {
		for x := range t.size {
			r, g, b, _ := resized.At(x, y).RGBA()

			result = append(result, float32(b/257.0))
			result = append(result, float32(g/257.0))
			result = append(result, float32(r/257.0))
		}
	}
	return result
}

func (t *WdTagger) postprocess(
	scores []float32,
	generalThreshold,
	characterThreshold float32,
	generalMCut,
	characterMCut bool,
) *Tags {
	var generalScores, characterScores []float32
	var generalNames, characterNames []string

	var ratingScore float32
	var ratingName string

	for i, score := range scores {
		tag := t.tags[i]
		category := tag[2]
		name := tag[1]
		if category == "0" {
			generalScores = append(generalScores, score)
			generalNames = append(generalNames, name)
			continue
		}
		if category == "4" {
			characterScores = append(characterScores, score)
			characterNames = append(characterNames, name)
			continue
		}
		if category == "9" && score > ratingScore {
			ratingName = name
			ratingScore = score
			continue
		}
	}

	if generalMCut {
		generalThreshold = t.mcutThreshold(generalScores)
	}
	if characterMCut {
		characterThreshold = t.mcutThreshold(characterScores)
	}
	return &Tags{
		GeneralTags:   t.filterTags(generalScores, generalNames, generalThreshold),
		CharacterTags: t.filterTags(characterScores, characterNames, characterThreshold),
		Rating:        ratingName,
	}
}

func (t *WdTagger) filterTags(scores []float32, names []string, threshold float32) []string {
	var result []string
	for idx, tagName := range names {
		score := scores[idx]
		if score < threshold {
			continue
		}
		result = append(result, tagName)
	}
	return result
}

func (t *WdTagger) mcutThreshold(scores []float32) float32 {
	sorted := slices.Clone(scores)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i] > sorted[j]
	})

	difs := make([]float32, len(sorted)-1)
	for i := range difs {
		difs[i] = sorted[i] - sorted[i+1]
	}

	maxDiff := difs[0]
	idx := 0
	for i := 1; i < len(difs); i++ {
		if difs[i] > maxDiff {
			maxDiff = difs[i]
			idx = i
		}
	}
	return (sorted[idx] + sorted[idx+1]) / 2
}
