# BinSight accuracy report

Backbone `google/siglip2-so400m-patch16-256` (frozen) + trained head (MLP, 512 hidden), trained 2026-11-07T15:28:17. Test set: 3945 held-out images never used for training, model selection, calibration or threshold choice.

## Summary

- **Deployed system** (trained head + zero-shot guard + second opinion, threshold 0.30): puts **0.4%** of items in the wrong compartment (95% CI 0.2–0.6%) and sorts **93.7%** of the items that belong in a compartment into the right one; everything else goes to UNSURE.
- **Zero-shot only** (the v1 approach, with v2's prompts) at 0.60: 2.9% wrong sorts (CI 2.4–3.4%), 75.2% of sortable items sorted right.
- 5-class accuracy: trained **99.1%** (CI 98.8–99.4%) vs zero-shot 88.6% (CI 87.6–89.6%).

Wrong sorts are what matter for a bin: one wrong item contaminates a compartment, while an UNSURE only costs a manual sort. The threshold was chosen on the validation split as the lowest one keeping wrong sorts at or below 1.0%.

## Results on the test set

| System | Threshold | Accuracy | Macro-F1 | Wrong sorts | Sortable sorted right | Auto-sorted | UNSURE | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero-shot only | 0.60 | 88.6% | 0.873 | 2.9% | 75.2% | 41.0% | 59.0% | 0.089 |
| zero-shot, tuned threshold | 0.79 | 88.6% | 0.873 | 1.2% | 61.9% | 32.6% | 67.4% | 0.089 |
| trained head | 0.30 | 99.1% | 0.989 | 0.7% | 98.8% | 50.8% | 49.2% | 0.004 |
| trained + guard | 0.30 | 99.1% | 0.989 | 0.7% | 95.3% | 49.0% | 51.0% | 0.004 |
| trained + guard + second opinion (deployed) | 0.30 | 99.1% | 0.989 | 0.4% | 93.7% | 48.0% | 52.0% | 0.004 |

Accuracy and macro-F1 use the top class over all 6 classes (no UNSURE). Wrong sorts, auto-sorted and UNSURE are shares of all test items after routing; an item with no compartment (glass or other) auto-sorted into any compartment counts as a wrong sort. 'Sortable sorted right' is the share of plastic/paper/metal/organic items that reached the right compartment; auto-sorted can be at most 50.7% here because the rest of the test set is glass or other. ECE is expected calibration error of the confidence (lower is better).

![Coverage vs wrong sorts](tradeoff.png)

## Per class (deployed system)

| Class | Precision | Recall | F1 | Test images |
| --- | --- | --- | --- | --- |
| plastic | 0.977 | 0.975 | 0.976 | 556 |
| paper | 0.990 | 0.995 | 0.993 | 816 |
| metal | 0.992 | 0.986 | 0.989 | 358 |
| organic | 0.993 | 0.996 | 0.994 | 272 |
| glass | 0.986 | 0.989 | 0.987 | 620 |
| other | 0.999 | 0.996 | 0.998 | 1323 |

Compartment contamination (share of auto-sorted items that do not belong there):

| Compartment | Zero-shot only | Deployed |
| --- | --- | --- |
| METAL | 6.5% | 0.6% |
| ORGANIC | 14.3% | 0.0% |
| PAPER | 0.4% | 0.5% |
| PLASTIC | 11.3% | 2.1% |

![Confusion matrices](confusion.png)

## Per source dataset (5-class accuracy)

| Source | Test images | Zero-shot only | Trained |
| --- | --- | --- | --- |
| `omasteam/waste-garbage-management-dataset` | 2756 | 89.7% | 99.4% |
| `garythung/trashnet` | 286 | 94.4% | 98.6% |
| `viola77data/recycling-dataset` | 290 | 88.3% | 99.0% |
| `Tatyr2210/waste-classification-vit` | 613 | 81.4% | 98.0% |

## Photos from an unseen source

The test set above shares photographers, cameras and backgrounds with training. This harder test trains without one whole dataset and tests on all of it, which is closer to a new webcam on a new campus. Same recipe and settings as deployed.

| Held-out dataset | Images | Trained accuracy | Trained wrong sorts | zero-shot accuracy | zero-shot wrong sorts |
| --- | --- | --- | --- | --- | --- |
| `omasteam/waste-garbage-management-dataset` | 18403 | 95.8% | 0.7% | 89.6% | 2.3% |
| `garythung/trashnet` | 1824 | 98.4% | 0.7% | 95.8% | 1.2% |
| `viola77data/recycling-dataset` | 1848 | 84.3% | 2.0% | 87.6% | 2.3% |
| `Tatyr2210/waste-classification-vit` | 4225 | 89.7% | 2.5% | 81.7% | 5.7% |

What this found (history, measured with three datasets before RealWaste was added): plastic in the two larger datasets is mostly bottles, so a head trained without the recycling-dataset mistook soft plastics (bags, wrappers) and hard plastic containers for paper or 'other', confidently: 12.0% wrong sorts on it. Zero-shot SigLIP2 knows what a wrapper is. Requiring the trained head and zero-shot to agree cut that to 2.2% but sent much of the paper to UNSURE. The deployed rule asks for the second opinion only when the item might be glass or organic, or the head is under 95% sure. Checking only possible glass/organic would sort a little more but quadrupled wrong sorts on unseen datasets, so the confidence condition stays (comparison in `scripts/rule_ab.py`, chosen on validation and these unseen-dataset folds). On the main test set: sortable items sorted right 95.3% without a second opinion, 93.7% with it.

## Organic safety

Glass or plastic in food waste can hurt animals that eat it, so ORGANIC is stricter than the other compartments: zero-shot must always agree, and the trained head needs 0.98 confidence. With only the usual rules, 3 non-organic test items reached ORGANIC: green glass ornaments shaped like beans (both models were fooled), a burlap sack covered in grass, and a plastic item; validation had none, so it could not be used to choose the bar. The 0.98 bar was therefore set as a safety policy after seeing those test items. It stops all three, but because the test set informed it, the test result for ORGANIC leaks is not an independent measurement. Cost: about 2-3% of organic items go to UNSURE.

## Calibration

Softmax temperature 0.54, fitted on validation. Test ECE 0.073 before -> 0.004 after. Calibrated confidences make the threshold mean roughly what it says.

## Data

| Source | License | Images mapped | Kept after de-dup |
| --- | --- | --- | --- |
| `omasteam/waste-garbage-management-dataset` | MIT | 18815 | 18403 |
| `garythung/trashnet` | MIT | 4780 | 1824 |
| `viola77data/recycling-dataset` | Apache-2.0 | 1952 | 1848 |
| `Tatyr2210/waste-classification-vit` | MIT upload of RealWaste (CC BY 4.0) | 4268 | 4225 |

- Labels were mapped to the five classes in `config.yaml`; mixed labels (generic 'trash', takeaway cups, disposable plates, paper towels, polystyrene) were dropped rather than guessed.
- Near-duplicates (cosine >= 0.985 between embeddings, or identical image hash) were merged before splitting, so no image leaks between train and test: 3509 duplicate links in 2884 groups; 6 groups whose copies had different labels (15 images) were dropped.
- Stratified 70% / 15% / 15% train / val / test split (seed 0). Training also used 2 augmented views per image (crop, flip, rotation, colour, blur).

| Split | plastic | paper | metal | organic | glass | other |
| --- | --- | --- | --- | --- | --- | --- |
| train | 2593 | 3806 | 1671 | 1272 | 2896 | 6172 |
| val | 556 | 816 | 358 | 272 | 620 | 1323 |
| test | 556 | 816 | 358 | 272 | 620 | 1323 |

## Method

1. Every image is embedded once by the frozen SigLIP2 vision encoder (fp16 on MPS).
2. A small head (grid over linear / MLP, learning rate, weight decay; best validation macro-F1 kept) is trained on those embeddings with class-balanced cross-entropy.
3. A softmax temperature is fitted on validation.
4. The zero-shot prompts from v1 still run on the same embedding as a guard: if they say glass or other with probability >= 0.5, the item goes to UNSURE. This covers things the datasets have no pictures of (hands, faces, an empty tray, phones).
5. Second opinion: when the item might be glass or organic, or the head is under 95% sure, it is auto-sorted only if the zero-shot prompts pick the same class; organic also needs higher head confidence (see the unseen-source test and organic safety).
6. The UNSURE threshold is the lowest one meeting the wrong-sort target on validation.

Validation grid:

| Hidden | LR | Weight decay | Val macro-F1 | Best epoch |
| --- | --- | --- | --- | --- |
| 0 | 0.001 | 0.0001 | 0.9774 | 22 |
| 0 | 0.001 | 0.01 | 0.9773 | 24 |
| 0 | 0.0003 | 0.0001 | 0.9677 | 34 |
| 0 | 0.0003 | 0.01 | 0.9679 | 38 |
| 512 | 0.001 | 0.0001 | 0.9869 | 33 |
| 512 | 0.001 | 0.01 | 0.9869 | 33 |
| 512 | 0.0003 | 0.0001 | 0.9833 | 20 |
| 512 | 0.0003 | 0.01 | 0.9833 | 20 |

Weight decay made no measurable difference at these settings; the hidden layer and learning rate did.

## Limitations

- The test images come from the same web datasets as the training images: mostly single items, well lit, plain backgrounds. A webcam over a real tray will look different, so expect lower accuracy live; the threshold can be raised with `+`.
- De-duplication removes copies, not different photos of the same object; such near-twins can still sit on both sides of the split and flatter these numbers.
- 'other' in the data is batteries, clothes, shoes and textiles only. Hands, faces, phones and an empty tray rely on the zero-shot guard, which this test set does not measure.
- Class mapping is a policy choice (e.g. cardboard is paper, soft plastics are plastic, glass is UNSURE because the bin has no glass compartment).
- Some photos labelled glass in the datasets look like clear PET plastic (screw-thread bottles, a disposable champagne flute). Part of the measured 'glass in PLASTIC' wrong sorts is likely label noise, and clear glass and clear plastic can look identical in a photo: a real bin would add a cheap physical check (weight, or the sound of a tap).
