---
title: BinSight
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Sorts a photo of rubbish into a bin, or says UNSURE
models:
  - google/siglip2-so400m-patch16-256
datasets:
  - garythung/trashnet
  - omasteam/waste-garbage-management-dataset
  - viola77data/recycling-dataset
  - Tatyr2210/waste-classification-vit
---

# BinSight

A campus smart bin that sorts rubbish into PLASTIC, PAPER, METAL or ORGANIC, and sends anything
it is not sure about to UNSURE for a person. Take a photo of one item and it shows where it would
go and why.

Code, training and the accuracy report: https://github.com/Shriram-Shekade/BinSight

This Space installs the `binsight` package from GitHub (tag `v2`), which includes the trained
classifier. Example photos are from [TrashNet](https://github.com/garythung/trashnet) (MIT).
