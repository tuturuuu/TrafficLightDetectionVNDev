"""Check mapping, coordinate normalization and safe conversion of VisDrone labels."""
import importlib.util
from pathlib import Path

import pytest
import yaml
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "visdrone_converter", Path(__file__).resolve().parents[1] / "scripts/data/convert_visdrone_to_yolo.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def test_categories_ignore_and_normalization():
    text, stats, classes = converter.convert_annotation(
        "10,20,30,40,1,1,0,0\n0,0,100,200,1,10,0,0\n"
        "0,0,10,10,0,4,0,0\n0,0,10,10,1,11,0,0\n"
        "0,0,10,10,1,0,0,0\n", 100, 200)
    rows = [list(map(float, line.split())) for line in text.splitlines()]
    assert rows == [[0, .25, .2, .3, .2], [9, .5, .5, 1, 1]]
    assert stats["ignored_boxes"] == 3
    assert classes == {0: 1, 9: 1}


def test_clip_and_reject_degenerate_boxes():
    text, stats, _ = converter.convert_annotation(
        "-10,180,30,40,1,4,0,0\n0,0,0,10,1,1,0,0\n"
        "110,0,10,10,1,1,0,0\n", 100, 200)
    assert list(map(float, text.split())) == [3, .1, .95, .2, .1]
    assert stats["clipped_boxes"] == 1
    assert stats["invalid_boxes"] == 2


@pytest.mark.parametrize("row", ["1,2,3", "nan,0,1,1,1,1,0,0",
                                     "0,0,1,1,1,12,0,0", "0,0,1,1,1,1.5,0,0"])
def test_reject_malformed_annotations(row):
    with pytest.raises(ValueError):
        converter.convert_annotation(row, 100, 200)


def test_dataset_matches_all_splits_and_preserves_existing_labels(tmp_path):
    for split, raw in converter.RAW_SPLITS.items():
        images = tmp_path / "images" / split
        annotations = tmp_path / "raw" / raw / "annotations"
        images.mkdir(parents=True)
        annotations.mkdir(parents=True)
        Image.new("RGB", (100, 200)).save(images / "sample.jpg")
        (annotations / "sample.txt").write_text("10,20,30,40,1,1,0,0\n")
    report = converter.convert_dataset(tmp_path)
    assert all(s["written_boxes"] == 1 for s in report["splits"].values())
    data = yaml.safe_load((tmp_path / "data.yaml").read_text())
    assert data["test"] == "images/test"
    assert data["names"][9] == "motor"
    assert report == converter.convert_dataset(tmp_path)
    label = tmp_path / "labels/train/sample.txt"
    label.write_text("user-edited\n")
    with pytest.raises(FileExistsError):
        converter.convert_dataset(tmp_path)
    assert label.read_text() == "user-edited\n"
