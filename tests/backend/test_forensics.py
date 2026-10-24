import numpy as np
import pytest
from PIL import Image

from app.ml.detector import VideoOptions
from app.ml.forensics.image_report import image_forensic_signals
from app.ml.forensics.video_report import video_forensic_signals
from app.ml.forensics_detector import ForensicsDetector
from app.ml.preprocessing import PreparedImage, load_image
from helpers import FakeDetector, write_clip


def jpeg_file(tmp_path, quality=75, name="a.jpg", **save_kwargs):
    pixels = np.random.default_rng(0).integers(0, 256, (96, 96, 3), dtype=np.uint8)
    path = tmp_path / name
    Image.fromarray(pixels).save(path, quality=quality, **save_kwargs)
    return path


def prepared(array, fmt="png"):
    array.flags.writeable = False
    return PreparedImage(rgb=array, format=fmt, mode="RGB", has_alpha=False, has_exif=False)


# ---- JPEG history and EXIF ---------------------------------------------------------------


@pytest.mark.parametrize("quality", [30, 50, 75, 90])
def test_jpeg_quality_is_recovered_from_the_quantization_tables(tmp_path, quality):
    assert load_image(jpeg_file(tmp_path, quality)).jpeg_quality_estimate == quality


def test_non_standard_tables_give_no_estimate(tmp_path):
    path = jpeg_file(tmp_path, qtables=[[40] * 64, [40] * 64])
    assert load_image(path).jpeg_quality_estimate is None


def test_png_has_no_jpeg_estimate(tmp_path):
    path = tmp_path / "a.png"
    Image.new("RGB", (64, 64)).save(path)
    assert load_image(path).jpeg_quality_estimate is None


def test_only_the_software_tag_is_read_from_exif(tmp_path):
    exif = Image.Exif()
    exif[0x0131] = "Test Editor 1.0"
    exif[0x010F] = "SomeCameraMaker"
    prepared_image = load_image(jpeg_file(tmp_path, exif=exif))
    assert prepared_image.exif_software == "Test Editor 1.0"
    assert prepared_image.has_exif is True
    assert not hasattr(prepared_image, "exif_make")


# ---- image signals -----------------------------------------------------------------------


def test_image_signals_are_context_only_and_forensic(tmp_path):
    signals = image_forensic_signals(load_image(jpeg_file(tmp_path, 75)))
    keys = {s.key for s in signals}
    assert {"forensic_dimensions", "forensic_color_stats", "forensic_channel_noise",
            "forensic_jpeg_quality"} <= keys
    assert all(not s.scored and s.source.value == "forensic" for s in signals)
    jpeg = next(s for s in signals if s.key == "forensic_jpeg_quality")
    assert jpeg.value == 75.0


def test_png_gets_no_jpeg_signal_and_flat_images_do_not_crash():
    flat = prepared(np.full((64, 64, 3), 128, np.uint8))
    keys = {s.key for s in image_forensic_signals(flat)}
    assert "forensic_jpeg_quality" not in keys
    assert "forensic_channel_noise" not in keys  # flat channels: the ratio is meaningless
    assert "forensic_dimensions" in keys


def test_software_signal_appears_only_when_the_tag_exists(tmp_path):
    exif = Image.Exif()
    exif[0x0131] = "Test Editor 1.0"
    with_tag = image_forensic_signals(load_image(jpeg_file(tmp_path, exif=exif, name="b.jpg")))
    without = image_forensic_signals(load_image(jpeg_file(tmp_path, name="c.jpg")))
    assert "forensic_exif_software" in {s.key for s in with_tag}
    assert "forensic_exif_software" not in {s.key for s in without}


def test_dimension_signal_never_calls_a_size_evidence():
    signal = image_forensic_signals(prepared(np.zeros((64, 128, 3), np.uint8)))[0]
    assert "not evidence" in signal.detail.lower()


# ---- the wrapper -------------------------------------------------------------------------


def test_wrapper_adds_signals_without_changing_the_score():
    image = prepared(np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8))
    inner = FakeDetector(fixed=0.42)
    wrapped = ForensicsDetector(inner)
    before, after = inner.predict_image(image), wrapped.predict_image(image)
    assert after.ai_probability == before.ai_probability
    assert len(after.signals) > len(before.signals)
    assert wrapped.info == inner.info


def test_wrapper_skips_video_frames():
    frame = np.zeros((64, 64, 3), np.uint8)
    frame.flags.writeable = False
    image = PreparedImage(rgb=frame, format="video_frame", mode="RGB", has_alpha=False, has_exif=False)
    assert len(ForensicsDetector(FakeDetector(fixed=0.5)).predict_image(image).signals) == 0


def test_a_failing_forensic_never_fails_the_analysis(monkeypatch):
    def explode(image):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.ml.forensics_detector.image_forensic_signals", explode)
    image = prepared(np.zeros((64, 64, 3), np.uint8))
    assert ForensicsDetector(FakeDetector(fixed=0.3)).predict_image(image).ai_probability == 0.3


# ---- video -------------------------------------------------------------------------------


def test_video_signals_cover_consistency_and_faces(tmp_path):
    clip = write_clip(tmp_path / "cut.avi", [20] * 17 + [200] * 23)
    signals = {s.key: s for s in video_forensic_signals(clip, [2, 10, 20, 30])}
    assert signals["temporal_consistency"].value > 0
    assert signals["face_presence"].value in (0.0, None)  # a flat clip has no faces
    assert all(not s.scored for s in signals.values())


def test_wrapper_adds_video_signals_after_the_frame_wise_result(tmp_path):
    clip = write_clip(tmp_path / "ramp.avi", [i * 5 for i in range(40)])
    options = VideoOptions(max_frames=8, flag_threshold=0.65, max_duration_sec=60)
    prediction = ForensicsDetector(FakeDetector()).predict_video(clip, options)
    keys = {s.key for s in prediction.signals}
    assert {"frame_score_aggregate", "temporal_consistency", "face_presence"} <= keys


# ---- through the API ---------------------------------------------------------------------


def test_api_results_gain_forensic_signals_but_keep_the_same_score(make_client, jpeg_bytes):
    files = {"file": ("photo.jpg", jpeg_bytes, "image/jpeg")}
    plain = make_client().post("/api/detect/image", files=files).json()
    forensic = make_client(forensics_enabled=True).post("/api/detect/image", files=files).json()
    assert forensic["ai_probability"] == plain["ai_probability"]
    keys = {s["key"] for s in forensic["explanation"]}
    assert {"forensic_dimensions", "forensic_jpeg_quality"} <= keys
    assert "forensic_dimensions" not in {s["key"] for s in plain["explanation"]}