import os
from app.image_cleanup import remove_orphans


def test_cleanup_preserves_referenced_recent_and_unrelated_files(tmp_path):
    referenced = 'a' * 64 + '.jpg'
    old = 'b' * 64 + '.jpg'
    recent = 'c' * 64 + '.jpg'
    for name in [referenced, old, recent, 'unrelated.txt']:
        path = tmp_path / name
        path.write_bytes(b'image')
        os.utime(path, (10, 10))
    os.utime(tmp_path / recent, (100, 100))
    assert remove_orphans(tmp_path, {referenced}, 50) == 1
    assert not (tmp_path / old).exists()
    assert {p.name for p in tmp_path.iterdir()} == {referenced, recent, 'unrelated.txt'}


def test_missing_storage(tmp_path):
    assert remove_orphans(tmp_path / 'missing', set(), 50) == 0
